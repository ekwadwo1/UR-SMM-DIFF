#!/usr/bin/env python3
"""
physics/corruption_operator.py — k-Space Physics Corruption Operator (Phase 3)
===============================================================================

Paper Eq. (4), Section 4.1:
    x_phys_0 = P(x_0; ξ),   ξ ~ p(ξ)

Implements four structured MRI artifact types operating in k-space:
  (a) Line-wise motion phase ramps
  (b) Nyquist ghosting (alternating-line phase+amplitude modulation)
  (c) Variable-density Cartesian undersampling
  (d) Complex Gaussian readout noise

CRITICAL DESIGN:
  1. Called TWICE per training volume with INDEPENDENT ξ:
       y      = P(x₀; ξ_obs)     → denoiser conditioning (Eq. 2: z^obs = E(y))
       x_phys = P(x₀; ξ_anchor)  → forward trajectory anchor (Eq. 5)
     ξ_obs ⊥ ξ_anchor prevents information leakage.

  2. Each contrast (C=4) corrupted INDEPENDENTLY in k-space.
     Output shape == input shape: [B, C, H, W, D].

  3. NOT differentiable — never in the backward graph during diffusion training.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ======================================================================= #
#  Individual artifact functions                                           #
# ======================================================================= #

def apply_motion_phase_ramps(
    kspace: torch.Tensor, pct: float, alpha: float,
) -> torch.Tensor:
    """
    Artifact (a) — Line-wise motion phase ramps (Section 4.1).

    Simulates inter-line rigid-body motion by applying random phase offsets
    to a fraction of phase-encode lines (dim=1, W axis) in k-space.

    Parameters
    ----------
    kspace : complex Tensor [H, W, D]  (single contrast, already FFT'd)
    pct    : fraction of PE lines to corrupt   (0.05 – 0.2)
    alpha  : phase severity                    (0.05 – 0.3)

    Returns
    -------
    kspace : corrupted complex Tensor [H, W, D]
    """
    H, W, D = kspace.shape
    device = kspace.device
    n_corrupt = max(1, int(W * pct))

    # Random subset of phase-encode lines
    indices = torch.randperm(W, device=device)[:n_corrupt]

    # Per-line random phase: φ_n ~ Uniform(-π·α, π·α)
    phi = (2.0 * torch.rand(n_corrupt, device=device) - 1.0) * (math.pi * alpha)

    # K[..., n, :] *= exp(i·φ_n)  — phase-encode = dim 1
    phase = torch.exp(1j * phi)                          # [n_corrupt]
    kspace[:, indices, :] = kspace[:, indices, :] * phase[None, :, None]

    return kspace


def apply_nyquist_ghosting(
    kspace: torch.Tensor, amplitude: float, phase: float,
) -> torch.Tensor:
    """
    Artifact (b) — Nyquist ghosting (Section 4.1).

    Alternating-line phase+amplitude modulation along phase-encode axis:
      even lines: K[..., 2n, :]   *= A · exp(i·φ)
      odd  lines: K[..., 2n+1, :] *= (1/A) · exp(-i·φ)

    Parameters
    ----------
    kspace    : complex Tensor [H, W, D]
    amplitude : A ~ Uniform(0.8, 1.2)
    phase     : φ ~ Uniform(-0.1, 0.1) radians

    Returns
    -------
    kspace : corrupted complex Tensor [H, W, D]
    """
    device = kspace.device
    even_mod = amplitude * torch.exp(torch.tensor(1j * phase, device=device))
    odd_mod = (1.0 / amplitude) * torch.exp(torch.tensor(-1j * phase, device=device))

    kspace[:, 0::2, :] = kspace[:, 0::2, :] * even_mod
    kspace[:, 1::2, :] = kspace[:, 1::2, :] * odd_mod

    return kspace


def apply_undersampling(
    kspace: torch.Tensor, accel_factor: float, acs_fraction: float = 0.08,
) -> torch.Tensor:
    """
    Artifact (c) — Variable-density Cartesian undersampling (Section 4.1).

    Polynomial variable-density mask along phase-encode axis (dim=1).
    Always retains 8% centre k-space (ACS region). Zero-filled IFFT.

    Parameters
    ----------
    kspace       : complex Tensor [H, W, D]
    accel_factor : R ~ Uniform(2, 6)
    acs_fraction : fraction of centre PE lines always kept (default 0.08)

    Returns
    -------
    kspace : zero-filled undersampled complex Tensor [H, W, D]
    """
    H, W, D = kspace.shape
    device = kspace.device

    n_acs = max(1, int(W * acs_fraction))
    acs_lo = (W - n_acs) // 2
    acs_hi = acs_lo + n_acs

    n_target = max(1, int(W / accel_factor))
    n_outer = max(0, n_target - n_acs)

    # Variable-density probability: p(k) ∝ (1 - |k - centre|/(W/2))³
    coords = torch.arange(W, device=device, dtype=torch.float32)
    dist = (coords - W / 2.0).abs() / (W / 2.0)
    prob = (1.0 - dist).clamp(min=0.01).pow(3)
    prob[acs_lo:acs_hi] = 0.0                     # exclude ACS from outer sampling
    total = prob.sum()
    if total > 0:
        prob = prob / total

    # Sample outer lines
    if n_outer > 0 and total > 0:
        outer_idx = torch.multinomial(prob, min(n_outer, W - n_acs), replacement=False)
    else:
        outer_idx = torch.tensor([], dtype=torch.long, device=device)

    # Build binary mask [W]
    mask = torch.zeros(W, device=device)
    mask[acs_lo:acs_hi] = 1.0
    if outer_idx.numel() > 0:
        mask[outer_idx] = 1.0

    # Apply along PE dim=1:  [H, W, D] × [1, W, 1]
    kspace = kspace * mask[None, :, None]

    return kspace


def apply_readout_noise(
    kspace: torch.Tensor, sigma_frac: float,
) -> torch.Tensor:
    """
    Artifact (d) — Complex Gaussian readout noise (Section 4.1).

    K += σ · (randn + i·randn) / √2,   σ = sigma_frac × max(|K|)

    Parameters
    ----------
    kspace     : complex Tensor [H, W, D]
    sigma_frac : noise fraction ~ Uniform(0.005, 0.05)

    Returns
    -------
    kspace : noisy complex Tensor [H, W, D]
    """
    peak = kspace.abs().max().clamp(min=1e-12)
    sigma = sigma_frac * peak
    noise = torch.complex(
        torch.randn_like(kspace.real),
        torch.randn_like(kspace.real),
    ) * (sigma / math.sqrt(2.0))
    return kspace + noise


# ======================================================================= #
#  Main class                                                              #
# ======================================================================= #

class PhysicsCorruptionOperator(nn.Module):
    """
    k-Space physics corruption operator  P(x₀; ξ)  — Eq. (4), Section 4.1.

    Simulates structured MRI acquisition artifacts by operating in k-space:
      (a) line-wise motion phase ramps
      (b) Nyquist ghosting
      (c) variable-density Cartesian undersampling
      (d) complex Gaussian readout noise

    Each contrast channel is corrupted independently. Called TWICE per
    training volume with independent ξ to produce y (conditioning) and
    x_phys (forward-trajectory anchor).

    NOT differentiable — never in the backward graph.

    Parameters
    ----------
    artifact_prob : dict
        Per-artifact inclusion probability. Each artifact independently
        sampled. A volume may receive 0–4 artifacts.
    device : str
        Device for k-space FFT operations.
    """

    def __init__(
        self,
        artifact_prob: Optional[Dict[str, float]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.artifact_prob: Dict[str, float] = artifact_prob or {
            "motion": 0.30,
            "ghosting": 0.25,
            "undersampling": 0.25,
            "noise": 0.20,
        }
        self._device = device

    # ------------------------------------------------------------------ #
    #  Parameter sampling   ξ ~ p(ξ)                                      #
    # ------------------------------------------------------------------ #

    def sample_xi(self) -> Dict[str, Any]:
        """
        Sample artifact parameters ξ ~ p(ξ)  (Eq. 4).

        Returns dict with 'artifacts' list + per-artifact param dicts.
        Can be passed to forward_with_params() for deterministic replay.
        """
        xi: Dict[str, Any] = {"artifacts": []}

        if torch.rand(1).item() < self.artifact_prob.get("motion", 0):
            xi["motion"] = {
                "pct":   0.05 + 0.15 * torch.rand(1).item(),    # U(0.05, 0.2)
                "alpha": 0.05 + 0.25 * torch.rand(1).item(),    # U(0.05, 0.3)
            }
            xi["artifacts"].append("motion")

        if torch.rand(1).item() < self.artifact_prob.get("ghosting", 0):
            xi["ghosting"] = {
                "amplitude": 0.8 + 0.4 * torch.rand(1).item(), # U(0.8, 1.2)
                "phase":    -0.1 + 0.2 * torch.rand(1).item(), # U(-0.1, 0.1)
            }
            xi["artifacts"].append("ghosting")

        if torch.rand(1).item() < self.artifact_prob.get("undersampling", 0):
            xi["undersampling"] = {
                "accel_factor": 2.0 + 4.0 * torch.rand(1).item(),  # U(2, 6)
                "acs_fraction": 0.08,
            }
            xi["artifacts"].append("undersampling")

        if torch.rand(1).item() < self.artifact_prob.get("noise", 0):
            xi["noise"] = {
                "sigma_frac": 0.005 + 0.045 * torch.rand(1).item(),  # U(0.005, 0.05)
            }
            xi["artifacts"].append("noise")

        return xi

    # ------------------------------------------------------------------ #
    #  Forward (random ξ)                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply physics corruption with freshly sampled ξ.  Eq. (4).

        Parameters
        ----------
        x : [B, C, H, W, D]  float32, artifact-free mpMRI volume

        Returns
        -------
        x_corrupt : [B, C, H, W, D]  same shape/dtype
        """
        return self.forward_with_params(x, self.sample_xi())

    # ------------------------------------------------------------------ #
    #  Forward (deterministic, given ξ)                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward_with_params(
        self, x: torch.Tensor, xi: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Apply corruption with specified ξ (deterministic replay).

        Parameters
        ----------
        x  : [B, C, H, W, D]
        xi : dict from sample_xi()

        Returns
        -------
        x_corrupt : [B, C, H, W, D]
        """
        B, C, H, W, D = x.shape
        x_out = torch.empty_like(x)
        x_min, x_max = x.min().item(), x.max().item()

        for b in range(B):
            for c in range(C):
                # 3D FFT → k-space
                ksp = torch.fft.fftn(x[b, c], dim=(0, 1, 2))  # [H,W,D] complex

                # Apply artifacts sequentially in sampled order
                for art in xi.get("artifacts", []):
                    if art == "motion":
                        ksp = apply_motion_phase_ramps(
                            ksp, xi["motion"]["pct"], xi["motion"]["alpha"])
                    elif art == "ghosting":
                        ksp = apply_nyquist_ghosting(
                            ksp, xi["ghosting"]["amplitude"], xi["ghosting"]["phase"])
                    elif art == "undersampling":
                        ksp = apply_undersampling(
                            ksp, xi["undersampling"]["accel_factor"],
                            xi["undersampling"]["acs_fraction"])
                    elif art == "noise":
                        ksp = apply_readout_noise(ksp, xi["noise"]["sigma_frac"])

                # IFFT → image domain → magnitude (real-valued)
                x_out[b, c] = torch.fft.ifftn(ksp, dim=(0, 1, 2)).abs()

        # Clamp to input range
        return x_out.clamp(x_min, x_max)

    def extra_repr(self) -> str:
        return f"artifact_prob={self.artifact_prob}"


# ======================================================================= #
#  PSNR utility                                                            #
# ======================================================================= #

def _psnr(clean: torch.Tensor, corrupted: torch.Tensor) -> float:
    """PSNR (dB) between two tensors."""
    mse = (clean - corrupted).pow(2).mean().item()
    if mse < 1e-12:
        return float("inf")
    dr = clean.max().item() - clean.min().item()
    return 10.0 * math.log10(dr ** 2 / mse) if dr > 1e-12 else 0.0


# ======================================================================= #
#  Test suite  (Jupyter-safe — NO argparse)                                #
# ======================================================================= #

def _run_tests(device: str = "auto", save_dir: str = ".") -> None:
    """
    Comprehensive test: each artifact individually, combined, dual-independent
    corruption, per-contrast independence, identity, + PDF figure.

    Parameters
    ----------
    device   : 'auto' | 'cpu' | 'cuda:0' etc.
    save_dir : where to write artifact_gallery.pdf
    """
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print("=" * 70)
    print(f"  PhysicsCorruptionOperator — Test Suite  (Eq. 4, Section 4.1)")
    print(f"  Device: {dev}")
    print("=" * 70)

    # ── Synthetic phantom [1, 4, 128, 128, 128] ──────────────────────────
    torch.manual_seed(42)
    H = W = D = 128
    B, C = 1, 4

    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, D), indexing="ij")
    brain = ((xx / 0.6)**2 + (yy / 0.5)**2 + (zz / 0.7)**2 < 1.0).float()
    tumour = ((xx - 0.1)**2 + (yy + 0.15)**2 + (zz - 0.05)**2 < 0.05).float()
    phantom = brain * 0.8 + tumour * 0.2
    phantom = phantom.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1, -1).clone()
    for ci in range(C):
        phantom[0, ci] *= (0.8 + 0.1 * ci)
    phantom = phantom.to(dev)

    print(f"\n  Phantom: {phantom.shape}  range=[{phantom.min():.3f}, {phantom.max():.3f}]")

    # ── Individual artifact tests ─────────────────────────────────────────
    operator = PhysicsCorruptionOperator(device=device)

    configs = {
        "motion":        {"artifacts": ["motion"],
                          "motion": {"pct": 0.15, "alpha": 0.2}},
        "ghosting":      {"artifacts": ["ghosting"],
                          "ghosting": {"amplitude": 1.1, "phase": 0.08}},
        "undersampling": {"artifacts": ["undersampling"],
                          "undersampling": {"accel_factor": 4.0, "acs_fraction": 0.08}},
        "noise":         {"artifacts": ["noise"],
                          "noise": {"sigma_frac": 0.03}},
    }

    print("\n--- Individual artifact tests ---")
    corrupted_vols: Dict[str, torch.Tensor] = {}
    results: Dict[str, float] = {}

    for name, xi in configs.items():
        out = operator.forward_with_params(phantom, xi)
        assert out.shape == phantom.shape, f"Shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"NaN in {name}"
        assert not torch.isinf(out).any(), f"Inf in {name}"
        p = _psnr(phantom, out)
        results[name] = p
        corrupted_vols[name] = out.cpu()
        print(f"  {name:20s}  PSNR={p:.2f} dB  "
              f"range=[{out.min():.3f}, {out.max():.3f}]  ✓")

    # ── Combined ──────────────────────────────────────────────────────────
    print("\n--- Combined artifacts ---")
    xi_combo = {
        "artifacts": ["motion", "ghosting", "noise"],
        "motion": {"pct": 0.10, "alpha": 0.15},
        "ghosting": {"amplitude": 1.05, "phase": 0.05},
        "noise": {"sigma_frac": 0.02},
    }
    out_combo = operator.forward_with_params(phantom, xi_combo)
    print(f"  motion+ghost+noise   PSNR={_psnr(phantom, out_combo):.2f} dB  ✓")

    # ── Random draws ──────────────────────────────────────────────────────
    print("\n--- Random sampling (10 draws) ---")
    for i in range(10):
        xi_r = operator.sample_xi()
        out_r = operator.forward_with_params(phantom, xi_r)
        arts = ", ".join(xi_r["artifacts"]) or "(none)"
        print(f"    draw {i}: [{arts:40s}]  PSNR={_psnr(phantom, out_r):.1f} dB")

    # ── Dual independent corruption (CRITICAL training pattern) ───────────
    print("\n--- Dual independent corruption ---")
    xi_obs = operator.sample_xi()
    xi_anc = operator.sample_xi()
    y      = operator.forward_with_params(phantom, xi_obs)
    x_phys = operator.forward_with_params(phantom, xi_anc)
    diff = (y - x_phys).abs().mean().item()
    print(f"  ξ_obs    : {xi_obs['artifacts']}")
    print(f"  ξ_anchor : {xi_anc['artifacts']}")
    print(f"  Mean |y - x_phys| = {diff:.6f}  "
          f"{'✓ different' if diff > 0 else '(identical ξ — rare but valid)'}")

    # ── Identity (no artifacts) ───────────────────────────────────────────
    print("\n--- Identity (empty ξ) ---")
    xi_empty = {"artifacts": []}
    out_id = operator.forward_with_params(phantom, xi_empty)
    err = (phantom - out_id).abs().max().item()
    print(f"  Max |x - P(x; ∅)| = {err:.2e}  {'✓' if err < 1e-4 else '✗'}")

    # ── Per-contrast independence ─────────────────────────────────────────
    print("\n--- Per-contrast independence ---")
    p_single = torch.zeros_like(phantom)
    p_single[0, 0] = phantom[0, 0].clone()
    xi_pc = {"artifacts": ["motion"], "motion": {"pct": 0.1, "alpha": 0.2}}
    out_pc = operator.forward_with_params(p_single, xi_pc)
    c0_ok = (out_pc[0, 0] - p_single[0, 0]).abs().sum().item() > 0
    c1_ok = out_pc[0, 1].abs().sum().item() < 1e-6
    c2_ok = out_pc[0, 2].abs().sum().item() < 1e-6
    c3_ok = out_pc[0, 3].abs().sum().item() < 1e-6
    print(f"  C0 corrupted:   {c0_ok}  {'✓' if c0_ok else '✗'}")
    print(f"  C1 stays zero:  {c1_ok}  {'✓' if c1_ok else '✗'}")
    print(f"  C2 stays zero:  {c2_ok}  {'✓' if c2_ok else '✗'}")
    print(f"  C3 stays zero:  {c3_ok}  {'✓' if c3_ok else '✗'}")

    # ── PDF artifact gallery ──────────────────────────────────────────────
    print("\n--- Generating PDF figure ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 3, figsize=(12, 18))
        mid = H // 2
        clean_np = phantom[0, 0].cpu().numpy()
        vmax = clean_np.max()

        rows = [("Clean", phantom[0, 0].cpu().numpy())]
        for name in configs:
            rows.append((name, corrupted_vols[name][0, 0].numpy()))

        for row, (label, vol) in enumerate(rows):
            slices = [vol[mid, :, :], vol[:, mid, :], vol[:, :, mid]]
            titles = ["Axial (z=64)", "Coronal (y=64)", "Sagittal (x=64)"]
            for col, (sl, ttl) in enumerate(zip(slices, titles)):
                ax = axes[row, col]
                ax.imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
                if row == 0:
                    ax.set_title(ttl, fontsize=11)
                if col == 0:
                    extra = f"  PSNR={results[label]:.1f}dB" if label in results else ""
                    ax.set_ylabel(f"{label}{extra}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(
            "PhysicsCorruptionOperator — Artifact Gallery (Eq. 4, Section 4.1)",
            fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, "artifact_gallery.pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved: {pdf_path}")
    except ImportError:
        print("  matplotlib not available — skipping PDF")
    except Exception as e:
        print(f"  PDF error: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ALL TESTS PASSED ✓")
    print(f"{'=' * 70}")


# ======================================================================= #
#  __main__  (Jupyter-safe — calls _run_tests directly, NO argparse)       #
# ======================================================================= #

if __name__ == "__main__":
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _save = os.path.join(
        "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/"
        "NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/figures"
    ) if os.path.isdir("/media/image522") else "."
    _run_tests(device=_device, save_dir=_save)

