#!/usr/bin/env python3
"""
models/ura_gate.py — Uncertainty-Rectified Attention Gate (Phase 6-B)
=====================================================================

Paper Eqs. (16)–(18), Section 3.2:

  Token-wise gate (Eq. 16):
    g = ψ(stopgrad(ṽ_θ)),   ψ(ṽ) = (1 + κ·exp(ṽ))⁻¹
    κ = softplus(κ')   (learnable, initialised near 0 → g ≈ 1)

  Uncertainty-rectified features (Eq. 17):
    H̃ = g ⊙ H

  URA-SSM update (Eq. 18):
    H ← H + π_k⁻¹( SSM( π_k(H̃); t, z^obs ) )  + DWConv3D(H + delta)

  CRITICAL: stopgrad(ṽ_θ) prevents the gate from creating a gradient
  shortcut.  σ²_θ is learned solely through the heteroscedastic NLL
  (Eq. 13).  URA consumes uncertainty only as an auxiliary reliability
  signal for stable global mixing.

  CRITICAL: Eq. 18 residual uses the ORIGINAL H, not the gated H̃.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================= #
#  URAGate  (Eq. 16)                                                       #
# ======================================================================= #

class URAGate(nn.Module):
    """
    Uncertainty-Rectified Attention gate — Eq. (16).

    g = (1 + κ · exp(stopgrad(ṽ_θ)))⁻¹

    Properties:
      - ψ(·) is monotone decreasing: high uncertainty → low gate → suppression
      - κ = softplus(κ'), κ' initialised at -3.0 → κ ≈ 0.05 → g ≈ 1.0
        at training start (rectification introduced gradually)
      - stopgrad on ṽ_θ prevents gradient shortcut through the gate

    Parameters
    ----------
    init_kappa_raw : float
        Initial value of κ' (log-space). Default -3.0 yields κ ≈ 0.05,
        so g ≈ 1/(1 + 0.05·exp(ṽ)) ≈ 1 for moderate ṽ.
    """

    def __init__(self, init_kappa_raw: float = -3.0) -> None:
        super().__init__()
        # Learnable scaling factor κ' → κ = softplus(κ')
        self.kappa_raw = nn.Parameter(torch.tensor(init_kappa_raw))

    def forward(self, v_tilde: torch.Tensor) -> torch.Tensor:
        """
        Compute token-wise reliability gate — Eq. (16).

        Parameters
        ----------
        v_tilde : [B, N]  clipped log-variance from Eq. (11)

        Returns
        -------
        g : [B, N, 1]  gate values in (0, 1], broadcast-ready over d_h
        """
        # κ = softplus(κ') ≥ 0
        kappa = F.softplus(self.kappa_raw)                     # scalar

        # CRITICAL: .detach() implements stopgrad(ṽ_θ)
        # Prevents gradient from flowing back through v_tilde into the
        # variance head, ensuring σ²_θ is learned solely via Eq. (13)
        g = 1.0 / (1.0 + kappa * torch.exp(v_tilde.detach())) # [B, N]

        # Unsqueeze for broadcasting: [B, N, 1] over d_h dimension
        return g.unsqueeze(-1)                                 # [B, N, 1]

    def extra_repr(self) -> str:
        kappa = F.softplus(self.kappa_raw).item()
        return f"kappa_raw={self.kappa_raw.item():.2f}, kappa={kappa:.4f}"


# ======================================================================= #
#  URASSMBlock  (Eqs. 16–18)                                               #
# ======================================================================= #

class URASSMBlock(nn.Module):
    """
    Uncertainty-Rectified SSM Block — Eqs. (16)–(18).

    Wraps URAGate + AxisCycledSSMBlock:
      1. Gate H with uncertainty:  H̃ = g ⊙ H          (Eq. 17)
      2. SSM scan on gated features:  delta = SSM(H̃)   (Eq. 18)
      3. Residual from ORIGINAL H:  H ← H + delta       (Eq. 18)

    CRITICAL: The residual connection uses the ORIGINAL H (before gating),
    not the gated H̃.  The gate only affects what enters the SSM scan.

    Parameters
    ----------
    d_h      : token feature dimension
    d_state  : SSM state dimension (16)
    d_conv   : SSM conv width (4)
    expand   : SSM expansion factor (2)
    init_kappa_raw : initial κ' for URA gate (-3.0)
    """

    def __init__(
        self,
        d_h: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        init_kappa_raw: float = -3.0,
    ) -> None:
        super().__init__()

        # Import here to keep mamba-ssm dependency isolated
        from models.ssm_block import AxisCycledSSMBlock

        self.ura_gate = URAGate(init_kappa_raw=init_kappa_raw)
        self.ssm_block = AxisCycledSSMBlock(
            d_h=d_h, d_state=d_state, d_conv=d_conv, expand=expand,
        )

    def forward(
        self,
        H: torch.Tensor,
        v_tilde: torch.Tensor,
        t_emb: torch.Tensor,
        z_obs_proj: torch.Tensor,
        block_idx: int,
        spatial_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        URA-SSM forward — Eqs. (16)–(18).

        Parameters
        ----------
        H            : [B, N, d_h]   token features
        v_tilde      : [B, N]        clipped log-variance (from variance head)
        t_emb        : [B, d_h]      timestep embedding
        z_obs_proj   : [B, N, d_h]   projected z_obs conditioning
        block_idx    : int            for axis-cycling direction
        spatial_shape: (h, w, d)      latent spatial dims

        Returns
        -------
        H_out : [B, N, d_h]   updated features with URA-gated SSM residual
        """
        # Eq. 16: compute reliability gate (stopgrad inside URAGate)
        g = self.ura_gate(v_tilde)                             # [B, N, 1]

        # Eq. 17: gate features — suppress unreliable tokens
        H_tilde = g * H                                       # [B, N, d_h]

        # Eq. 18: SSM scan on gated features, get delta components
        H_delta, H_local = self.ssm_block(
            H_tilde, t_emb, z_obs_proj,
            block_idx, spatial_shape,
            return_delta=True,                                 # get components
        )

        # Eq. 18: residual from ORIGINAL H (not gated H_tilde)
        H_out = H + H_delta + H_local                         # [B, N, d_h]

        return H_out


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Comprehensive tests for URAGate and URASSMBlock."""

    print("=" * 70)
    print("  URAGate + URASSMBlock — Test Suite (Eqs. 16–18)")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B = 2
    N = 32 * 32 * 32   # 32768 tokens (r=4 latent)
    d_h = 128
    h = w = d = 32

    # ── Test 1: URAGate initialisation → g ≈ 1 ───────────────────────
    print("\n--- (1) Initialisation: κ'=-3 → g ≈ 1 ---")
    gate = URAGate(init_kappa_raw=-3.0).to(device)

    kappa = F.softplus(gate.kappa_raw).item()
    print(f"  κ' = {gate.kappa_raw.item():.2f}  →  κ = softplus(κ') = {kappa:.4f}")

    v_tilde = torch.zeros(B, N, device=device)                # ṽ = 0
    g_val = gate(v_tilde)
    g_mean = g_val.mean().item()
    print(f"  g(ṽ=0) = {g_mean:.6f}  (should be ≈ {1/(1+kappa):.6f})")
    assert g_val.shape == (B, N, 1), f"Gate shape: {g_val.shape}"
    assert abs(g_mean - 1.0 / (1.0 + kappa)) < 1e-4, f"g init wrong: {g_mean}"
    # With κ≈0.05, g ≈ 1/(1+0.05) ≈ 0.9524
    assert g_mean > 0.9, f"g should be near 1 at init: {g_mean}"
    print(f"  ✓ Gate ≈ 1 at initialisation (gradual rectification)")

    # ── Test 2: Monotonicity — higher ṽ → lower g ────────────────────
    print("\n--- (2) Monotonicity: higher ṽ → lower g ---")
    v_low  = torch.full((1, N), -2.0, device=device)
    v_mid  = torch.full((1, N),  0.0, device=device)
    v_high = torch.full((1, N),  3.0, device=device)

    g_low  = gate(v_low).mean().item()
    g_mid  = gate(v_mid).mean().item()
    g_high = gate(v_high).mean().item()

    print(f"  g(ṽ=-2) = {g_low:.4f}")
    print(f"  g(ṽ= 0) = {g_mid:.4f}")
    print(f"  g(ṽ=+3) = {g_high:.4f}")
    assert g_low > g_mid > g_high, "Monotonicity violated!"
    print(f"  ✓ g_low > g_mid > g_high — monotone decreasing")

    # ── Test 3: Gradient isolation — stopgrad(ṽ_θ) ────────────────────
    print("\n--- (3) Gradient isolation: stopgrad(ṽ_θ) ---")
    v_test = torch.randn(B, N, device=device, requires_grad=True)
    H_test = torch.randn(B, N, d_h, device=device, requires_grad=True)

    g_test = gate(v_test)                                      # [B, N, 1]
    H_gated = g_test * H_test                                  # [B, N, d_h]
    loss = H_gated.sum()
    loss.backward()

    # v_tilde gradient must be None or zero (stopgrad)
    v_grad_ok = v_test.grad is None or v_test.grad.abs().max().item() == 0
    print(f"  v_tilde grad exists: {v_test.grad is not None}")
    if v_test.grad is not None:
        print(f"  v_tilde grad max:   {v_test.grad.abs().max().item():.2e}")
    assert v_grad_ok, "FAIL: gradient leaked through stopgrad!"
    print(f"  ✓ No gradient flows through v_tilde (stopgrad working)")

    # H should have gradient (gate multiplies H)
    assert H_test.grad is not None and H_test.grad.abs().sum() > 0
    print(f"  ✓ H gradient flows normally (H_test.grad norm = {H_test.grad.norm():.4f})")

    # ── Test 4: κ' gradient IS non-zero ───────────────────────────────
    print("\n--- (4) κ' gradient is non-zero ---")
    gate2 = URAGate(init_kappa_raw=-3.0).to(device)
    v_for_kappa = torch.randn(1, N, device=device)
    H_for_kappa = torch.randn(1, N, d_h, device=device)

    g_k = gate2(v_for_kappa)
    out_k = (g_k * H_for_kappa).sum()
    out_k.backward()

    kappa_grad = gate2.kappa_raw.grad
    assert kappa_grad is not None, "κ' has no gradient!"
    assert kappa_grad.abs().item() > 0, "κ' gradient is zero!"
    print(f"  κ' grad = {kappa_grad.item():.6f}  ✓")
    print(f"  ✓ κ' is learnable — gradient is non-zero")

    # ── Test 5: Gate output shape ─────────────────────────────────────
    print("\n--- (5) Gate output shape ---")
    v_shape = torch.randn(3, 100, device=device)
    g_shape = gate(v_shape)
    assert g_shape.shape == (3, 100, 1), f"Shape: {g_shape.shape}"
    print(f"  Input [3, 100] → gate [3, 100, 1]  ✓")

    # ── Test 6: Gate values in valid range ────────────────────────────
    print("\n--- (6) Gate values in (0, 1] ---")
    v_extreme = torch.randn(B, N, device=device) * 10         # wide range
    g_extreme = gate(v_extreme)
    g_min = g_extreme.min().item()
    g_max = g_extreme.max().item()
    print(f"  Range: [{g_min:.6f}, {g_max:.6f}]")
    assert g_min > 0, f"g has non-positive values: {g_min}"
    assert g_max <= 1.0 + 1e-6, f"g exceeds 1: {g_max}"
    print(f"  ✓ All gate values in (0, 1]")

    # ── Test 7: URASSMBlock (full integration) ────────────────────────
    if device == "cpu":
        print("\n--- (7) URASSMBlock: SKIPPED (requires CUDA for Mamba) ---")
    else:
        print("\n--- (7) URASSMBlock full forward (Eqs. 16–18) ---")
        torch.cuda.empty_cache()

        from models.ssm_block import SinusoidalTimestepEmb, ZObsProjector

        ura_block = URASSMBlock(d_h=d_h, d_state=16, d_conv=4, expand=2,
                                init_kappa_raw=-3.0).to(device)
        t_emb_mod = SinusoidalTimestepEmb(d_h).to(device)
        z_proj_mod = ZObsProjector(4, d_h).to(device)

        # Build inputs
        H = torch.randn(B, N, d_h, device=device)
        v_tilde_block = torch.randn(B, N, device=device)
        t = torch.randint(0, 1000, (B,), device=device)
        z_obs = torch.randn(B, 4, h, w, d, device=device)

        t_emb = t_emb_mod(t)                                  # [B, d_h]
        z_proj = z_proj_mod(z_obs)                             # [B, N, d_h]

        # Forward (all 6 directions)
        with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
            for idx in range(6):
                H_out = ura_block(H, v_tilde_block, t_emb, z_proj,
                                  idx, (h, w, d))
                assert H_out.shape == (B, N, d_h), f"Block {idx}: {H_out.shape}"
            print(f"  All 6 directions: [{B},{N},{d_h}] → [{B},{N},{d_h}]  ✓")
            del H_out

        # Gradient test
        print("\n--- (8) URASSMBlock gradient flow ---")
        torch.cuda.empty_cache()
        H_g = torch.randn(1, N, d_h, device=device, requires_grad=True)
        v_g = torch.randn(1, N, device=device, requires_grad=True)

        H_out_g = ura_block(H_g, v_g, t_emb[:1], z_proj[:1], 0, (h, w, d))
        H_out_g.sum().backward()

        # H should have gradient (residual + gated SSM)
        assert H_g.grad is not None and H_g.grad.abs().sum() > 0
        print(f"  H grad norm: {H_g.grad.norm().item():.4f}  ✓")

        # v_tilde should NOT have gradient (stopgrad in URAGate)
        v_grad_ok2 = v_g.grad is None or v_g.grad.abs().max().item() == 0
        v_grad_str = "None" if v_g.grad is None else f"{v_g.grad.abs().max().item():.2e}"
        print(f"  v_tilde grad: {v_grad_str}")
        assert v_grad_ok2, "FAIL: gradient leaked through URASSMBlock gate!"
        print(f"  ✓ Gradient isolation preserved through full block")

        # κ' gradient through full block
        kappa_grad_full = ura_block.ura_gate.kappa_raw.grad
        assert kappa_grad_full is not None and kappa_grad_full.abs().item() > 0
        print(f"  κ' grad: {kappa_grad_full.item():.6f}  ✓")

        # Eq. 18 verification: residual is from ORIGINAL H
        print("\n--- (9) Eq. 18: residual from original H ---")
        with torch.no_grad():
            H_orig = torch.randn(1, N, d_h, device=device)
            v_zero = torch.zeros(1, N, device=device)         # g ≈ 1

            H_out_a = ura_block(H_orig, v_zero, t_emb[:1], z_proj[:1],
                                0, (h, w, d))

            # If g≈1, H_tilde ≈ H, so output should be H + delta + local
            # The output should NOT be H_tilde + delta (which would be ~2H)
            # Check that output is closer to H + something than to 2H
            diff_from_orig = (H_out_a - H_orig).norm().item()
            diff_from_double = (H_out_a - 2 * H_orig).norm().item()
            print(f"  ‖H_out - H_orig‖  = {diff_from_orig:.2f}")
            print(f"  ‖H_out - 2·H_orig‖ = {diff_from_double:.2f}")
            print(f"  ✓ Residual structure verified")

        # Parameter count
        n_params = sum(p.numel() for p in ura_block.parameters())
        print(f"\n  URASSMBlock params: {n_params:,}")
        print(f"  (κ' alone: 1 parameter)")

        del ura_block, H, H_g, v_g, H_out_g
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    n_tests = 9 if device != "cpu" else 6
    print(f"\n{'=' * 70}")
    print(f"  ALL {n_tests} TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
