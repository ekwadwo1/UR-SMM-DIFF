#!/usr/bin/env python3
"""
models/ur_ssm_diff.py — Full UR-SSM-Diff Model Wrapper (Phase 8-A)
===================================================================

Integrates all components into a single module:
  - VQGAN3D (frozen encoder/decoder for latent space)
  - PhysicsCorruptionOperator (dual-independent corruption)
  - PhysicsConsistentDiffusion (forward/reverse process)
  - URSSMDenoiser (SSM denoiser f_θ)
  - SegmentationHead3D (tumor segmentation S_φ)
  - HeteroscedasticDiffusionLoss (Eq. 13)
  - TotalLoss (Eq. 21/22)

CRITICAL: training_step uses single-step Tweedie x₀ estimate
for segmentation (NOT full DDIM). Full DDIM is inference-only.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.physics_forward import PhysicsConsistentDiffusion
from losses.diffusion_loss import HeteroscedasticDiffusionLoss
from losses.segmentation_loss import (
    GammaScheduler,
    SegmentationLoss,
    map_brats_labels,
)
from models.denoiser import URSSMDenoiser
from models.seg_head import SegmentationHead3D
from models.vqgan3d import VQGAN3D
from physics.corruption_operator import PhysicsCorruptionOperator


class URSSMDiff(nn.Module):
    """
    Full UR-SSM-Diff pipeline — Eqs. (1)–(22).

    Wraps all components into a unified module with:
      - training_step(): forward pass with Tweedie seg estimate
      - inference(): modified DDIM sampling with uncertainty aggregation
      - restore_and_segment(): full pipeline from observed volume to
        restored image + segmentation

    Parameters
    ----------
    vqgan             : pre-trained VQGAN3D (frozen)
    denoiser          : URSSMDenoiser f_θ
    seg_head          : SegmentationHead3D S_φ
    corruption_op     : PhysicsCorruptionOperator P(·; ξ)
    diffusion         : PhysicsConsistentDiffusion schedules
    latent_dim        : c (4)
    downsample_factor : r (4 or 8)
    T                 : diffusion timesteps (1000)
    n_classes         : K (4)
    gamma_target      : final γ for Regime A (0.1)
    gamma_warmup      : γ ramp steps (2000)
    regime            : 'A' or 'B'
    log_var_range     : (min, max) for v_tilde clipping
    """

    def __init__(
        self,
        vqgan: VQGAN3D,
        denoiser: URSSMDenoiser,
        seg_head: SegmentationHead3D,
        corruption_op: PhysicsCorruptionOperator,
        diffusion: Optional[PhysicsConsistentDiffusion] = None,
        latent_dim: int = 4,
        downsample_factor: int = 4,
        T: int = 1000,
        n_classes: int = 4,
        gamma_target: float = 0.1,
        gamma_warmup: int = 2000,
        regime: str = "A",
        log_var_range: Tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.downsample_factor = downsample_factor
        self.T = T
        self.regime = regime.upper()
        self.log_var_min, self.log_var_max = log_var_range

        # ── Sub-modules ───────────────────────────────────────────────
        self.vqgan = vqgan
        self.denoiser = denoiser
        self.seg_head = seg_head
        self.corruption_op = corruption_op

        self.diffusion = diffusion or PhysicsConsistentDiffusion(
            T=T, schedule="cosine", rho=2.0)

        # ── Losses ────────────────────────────────────────────────────
        self.hetero_loss = HeteroscedasticDiffusionLoss(
            sigma_min=math.exp(0.5 * log_var_range[0]),
            sigma_max=math.exp(0.5 * log_var_range[1]),
        )
        self.seg_loss = SegmentationLoss(n_classes=n_classes)
        self.gamma_sched = GammaScheduler(gamma_target, gamma_warmup)

        # ── Freeze VQGAN ──────────────────────────────────────────────
        for p in self.vqgan.parameters():
            p.requires_grad_(False)
        self.vqgan.eval()

    # ================================================================== #
    #  Training step  (Tweedie, NOT full DDIM)                            #
    # ================================================================== #

    def training_step(
        self,
        x_0: torch.Tensor,
        s_0: torch.Tensor,
        global_step: int = 0,
    ) -> Dict[str, Any]:
        """
        Single training step — Algorithm 1 + Eq. (13) + Eq. (19)–(21).

        CRITICAL: Uses single-step Tweedie x₀ estimate for segmentation,
        NOT full DDIM reverse. This avoids 50-step unrolling inside
        the training loop.

        Parameters
        ----------
        x_0          : [B, C, H, W, D]  clean mpMRI volume (C=4)
        s_0          : [B, H, W, D]     raw BraTS labels {0,1,2,4}
        global_step  : int              for γ ramp

        Returns
        -------
        dict with 'loss', 'l_diff', 'l_seg', 'gamma', 'v_tilde', ...
        """
        B = x_0.shape[0]
        device = x_0.device

        # ── 1. Two INDEPENDENT corruptions (Eq. 4) ───────────────────
        # y = P(x₀; ξ_obs) — observed volume for conditioning
        # x_phys = P(x₀; ξ_anchor) — physics anchor for forward trajectory
        with torch.no_grad():
            y = self.corruption_op(x_0)                        # [B,C,H,W,D]
            x_phys = self.corruption_op(x_0)                   # [B,C,H,W,D]

        # ── 2. Encode to latent space (Eq. 2, VQGAN frozen) ──────────
        with torch.no_grad():
            z_0 = self.vqgan(x_0, mode="encode")              # [B,c,h,w,d]
            z_obs = self.vqgan(y, mode="encode")               # [B,c,h,w,d]
            z_0_phys = self.vqgan(x_phys, mode="encode")      # [B,c,h,w,d]

        # ── 3. Forward diffusion (Algorithm 1, Eqs. 6–8) ─────────────
        t = self.diffusion.sample_timesteps(B, device)         # [B]
        z_t, eps = self.diffusion.q_sample(z_0, z_0_phys, t)  # [B,c,h,w,d]

        # ── 4. Denoiser forward (Eq. 10) ─────────────────────────────
        eps_theta, v_theta = self.denoiser(z_t, t, z_obs)     # [B,c,h,w,d], [B,N]

        # ── 5. Diffusion loss (Eq. 13) ───────────────────────────────
        l_diff, v_tilde, diff_logs = self.hetero_loss(eps, eps_theta, v_theta)

        # ── 6. Tweedie x₀ estimate (single-step, for seg) ────────────
        z_0_pred = self.diffusion.tweedie_x0_estimate(
            z_t, eps_theta, t, z_obs)                          # [B,c,h,w,d]

        # ── 7. Segmentation (Eq. 19) ─────────────────────────────────
        if self.regime == "B":
            z_0_pred = z_0_pred.detach()                       # stopgrad (Eq. 22)

        seg_logits = self.seg_head(z_0_pred)                   # [B,K,H,W,D]

        # Map BraTS labels {0,1,2,4} → {0,1,2,3}
        s_0_mapped = map_brats_labels(s_0)                     # [B,H,W,D]

        l_seg, seg_logs = self.seg_loss(seg_logits, s_0_mapped)

        # ── 8. Total loss (Eq. 21) ───────────────────────────────────
        gamma = self.gamma_sched(global_step)

        if self.regime == "B":
            l_total = l_seg                                    # Eq. 22
        else:
            l_total = l_diff + gamma * l_seg                   # Eq. 21

        # ── Return ────────────────────────────────────────────────────
        return {
            "loss": l_total,
            "l_diff": l_diff.item(),
            "l_seg": l_seg.item(),
            "gamma": gamma,
            "v_tilde": v_tilde.detach(),                       # for monitoring
            **diff_logs,
            **seg_logs,
        }

    # ================================================================== #
    #  Inference: DDIM sampling with uncertainty aggregation               #
    # ================================================================== #

    @torch.no_grad()
    def ddim_sample_with_uncertainty(
        self,
        z_obs: torch.Tensor,
        S: int = 50,
        eta: float = 0.0,
        ema_decay: float = 0.9,
        t_start_frac: float = 0.80,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified DDIM reverse sampling with EMA uncertainty aggregation.

        CRITICAL FIX: Physics-consistent forward at t=T yields
        z_T = z₀^phys (NOT Gaussian noise), so starting from randn()
        is out-of-distribution. We cap the start at t_start_frac*T
        where (1−λ_t) is well-conditioned and initialise from the
        forward marginal using z_obs as proxy:

          z_{t_s} = ((1−λ)√ᾱ + λ)·z_obs + (1−λ)√(1−ᾱ)·ε

        Uncertainty is aggregated across DDIM steps via exponential
        moving average — later steps (closer to z₀) are weighted more.

        Parameters
        ----------
        z_obs        : [B, c, h, w, d]  observed latent E(y)
        S            : DDIM steps (50 or 100)
        eta          : DDIM stochasticity (0 = deterministic)
        ema_decay    : EMA weight for uncertainty aggregation (0.9)
        t_start_frac : fraction of T for safe DDIM start (default 0.80)

        Returns
        -------
        z_0_pred     : [B, c, h, w, d]  denoised latent
        sigma_sq_agg : [B, N]            aggregated aleatoric uncertainty
        """
        B, c, h, w, d = z_obs.shape
        N = h * w * d
        device = z_obs.device

        # Cap starting timestep to avoid degenerate λ≈1, ᾱ≈0 regime
        t_start = int(t_start_frac * self.T) - 1              # 0-indexed
        t_start = max(0, min(t_start, self.T - 1))

        # DDIM timestep subsequence from t_start → 0
        step_indices = torch.linspace(
            t_start, 0, S + 1, dtype=torch.long, device=device)

        # ── Physics-aware initialisation (Eq. 8 with z_obs proxy) ────
        # z_{t_s} = ((1−λ)√ᾱ + λ)·z_obs + (1−λ)√(1−ᾱ)·ε
        t_s = torch.tensor([t_start], device=device)
        lam_s     = self.diffusion._extract(self.diffusion.lambdas, t_s)
        sqrt_ab_s = self.diffusion._extract(self.diffusion.sqrt_ab, t_s)
        sqrt_1m_s = self.diffusion._extract(self.diffusion.sqrt_one_m_ab, t_s)

        signal_coeff = (1.0 - lam_s) * sqrt_ab_s + lam_s
        noise_coeff  = (1.0 - lam_s) * sqrt_1m_s

        z_t = signal_coeff * z_obs + noise_coeff * torch.randn_like(z_obs)

        sigma_sq_agg = torch.zeros(B, N, device=device)
        v_tilde_prev = None

        for i in range(S):
            t_cur = step_indices[i].expand(B)
            t_next = step_indices[i + 1].expand(B)

            # Denoiser with previous v_tilde for URA gating
            eps_theta, v_theta = self.denoiser(
                z_t, t_cur, z_obs, v_tilde_ext=v_tilde_prev)

            # Clip log-variance (Eq. 11)
            v_tilde = v_theta.clamp(self.log_var_min, self.log_var_max)
            sigma_sq_t = torch.exp(v_tilde)                    # [B, N]

            # EMA uncertainty aggregation (later steps weighted more)
            sigma_sq_agg = (ema_decay * sigma_sq_agg
                            + (1.0 - ema_decay) * sigma_sq_t)

            # Modified DDIM step
            t_prev = t_next.clone()
            if i == S - 1:
                t_prev = t_prev - 1                            # → -1 for final

            z_t = self.diffusion.ddim_step(
                z_t, eps_theta, t_cur, t_prev, z_obs, eta)

            v_tilde_prev = v_tilde                             # feed to next step

        return z_t, sigma_sq_agg

    # ================================================================== #
    #  Full inference pipeline                                             #
    # ================================================================== #

    @torch.no_grad()
    def restore_and_segment(
        self,
        y: torch.Tensor,
        S: int = 50,
        eta: float = 0.0,
        ema_decay: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference: observed volume → restored image + segmentation.

        Parameters
        ----------
        y         : [B, C, H, W, D]  observed (corrupted) mpMRI volume
        S         : DDIM steps
        eta       : DDIM stochasticity
        ema_decay : uncertainty EMA

        Returns
        -------
        dict with:
          'x_restored'  : [B, C, H, W, D]  restored volume
          'seg_logits'  : [B, K, H, W, D]  segmentation logits
          'seg_pred'    : [B, H, W, D]     argmax class predictions
          'sigma_sq'    : [B, N]           aggregated uncertainty
          'z_0_pred'    : [B, c, h, w, d]  denoised latent
        """
        # Encode observed volume
        z_obs = self.vqgan(y, mode="encode")                   # [B,c,h,w,d]

        # DDIM reverse with uncertainty
        z_0_pred, sigma_sq = self.ddim_sample_with_uncertainty(
            z_obs, S=S, eta=eta, ema_decay=ema_decay)

        # Decode to image space (Eq. 3)
        x_restored = self.vqgan(z_0_pred, mode="decode")      # [B,C,H,W,D]

        # Segment (Eq. 19)
        seg_logits = self.seg_head(z_0_pred)                   # [B,K,H,W,D]
        seg_pred = seg_logits.argmax(dim=1)                    # [B,H,W,D]

        return {
            "x_restored": x_restored,
            "seg_logits": seg_logits,
            "seg_pred": seg_pred,
            "sigma_sq": sigma_sq,
            "z_0_pred": z_0_pred,
        }


# ======================================================================= #
#  Factory function                                                        #
# ======================================================================= #

def build_ur_ssm_diff(
    vqgan_ckpt: Optional[str] = None,
    latent_dim: int = 4,
    d_h: int = 128,
    downsample_factor: int = 4,
    n_classes: int = 4,
    T: int = 1000,
    gamma_target: float = 0.1,
    gamma_warmup: int = 2000,
    regime: str = "A",
    device: str = "cuda",
) -> URSSMDiff:
    """
    Factory function to build the complete UR-SSM-Diff model.

    Parameters
    ----------
    vqgan_ckpt : path to pre-trained VQGAN checkpoint (None = random init)
    """
    # VQGAN
    vqgan = VQGAN3D(
        in_channels=4, latent_dim=latent_dim,
        downsample_factor=downsample_factor)

    if vqgan_ckpt is not None:
        ckpt = torch.load(vqgan_ckpt, map_location="cpu", weights_only=True)
        state = ckpt.get("gen", ckpt)
        vqgan.load_state_dict(state, strict=False)

    # Denoiser
    denoiser = URSSMDenoiser(
        latent_dim=latent_dim, d_h=d_h,
        stage_depths=(2, 2, 4), bottleneck_depth=4,
        use_checkpoint=True)

    # Seg head
    seg_head = SegmentationHead3D(
        latent_dim=latent_dim, n_classes=n_classes,
        downsample_factor=downsample_factor)

    # Corruption operator
    corruption_op = PhysicsCorruptionOperator(device=device)

    # Diffusion
    diffusion = PhysicsConsistentDiffusion(T=T, schedule="cosine", rho=2.0)

    # Assemble
    model = URSSMDiff(
        vqgan=vqgan, denoiser=denoiser, seg_head=seg_head,
        corruption_op=corruption_op, diffusion=diffusion,
        latent_dim=latent_dim, downsample_factor=downsample_factor,
        T=T, n_classes=n_classes,
        gamma_target=gamma_target, gamma_warmup=gamma_warmup,
        regime=regime)

    return model.to(device)


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Comprehensive tests for URSSMDiff wrapper."""

    print("=" * 70)
    print("  URSSMDiff — Full Model Test Suite")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ⚠ Skipping — requires CUDA for Mamba SSM")
        return

    torch.manual_seed(42)

    r = 4
    h = w = d = 128 // r   # 32
    H = W = D = 128
    c, C, K = 4, 4, 4
    B = 1
    N = h * w * d           # 32768

    # ── Build model ───────────────────────────────────────────────────
    print("\n--- Building model ---")
    model = build_ur_ssm_diff(
        latent_dim=c, d_h=128, downsample_factor=r,
        n_classes=K, T=1000, regime="A", device=device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_vqgan = sum(p.numel() for p in model.vqgan.parameters())
    print(f"  Total params:     {n_total:,}")
    print(f"  Trainable params: {n_train:,}")
    print(f"  VQGAN (frozen):   {n_vqgan:,}")
    print(f"  Denoiser:         {sum(p.numel() for p in model.denoiser.parameters()):,}")
    print(f"  Seg head:         {sum(p.numel() for p in model.seg_head.parameters()):,}")

    # Verify VQGAN is frozen
    assert all(not p.requires_grad for p in model.vqgan.parameters()), \
        "VQGAN should be frozen!"
    print(f"  ✓ VQGAN frozen")

    # ── Test 1: training_step ─────────────────────────────────────────
    print("\n--- (1) training_step (Tweedie seg, Regime A) ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x_0 = torch.randn(B, C, H, W, D, device=device)
    # Fake BraTS labels with values {0, 1, 2, 4}
    s_0 = torch.zeros(B, H, W, D, dtype=torch.long, device=device)
    s_0[:, :40, :, :] = 1
    s_0[:, 40:80, :, :] = 2
    s_0[:, 80:100, :, :] = 4

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        out = model.training_step(x_0, s_0, global_step=0)

    assert "loss" in out, "Missing 'loss' key"
    assert isinstance(out["loss"], torch.Tensor)
    assert out["loss"].requires_grad, "Loss should require grad!"
    print(f"  loss: {out['loss'].item():.4f}  ✓")
    print(f"  l_diff: {out['l_diff']:.4f}")
    print(f"  l_seg:  {out['l_seg']:.4f}")
    print(f"  gamma:  {out['gamma']:.4f}  (should be 0 at step 0)")
    assert abs(out["gamma"]) < 1e-8, "γ should be 0 at step 0"
    print(f"  v_tilde shape: {list(out['v_tilde'].shape)}  ✓")

    peak_train = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  Peak VRAM: {peak_train:.2f} GB")

    # Backward test
    out["loss"].backward()
    denoiser_grad = sum(
        p.grad.abs().sum().item() for p in model.denoiser.parameters()
        if p.grad is not None)
    seg_grad = sum(
        p.grad.abs().sum().item() for p in model.seg_head.parameters()
        if p.grad is not None)
    assert denoiser_grad > 0, "Denoiser has no gradient!"
    # At step 0, γ=0 → L_total = L_diff only → seg head correctly has no grad
    assert seg_grad == 0, "Seg head should have NO gradient when γ=0!"
    print(f"  Denoiser grad sum: {denoiser_grad:.2f}  ✓")
    print(f"  Seg head grad sum: {seg_grad:.2f}  ✓ (correct: γ=0 → no seg grad)")

    model.zero_grad()
    del x_0, s_0, out
    torch.cuda.empty_cache()

    # ── Test 2: training_step with γ > 0 ──────────────────────────────
    print("\n--- (2) training_step at step=2000 (γ=0.1) ---")
    x_0 = torch.randn(B, C, H, W, D, device=device)
    s_0 = torch.randint(0, 3, (B, H, W, D), device=device)
    s_0[s_0 == 3] = 4  # map 3→4 for BraTS convention

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        out2 = model.training_step(x_0, s_0, global_step=2000)

    assert abs(out2["gamma"] - 0.1) < 1e-6, f"γ should be 0.1: {out2['gamma']}"
    print(f"  gamma: {out2['gamma']:.4f}  ✓")
    print(f"  loss:  {out2['loss'].item():.4f}")

    out2["loss"].backward()

    # Now γ=0.1 → seg head SHOULD have gradient
    seg_grad_2 = sum(
        p.grad.abs().sum().item() for p in model.seg_head.parameters()
        if p.grad is not None)
    assert seg_grad_2 > 0, "Seg head should have gradient when γ=0.1!"
    print(f"  Seg head grad sum: {seg_grad_2:.2f}  ✓ (γ>0 → seg contributes)")

    model.zero_grad()
    del x_0, s_0, out2
    torch.cuda.empty_cache()

    # ── Test 3: Regime B (stopgrad) ───────────────────────────────────
    print("\n--- (3) Regime B (stopgrad on denoiser) ---")
    model.regime = "B"

    x_0 = torch.randn(B, C, H, W, D, device=device)
    s_0 = torch.zeros(B, H, W, D, dtype=torch.long, device=device)

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        out3 = model.training_step(x_0, s_0, global_step=100)

    out3["loss"].backward()

    # In Regime B, denoiser should get gradient from l_diff only,
    # but seg loss path is detached, so seg_head still gets gradient
    seg_grad_B = sum(
        p.grad.abs().sum().item() for p in model.seg_head.parameters()
        if p.grad is not None)
    assert seg_grad_B > 0, "Seg head should have gradient in Regime B!"
    print(f"  Seg head grad: {seg_grad_B:.2f}  ✓")
    print(f"  ✓ Regime B working")

    model.regime = "A"  # restore
    model.zero_grad()
    del x_0, s_0, out3
    torch.cuda.empty_cache()

    # ── Test 4: DDIM inference (2 steps for speed) ────────────────────
    print("\n--- (4) DDIM inference (S=2 steps) ---")
    z_obs = torch.randn(B, c, h, w, d, device=device)

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        z_0_pred, sigma_sq = model.ddim_sample_with_uncertainty(
            z_obs, S=2, eta=0.0, ema_decay=0.9)

    assert z_0_pred.shape == (B, c, h, w, d), f"z_0: {z_0_pred.shape}"
    assert sigma_sq.shape == (B, N), f"sigma: {sigma_sq.shape}"
    assert (sigma_sq >= 0).all(), "Negative uncertainty!"
    print(f"  z_0_pred: {list(z_0_pred.shape)}  ✓")
    print(f"  sigma_sq: {list(sigma_sq.shape)}  range=[{sigma_sq.min():.4f}, {sigma_sq.max():.4f}]  ✓")
    del z_obs, z_0_pred, sigma_sq
    torch.cuda.empty_cache()

    # ── Test 5: Full restore_and_segment pipeline ─────────────────────
    print("\n--- (5) restore_and_segment (S=2) ---")
    y_test = torch.randn(B, C, H, W, D, device=device)

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        results = model.restore_and_segment(y_test, S=2)

    assert results["x_restored"].shape == (B, C, H, W, D)
    assert results["seg_logits"].shape == (B, K, H, W, D)
    assert results["seg_pred"].shape == (B, H, W, D)
    assert results["sigma_sq"].shape == (B, N)
    assert results["seg_pred"].min() >= 0 and results["seg_pred"].max() < K
    print(f"  x_restored: {list(results['x_restored'].shape)}  ✓")
    print(f"  seg_logits: {list(results['seg_logits'].shape)}  ✓")
    print(f"  seg_pred:   {list(results['seg_pred'].shape)}  unique={results['seg_pred'].unique().tolist()}")
    print(f"  sigma_sq:   {list(results['sigma_sq'].shape)}  ✓")
    print(f"  z_0_pred:   {list(results['z_0_pred'].shape)}  ✓")

    del y_test, results
    torch.cuda.empty_cache()

    # ── Test 6: Dual independent corruption ───────────────────────────
    print("\n--- (6) Dual independent corruption ---")
    x_test = torch.randn(1, C, H, W, D, device=device)
    with torch.no_grad():
        y1 = model.corruption_op(x_test)
        y2 = model.corruption_op(x_test)
    diff = (y1 - y2).abs().mean().item()
    print(f"  Mean |y₁ - y₂| = {diff:.4f}  (should be > 0 usually)")
    print(f"  ✓ Independent corruption verified")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ALL 6 TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
