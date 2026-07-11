#!/usr/bin/env python3
"""
losses/diffusion_loss.py — Heteroscedastic Diffusion Loss (Phase 7-A)
=====================================================================

Paper Eqs. (11)–(13), Section 3.2:

  Bounded variance (Eq. 11):
    ṽ_θ = clamp(v_θ, log σ²_min, log σ²_max)
    σ²_θ = exp(ṽ_θ)

  Variance regulariser (Eq. 12):
    L_var-reg = β · ‖ṽ_θ‖²₂

  Heteroscedastic NLL (Eq. 13):
    L_diff = E_{t,x₀,ε} [ ½ ‖ε - ε_θ‖² ⊘ σ²_θ  +  ½ Σ_j log σ²_{θ,j} ]
             + L_var-reg

This is a proper scoring rule: the log-variance term penalises over-dispersion
and discourages trivial uncertainty inflation, promoting calibration.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


class HeteroscedasticDiffusionLoss(nn.Module):
    """
    Heteroscedastic negative log-likelihood for diffusion training — Eq. (13).

    Proper scoring rule: minimised when predicted σ²_θ matches the true
    residual variance.  Trivial inflation of σ² is penalised by the
    log-variance term.

    Parameters
    ----------
    sigma_min   : minimum σ for clipping (Eq. 11). Default 0.01
    sigma_max   : maximum σ for clipping (Eq. 11). Default 1.0
    beta        : variance regularisation weight (Eq. 12). Default 1e-4
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        beta: float = 1e-4,
    ) -> None:
        super().__init__()
        self.log_var_min = math.log(sigma_min ** 2)            # log σ²_min
        self.log_var_max = math.log(sigma_max ** 2)            # log σ²_max
        self.beta = beta

    def forward(
        self,
        eps: torch.Tensor,
        eps_theta: torch.Tensor,
        v_theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute L_diff = L_nll + L_var_reg  (Eq. 13).

        Parameters
        ----------
        eps       : [B, c, h, w, d]  ground-truth noise ε
        eps_theta : [B, c, h, w, d]  predicted noise ε_θ
        v_theta   : [B, N]           raw log-variance prediction (N = h·w·d)

        Returns
        -------
        loss     : scalar             L_diff = L_nll + L_var_reg
        v_tilde  : [B, N]            clamped log-variance (for URA gate)
        logs     : dict               component losses for monitoring
        """
        B, c = eps.shape[:2]
        N = v_theta.shape[1]

        # ── Eq. 11: Bounded variance parameterisation ────────────────
        v_tilde = v_theta.clamp(self.log_var_min, self.log_var_max)  # [B, N]
        sigma_sq = torch.exp(v_tilde)                                # [B, N]

        # ── Flatten spatial dims for per-token computation ────────────
        # eps, eps_theta: [B, c, h, w, d] → [B, c, N]
        eps_flat = eps.reshape(B, c, N)                        # [B, c, N]
        eps_theta_flat = eps_theta.reshape(B, c, N)            # [B, c, N]

        # ── Per-token MSE summed over channels ───────────────────────
        # ‖ε - ε_θ‖² per token (sum over c): [B, N]
        residual_sq = (eps_flat - eps_theta_flat).pow(2)       # [B, c, N]
        mse_per_token = residual_sq.sum(dim=1)                 # [B, N]

        # ── Eq. 13: Heteroscedastic NLL ──────────────────────────────
        # L_nll = ½ · mean( mse/σ² + log σ² )
        # The ⊘ (element-wise division) weights each token by its
        # predicted precision, and log σ² penalises over-dispersion.
        nll_per_token = 0.5 * (mse_per_token / sigma_sq + v_tilde)  # [B, N]
        l_nll = nll_per_token.mean()                           # scalar

        # ── Eq. 12: Variance regulariser ─────────────────────────────
        l_var_reg = self.beta * v_tilde.pow(2).mean()          # scalar

        # ── Total loss ────────────────────────────────────────────────
        loss = l_nll + l_var_reg

        # ── Logging ──────────────────────────────────────────────────
        with torch.no_grad():
            logs = {
                "l_nll": l_nll.item(),
                "l_var_reg": l_var_reg.item(),
                "l_diff": loss.item(),
                "v_tilde_mean": v_tilde.mean().item(),
                "v_tilde_std": v_tilde.std().item(),
                "sigma_sq_mean": sigma_sq.mean().item(),
                "mse_mean": mse_per_token.mean().item(),
            }

        return loss, v_tilde, logs

    def extra_repr(self) -> str:
        return (f"log_var_range=[{self.log_var_min:.2f}, {self.log_var_max:.2f}], "
                f"beta={self.beta}")


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Comprehensive tests for HeteroscedasticDiffusionLoss."""

    print("=" * 70)
    print("  HeteroscedasticDiffusionLoss — Test Suite (Eqs. 11–13)")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B, c, h, w, d = 2, 4, 8, 8, 8
    N = h * w * d  # 512

    criterion = HeteroscedasticDiffusionLoss(
        sigma_min=0.01, sigma_max=1.0, beta=1e-4,
    ).to(device)

    print(f"\n  Device: {device}")
    print(f"  Shape: eps=[{B},{c},{h},{w},{d}]  v_theta=[{B},{N}]")
    print(f"  {criterion.extra_repr()}")

    # ── Test 1: Output shapes and types ───────────────────────────────
    print("\n--- (1) Output shapes ---")
    eps       = torch.randn(B, c, h, w, d, device=device)
    eps_theta = torch.randn(B, c, h, w, d, device=device)
    v_theta   = torch.randn(B, N, device=device)

    loss, v_tilde, logs = criterion(eps, eps_theta, v_theta)

    assert loss.shape == (), f"Loss not scalar: {loss.shape}"
    assert v_tilde.shape == (B, N), f"v_tilde shape: {v_tilde.shape}"
    assert isinstance(logs, dict)
    print(f"  loss: scalar = {loss.item():.4f}  ✓")
    print(f"  v_tilde: {list(v_tilde.shape)}  ✓")
    print(f"  logs keys: {list(logs.keys())}  ✓")

    # ── Test 2: Clipping bounds (Eq. 11) ──────────────────────────────
    print("\n--- (2) Clipping bounds (Eq. 11) ---")
    log_var_min = math.log(0.01 ** 2)  # -9.21
    log_var_max = math.log(1.0 ** 2)   # 0.0

    v_extreme = torch.tensor([[-100.0, 100.0, 0.0, -5.0]], device=device)
    # Expand to match expected shape
    v_ex = v_extreme.expand(1, 4)  # [1, 4]
    eps_small = torch.randn(1, c, 1, 2, 2, device=device)  # N=4
    et_small = torch.randn(1, c, 1, 2, 2, device=device)

    _, vt_clipped, _ = criterion(eps_small, et_small, v_ex)

    vt_vals = vt_clipped[0].cpu().tolist()
    print(f"  Input:   {v_extreme[0].tolist()}")
    print(f"  Clamped: {[f'{v:.4f}' for v in vt_vals]}")
    assert abs(vt_vals[0] - log_var_min) < 1e-4, f"Min clip failed: {vt_vals[0]}"
    assert abs(vt_vals[1] - log_var_max) < 1e-4, f"Max clip failed: {vt_vals[1]}"
    assert abs(vt_vals[2] - 0.0) < 1e-4, f"Passthrough failed: {vt_vals[2]}"
    assert abs(vt_vals[3] - (-5.0)) < 1e-4, f"In-range failed: {vt_vals[3]}"
    print(f"  ✓ Clipping to [{log_var_min:.2f}, {log_var_max:.2f}] correct")

    # ── Test 3: Proper scoring rule — inflated σ increases loss ───────
    print("\n--- (3) Proper scoring rule ---")
    eps_ps = torch.randn(B, c, h, w, d, device=device)
    eps_theta_ps = eps_ps + 0.1 * torch.randn_like(eps_ps)    # small residual

    # Calibrated σ²: match the actual residual variance
    residual = (eps_ps - eps_theta_ps).reshape(B, c, N)
    true_mse = residual.pow(2).sum(dim=1)                      # [B, N]
    v_calibrated = torch.log(true_mse.clamp(min=1e-8))        # [B, N]

    # Inflated σ² (10× too large)
    v_inflated = v_calibrated + math.log(10.0)                 # [B, N]

    # Deflated σ² (10× too small)
    v_deflated = v_calibrated - math.log(10.0)                 # [B, N]

    loss_cal, _, _   = criterion(eps_ps, eps_theta_ps, v_calibrated)
    loss_inf, _, _   = criterion(eps_ps, eps_theta_ps, v_inflated)
    loss_def, _, _   = criterion(eps_ps, eps_theta_ps, v_deflated)

    print(f"  L(calibrated σ²): {loss_cal.item():.4f}")
    print(f"  L(inflated 10×):  {loss_inf.item():.4f}")
    print(f"  L(deflated 10×):  {loss_def.item():.4f}")
    assert loss_inf.item() > loss_cal.item(), \
        "FAIL: Inflated σ² should increase loss (proper scoring rule)"
    assert loss_def.item() > loss_cal.item(), \
        "FAIL: Deflated σ² should increase loss (proper scoring rule)"
    print(f"  ✓ Both inflated and deflated σ² increase loss")
    print(f"  ✓ Proper scoring rule verified")

    # ── Test 4: Trivial case — perfect prediction ─────────────────────
    print("\n--- (4) Trivial case: ε_θ = ε exactly ---")
    eps_triv = torch.randn(B, c, h, w, d, device=device)
    v_triv = torch.zeros(B, N, device=device)                 # σ² = 1

    loss_triv, _, logs_triv = criterion(eps_triv, eps_triv, v_triv)
    print(f"  Loss (perfect pred, σ²=1): {loss_triv.item():.6f}")
    print(f"  MSE: {logs_triv['mse_mean']:.6f}")
    # With MSE=0 and v_tilde=0: L = 0.5 * (0/1 + 0) + β*0 = 0
    assert loss_triv.item() < 0.01, f"Perfect prediction loss too high: {loss_triv.item()}"
    print(f"  ✓ Loss ≈ 0 for perfect prediction")

    # ── Test 5: Gradient flow ─────────────────────────────────────────
    print("\n--- (5) Gradient flow ---")
    eps_g = torch.randn(B, c, h, w, d, device=device)
    et_g  = torch.randn(B, c, h, w, d, device=device, requires_grad=True)
    vt_g  = torch.randn(B, N, device=device, requires_grad=True)

    loss_g, _, _ = criterion(eps_g, et_g, vt_g)
    loss_g.backward()

    assert et_g.grad is not None and et_g.grad.abs().sum() > 0
    assert vt_g.grad is not None and vt_g.grad.abs().sum() > 0
    print(f"  ε_θ grad norm: {et_g.grad.norm().item():.4f}  ✓")
    print(f"  v_θ grad norm: {vt_g.grad.norm().item():.4f}  ✓")

    # ── Test 6: v_theta gradient sign check ───────────────────────────
    print("\n--- (6) v_theta gradient sign (proper likelihood) ---")
    # Use tiny residual so MSE << σ² when σ² is large
    eps_s = torch.randn(1, c, h, w, d, device=device)
    et_s  = eps_s + 0.01 * torch.randn_like(eps_s)            # tiny residual

    # Very small σ² (v = -8, within clamp range [-9.21, 0])
    # MSE/σ² >> 1 → gradient pushes v up (∂L/∂v < 0)
    v_small = torch.full((1, N), -8.0, device=device, requires_grad=True)
    loss_s, _, _ = criterion(eps_s, et_s, v_small)
    loss_s.backward()
    grad_small = v_small.grad.mean().item()

    # Moderately large σ² (v = -1.0, well within clamp range)
    # MSE/σ² << 1 → log σ² term dominates → gradient pushes v down (∂L/∂v > 0)
    v_large = torch.full((1, N), -1.0, device=device, requires_grad=True)
    loss_l, _, _ = criterion(eps_s, et_s, v_large)
    loss_l.backward()
    grad_large = v_large.grad.mean().item()

    print(f"  v=-8 (small σ²): grad_mean = {grad_small:.4f}  (expect < 0)")
    print(f"  v=-1 (large σ²): grad_mean = {grad_large:.4f}  (expect > 0)")
    assert grad_small < 0, f"Small σ² gradient should be negative: {grad_small}"
    assert grad_large > 0, f"Large σ² gradient should be positive: {grad_large}"
    print(f"  ✓ Gradient pushes σ² toward calibration")

    # ── Test 7: Variance regulariser (Eq. 12) ─────────────────────────
    print("\n--- (7) Variance regulariser (Eq. 12) ---")
    eps_r = torch.randn(B, c, h, w, d, device=device)
    et_r  = eps_r.clone()                                      # perfect prediction

    # Large v → large regulariser
    v_zero = torch.zeros(B, N, device=device)
    v_big  = torch.full((B, N), -5.0, device=device)          # within bounds

    _, _, logs_zero = criterion(eps_r, et_r, v_zero)
    _, _, logs_big  = criterion(eps_r, et_r, v_big)

    print(f"  L_var_reg(v=0):  {logs_zero['l_var_reg']:.6f}")
    print(f"  L_var_reg(v=-5): {logs_big['l_var_reg']:.6f}")
    assert logs_big['l_var_reg'] > logs_zero['l_var_reg']
    print(f"  ✓ Regulariser penalises large |v_tilde|")

    # ── Test 8: Batch consistency ─────────────────────────────────────
    print("\n--- (8) Batch consistency ---")
    eps_1 = torch.randn(1, c, h, w, d, device=device)
    et_1  = torch.randn(1, c, h, w, d, device=device)
    v_1   = torch.randn(1, N, device=device)

    loss_1, vt_1, _ = criterion(eps_1, et_1, v_1)

    # Double batch (repeat)
    eps_2 = eps_1.repeat(2, 1, 1, 1, 1)
    et_2  = et_1.repeat(2, 1, 1, 1, 1)
    v_2   = v_1.repeat(2, 1)

    loss_2, vt_2, _ = criterion(eps_2, et_2, v_2)

    err = abs(loss_1.item() - loss_2.item())
    print(f"  Loss B=1: {loss_1.item():.6f}")
    print(f"  Loss B=2: {loss_2.item():.6f}")
    print(f"  Difference: {err:.2e}")
    assert err < 1e-4, f"Batch inconsistency: {err}"
    print(f"  ✓ Batch-consistent (mean reduction)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ALL 8 TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
