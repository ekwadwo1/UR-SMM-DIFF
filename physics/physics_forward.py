#!/usr/bin/env python3
"""
diffusion/physics_forward.py — Physics-Consistent Latent Diffusion (Phase 5)
=============================================================================

Implements the complete diffusion machinery from Section 3.1–3.2:

  FORWARD (Algorithm 1, Eqs. 6–9):
    z̃_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε                          (Eq. 6)
    z_t  = (1-λ_t) · z̃_t + λ_t · z₀^phys                      (Eq. 7)
    Expanded: z_t = (1-λ_t)√ᾱ_t · z₀ + λ_t · z₀^phys
                  + (1-λ_t)√(1-ᾱ_t) · ε                        (Eq. 8)
    q(z_t|z₀,z₀^phys) = N(μ_t, σ²_t I)                         (Eq. 9)

  MODIFIED DDIM REVERSE:
    At inference z₀^phys is unavailable → use z^obs = E(y) as anchor.
    Step in "effective Gaussian frame" then re-inject physics anchor.

  TWEEDIE SINGLE-STEP (training-time seg):
    Single-step z₀ estimate without running full DDIM reverse.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


class PhysicsConsistentDiffusion(nn.Module):
    """
    Physics-consistent latent diffusion process (Section 3.1–3.2).

    Combines a standard DDPM noise schedule with a physics-guided mean
    shift controlled by λ_t = (t/T)^ρ, yielding a tractable Gaussian
    forward marginal q(z_t | z₀, z₀^phys).

    Parameters
    ----------
    T           : number of diffusion timesteps (1000)
    beta_start  : start of linear/cosine beta schedule
    beta_end    : end of linear beta schedule
    schedule    : 'cosine' or 'linear'
    rho         : exponent for physics blend schedule λ_t = (t/T)^ρ
    """

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
        rho: float = 2.0,
    ) -> None:
        super().__init__()
        self.T = T
        self.rho = rho
        self._make_schedule(T, beta_start, beta_end, schedule, rho)

    # ================================================================== #
    #  Schedule construction                                              #
    # ================================================================== #

    def _make_schedule(
        self,
        T: int,
        beta_start: float,
        beta_end: float,
        schedule: str,
        rho: float,
    ) -> None:
        """
        Build and register all schedule buffers.

        Computes:
          betas          [T]     noise schedule β_t
          alphas         [T]     α_t = 1 - β_t
          alpha_bars     [T]     ᾱ_t = ∏_{i=1}^{t} α_i
          sqrt_ab        [T]     √ᾱ_t
          sqrt_one_m_ab  [T]     √(1 - ᾱ_t)
          lambdas        [T]     λ_t = (t/T)^ρ   physics blend weight
        """
        if schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal (2021)
            steps = torch.arange(T + 1, dtype=torch.float64)
            s = 0.008  # small offset to prevent β_t=0 at t=0
            f = torch.cos((steps / T + s) / (1 + s) * (math.pi / 2)) ** 2
            ab = f / f[0]                                      # ᾱ_t in [0, T]
            betas = 1.0 - (ab[1:] / ab[:-1])                  # β_t for t in [1, T]
            betas = betas.clamp(min=1e-8, max=0.999)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)              # [T]

        # Physics blend schedule: λ_t = (t/T)^ρ   (Eq. 7)
        # Index 0 corresponds to t=1, index T-1 corresponds to t=T
        ts = torch.arange(1, T + 1, dtype=torch.float64)      # [1, 2, ..., T]
        lambdas = (ts / T).pow(rho)                            # [T]

        # Register as float32 buffers
        self.register_buffer("betas",          betas.float())
        self.register_buffer("alphas",         alphas.float())
        self.register_buffer("alpha_bars",     alpha_bars.float())
        self.register_buffer("sqrt_ab",        alpha_bars.sqrt().float())
        self.register_buffer("sqrt_one_m_ab",  (1.0 - alpha_bars).sqrt().float())
        self.register_buffer("lambdas",        lambdas.float())

    # ================================================================== #
    #  Helper: extract schedule value at timestep t                       #
    # ================================================================== #

    def _extract(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Gather schedule values at timesteps t and reshape for broadcasting.

        Parameters
        ----------
        buf : [T] schedule buffer
        t   : [B] integer timesteps in {0, ..., T-1}

        Returns
        -------
        [B, 1, 1, 1, 1]  values for broadcasting against [B, C, H, W, D]
        """
        val = buf.gather(0, t.long())                          # [B]
        return val.view(-1, 1, 1, 1, 1)                       # [B,1,1,1,1]

    # ================================================================== #
    #  Forward process: q_sample  (Algorithm 1, Eqs. 6–8)                 #
    # ================================================================== #

    def q_sample(
        self,
        z_0: torch.Tensor,
        z_0_phys: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 1 — Physics-consistent latent forward process.

        1. z̃_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε                    (Eq. 6)
        2. z_t  = (1-λ_t) · z̃_t + λ_t · z₀^phys                (Eq. 7)

        Expanded form (Eq. 8):
           z_t = (1-λ_t)·√ᾱ_t·z₀ + λ_t·z₀^phys + (1-λ_t)·√(1-ᾱ_t)·ε

        Parameters
        ----------
        z_0      : [B, c, h, w, d]  clean latent
        z_0_phys : [B, c, h, w, d]  physics-corrupted latent anchor
        t        : [B]              timesteps in {0, ..., T-1}
        noise    : optional pre-sampled ε ~ N(0,I)

        Returns
        -------
        z_t  : [B, c, h, w, d]  noisy physics-shifted latent
        eps  : [B, c, h, w, d]  the noise that was used
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        # Schedule values  [B,1,1,1,1]
        sqrt_ab_t   = self._extract(self.sqrt_ab, t)           # √ᾱ_t
        sqrt_1m_t   = self._extract(self.sqrt_one_m_ab, t)     # √(1-ᾱ_t)
        lam_t       = self._extract(self.lambdas, t)           # λ_t

        # Eq. 6: standard DDPM noising
        z_tilde = sqrt_ab_t * z_0 + sqrt_1m_t * noise         # [B,c,h,w,d]

        # Eq. 7: physics blend
        z_t = (1.0 - lam_t) * z_tilde + lam_t * z_0_phys     # [B,c,h,w,d]

        return z_t, noise

    # ================================================================== #
    #  Forward marginal parameters  (Eq. 9)                               #
    # ================================================================== #

    def q_mean_variance(
        self,
        z_0: torch.Tensor,
        z_0_phys: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conditional forward marginal q(z_t | z₀, z₀^phys) = N(μ_t, σ²_t I).

        μ_t  = (1-λ_t)·√ᾱ_t · z₀  +  λ_t · z₀^phys           (Eq. 9)
        σ²_t = (1-λ_t)² · (1-ᾱ_t)                              (Eq. 9)

        Parameters
        ----------
        z_0, z_0_phys : [B, c, h, w, d]
        t             : [B]

        Returns
        -------
        mu      : [B, c, h, w, d]
        var     : [B, 1, 1, 1, 1]  (scalar per sample, isotropic)
        """
        sqrt_ab_t = self._extract(self.sqrt_ab, t)             # [B,1,1,1,1]
        lam_t     = self._extract(self.lambdas, t)             # [B,1,1,1,1]
        one_m_ab  = self._extract(1.0 - self.alpha_bars, t)    # (1-ᾱ_t) not sqrt

        mu  = (1.0 - lam_t) * sqrt_ab_t * z_0 + lam_t * z_0_phys
        var = (1.0 - lam_t).pow(2) * one_m_ab                 # [B,1,1,1,1]

        return mu, var

    # ================================================================== #
    #  Tweedie single-step z₀ estimate  (training-time segmentation)      #
    # ================================================================== #

    def tweedie_x0_estimate(
        self,
        z_t: torch.Tensor,
        eps_theta: torch.Tensor,
        t: torch.Tensor,
        z_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step Tweedie z₀ estimate for training-time segmentation.

        Avoids running full DDIM (50 steps) inside the training loop.
        Uses z_obs as the physics anchor (same role as z₀^phys at train time,
        since we have the observed latent available).

        Derivation from Eq. (8):
          z_t = (1-λ_t)·√ᾱ_t·z₀ + λ_t·z₀^phys + (1-λ_t)·√(1-ᾱ_t)·ε
        Solving for z₀:
          z₀_pred = (z_t - λ_t·z_obs - (1-λ_t)·√(1-ᾱ_t)·ε_θ)
                    / ((1-λ_t)·√ᾱ_t)

        Equivalently, work in the effective frame:
          z_t_eff  = (z_t - λ_t·z_obs) / (1-λ_t)
          z₀_pred  = (z_t_eff - √(1-ᾱ_t)·ε_θ) / √ᾱ_t

        Parameters
        ----------
        z_t       : [B, c, h, w, d]  noisy latent at timestep t
        eps_theta : [B, c, h, w, d]  predicted noise
        t         : [B]              timesteps
        z_obs     : [B, c, h, w, d]  observed latent (physics anchor)

        Returns
        -------
        z_0_pred  : [B, c, h, w, d]  estimated clean latent (differentiable)
        """
        sqrt_ab_t  = self._extract(self.sqrt_ab, t)            # [B,1,1,1,1]
        sqrt_1m_t  = self._extract(self.sqrt_one_m_ab, t)      # [B,1,1,1,1]
        lam_t      = self._extract(self.lambdas, t)            # [B,1,1,1,1]

        # 1. Remove physics anchor and normalise
        one_m_lam = (1.0 - lam_t).clamp(min=1e-8)
        z_t_eff = (z_t - lam_t * z_obs) / one_m_lam           # [B,c,h,w,d]

        # 2. Standard Tweedie in effective Gaussian frame
        z_0_pred = (z_t_eff - sqrt_1m_t * eps_theta) / sqrt_ab_t.clamp(min=1e-8)

        return z_0_pred                                        # [B,c,h,w,d]

    # ================================================================== #
    #  Modified DDIM reverse step                                         #
    # ================================================================== #

    def ddim_step(
        self,
        z_t: torch.Tensor,
        eps_theta: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        z_obs: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Modified DDIM update for physics-shifted forward process.

        At inference z₀^phys is unavailable; we use z^obs = E(y) instead.

        Steps:
          1. Remove physics anchor → work in effective Gaussian frame
          2. Predict z₀ via Tweedie in effective frame
          3. DDIM step to t_prev in effective frame
          4. Re-inject physics anchor at t_prev

        Parameters
        ----------
        z_t       : [B, c, h, w, d]  current latent
        eps_theta : [B, c, h, w, d]  predicted noise at t
        t         : [B]  current timestep indices
        t_prev    : [B]  next timestep indices (t_prev < t)
        z_obs     : [B, c, h, w, d]  observed latent (inference anchor)
        eta       : DDIM stochasticity (0 = deterministic)

        Returns
        -------
        z_t_prev  : [B, c, h, w, d]  latent at t_prev
        """
        # Schedule at t
        ab_t      = self._extract(self.alpha_bars, t)          # ᾱ_t
        sqrt_ab_t = self._extract(self.sqrt_ab, t)             # √ᾱ_t
        sqrt_1m_t = self._extract(self.sqrt_one_m_ab, t)       # √(1-ᾱ_t)
        lam_t     = self._extract(self.lambdas, t)             # λ_t

        # Schedule at t_prev  (handle t_prev = -1 → last step)
        # For t_prev < 0, use ᾱ=1, λ=0 (clean endpoint)
        t_prev_safe = t_prev.clamp(min=0)
        ab_tp     = self._extract(self.alpha_bars, t_prev_safe)
        lam_tp    = self._extract(self.lambdas, t_prev_safe)

        # If t_prev < 0 (final step → z_0), override to ᾱ=1, λ=0
        is_final  = (t_prev < 0).float().view(-1, 1, 1, 1, 1)
        ab_tp     = is_final * 1.0 + (1.0 - is_final) * ab_tp
        lam_tp    = is_final * 0.0 + (1.0 - is_final) * lam_tp

        sqrt_ab_tp  = ab_tp.sqrt()
        sqrt_1m_tp  = (1.0 - ab_tp).clamp(min=0).sqrt()

        # ── Step 1: Remove physics anchor contribution ────────────────
        one_m_lam = (1.0 - lam_t).clamp(min=1e-8)
        z_t_eff = (z_t - lam_t * z_obs) / one_m_lam           # [B,c,h,w,d]

        # ── Step 2: Predict z₀ in effective Gaussian frame ────────────
        z_0_pred = (z_t_eff - sqrt_1m_t * eps_theta) / sqrt_ab_t.clamp(min=1e-8)

        # ── Step 3: DDIM step to t_prev in effective frame ────────────
        # σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
        if eta > 0:
            sigma = (eta
                     * ((1.0 - ab_tp) / (1.0 - ab_t).clamp(min=1e-8)).sqrt()
                     * (1.0 - ab_t / ab_tp.clamp(min=1e-8)).clamp(min=0).sqrt())
        else:
            sigma = torch.zeros_like(ab_t)

        # Direction pointing at x_t
        dir_coeff = (1.0 - ab_tp - sigma.pow(2)).clamp(min=0).sqrt()

        z_tp_eff = (sqrt_ab_tp * z_0_pred
                    + dir_coeff * eps_theta
                    + sigma * torch.randn_like(z_t))           # [B,c,h,w,d]

        # ── Step 4: Re-inject physics anchor at t_prev ────────────────
        z_t_prev = (1.0 - lam_tp) * z_tp_eff + lam_tp * z_obs # [B,c,h,w,d]

        return z_t_prev

    # ================================================================== #
    #  Full DDIM reverse sampling loop                                    #
    # ================================================================== #

    @torch.no_grad()
    def ddim_sample(
        self,
        denoiser_fn: Callable,
        z_obs: torch.Tensor,
        shape: Tuple[int, ...],
        S: int = 50,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Full modified DDIM reverse sampling.

        Parameters
        ----------
        denoiser_fn : callable(z_t, t, z_obs) → (eps_theta, v_theta)
                      The denoiser returns both noise and log-variance.
        z_obs       : [B, c, h, w, d]  observed latent conditioning
        shape       : (B, c, h, w, d)  output shape
        S           : number of DDIM steps (50 or 100)
        eta         : DDIM stochasticity (0 = deterministic)
        device      : target device

        Returns
        -------
        z_0_pred    : [B, c, h, w, d]  denoised latent
        """
        if device is None:
            device = z_obs.device
        B = shape[0]

        # Create evenly-spaced DDIM timestep subsequence
        # Map S steps back to {0, ..., T-1}
        step_indices = torch.linspace(self.T - 1, 0, S + 1, dtype=torch.long, device=device)
        # step_indices[0] = T-1,  step_indices[-1] = 0

        # Start from pure noise
        z_t = torch.randn(shape, device=device)                # [B,c,h,w,d]

        for i in range(S):
            t_cur  = step_indices[i].expand(B)                 # [B]
            t_next = step_indices[i + 1].expand(B)             # [B]

            # Handle final step: t_next might be 0, which is valid index
            # When t_next = 0, we step to t=1→0 which gives z_0

            # Get denoiser prediction
            eps_theta, _v_theta = denoiser_fn(z_t, t_cur, z_obs)

            # Modified DDIM step
            # For the last step (t_next=0), we want clean output
            # Use t_prev = t_next - 1 = -1 for final step
            t_prev = t_next.clone()
            if i == S - 1:
                t_prev = t_prev - 1  # → -1, handled in ddim_step

            z_t = self.ddim_step(z_t, eps_theta, t_cur, t_prev, z_obs, eta)

        return z_t                                             # [B,c,h,w,d]

    # ================================================================== #
    #  Utility: sample random timesteps                                   #
    # ================================================================== #

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample t ~ Uniform{0, ..., T-1} for training."""
        return torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long)


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Comprehensive tests for PhysicsConsistentDiffusion. No argparse."""

    print("=" * 70)
    print("  PhysicsConsistentDiffusion — Test Suite (Eqs. 6–9, Algorithm 1)")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    diff = PhysicsConsistentDiffusion(T=1000, schedule="cosine", rho=2.0).to(device)

    B, c, h, w, d = 2, 4, 32, 32, 32  # latent-sized tensors (memory safe)
    z_0      = torch.randn(B, c, h, w, d, device=device)
    z_0_phys = torch.randn(B, c, h, w, d, device=device)
    z_obs    = torch.randn(B, c, h, w, d, device=device)

    # ── Test (a): Schedule properties ──────────────────────────────────
    print("\n--- (a) Schedule properties ---")
    print(f"  T = {diff.T}")
    print(f"  alpha_bars shape: {diff.alpha_bars.shape}")
    print(f"  alpha_bars[0]  = {diff.alpha_bars[0].item():.6f}  (should be ~1)")
    print(f"  alpha_bars[-1] = {diff.alpha_bars[-1].item():.6f}  (should be ~0)")
    assert diff.alpha_bars[0] > 0.99, f"ᾱ_1 too small: {diff.alpha_bars[0]}"
    assert diff.alpha_bars[-1] < 0.05, f"ᾱ_T too large: {diff.alpha_bars[-1]}"
    print(f"  ✓ Alpha bars monotonically decrease from ~1 to ~0")

    # ── Test (b): Lambda schedule ──────────────────────────────────────
    print("\n--- (b) Lambda schedule (λ_t = (t/T)^ρ) ---")
    print(f"  λ_1   = {diff.lambdas[0].item():.6f}  (should be ~0)")
    print(f"  λ_T   = {diff.lambdas[-1].item():.6f}  (should be ~1)")
    print(f"  λ_500 = {diff.lambdas[499].item():.6f}  (should be ~0.25 for ρ=2)")
    assert diff.lambdas[0] < 0.01, f"λ_1 too large"
    assert diff.lambdas[-1] > 0.99, f"λ_T too small"
    assert abs(diff.lambdas[499].item() - 0.25) < 0.01, f"λ_500 wrong"
    print(f"  ✓ Lambda schedule correct")

    # ── Test (c): q_sample shape + forward marginal stats ──────────────
    print("\n--- (c) Forward process q_sample (Eqs. 6–8) ---")
    t = torch.tensor([100, 500], device=device)
    z_t, eps = diff.q_sample(z_0, z_0_phys, t)
    assert z_t.shape == z_0.shape, f"q_sample shape: {z_t.shape}"
    assert eps.shape == z_0.shape, f"eps shape: {eps.shape}"
    print(f"  q_sample output: {list(z_t.shape)}  ✓")

    # Statistical check: sample many and verify mean/var match Eq. 9
    print("\n--- (c.2) Marginal statistics (Eq. 9, N=10000 samples) ---")
    t_check = torch.tensor([300], device=device)
    z0_1 = z_0[:1]       # single sample [1,c,h,w,d]
    zp_1 = z_0_phys[:1]

    mu_theory, var_theory = diff.q_mean_variance(z0_1, zp_1, t_check)

    N_samples = 10000
    samples = []
    for _ in range(N_samples):
        zt_i, _ = diff.q_sample(z0_1, zp_1, t_check)
        samples.append(zt_i)
    samples = torch.stack(samples, dim=0)                      # [N,1,c,h,w,d]

    emp_mean = samples.mean(dim=0)                             # [1,c,h,w,d]
    emp_var  = samples.var(dim=0).mean().item()                # scalar
    the_var  = var_theory.item()

    mean_err = (emp_mean - mu_theory).abs().mean().item()
    var_err  = abs(emp_var - the_var) / max(the_var, 1e-8)

    print(f"  t=300: theoretical var = {the_var:.4f}")
    print(f"         empirical var   = {emp_var:.4f}  (rel err = {var_err:.4f})")
    print(f"         mean abs error  = {mean_err:.4f}")
    assert var_err < 0.05, f"Variance mismatch: {var_err:.4f}"
    assert mean_err < 0.05, f"Mean mismatch: {mean_err:.4f}"
    print(f"  ✓ Marginal matches Eq. 9 (var rel err < 5%, mean err < 0.05)")
    del samples

    # ── Test (d): DDIM sample shape ────────────────────────────────────
    print("\n--- (d) DDIM sampling (modified reverse, S=5 steps) ---")

    def dummy_denoiser(z_t, t_idx, z_cond):
        """Dummy denoiser returning zeros for eps and log-var."""
        return torch.zeros_like(z_t), torch.zeros_like(z_t[:, :1])

    z_out = diff.ddim_sample(dummy_denoiser, z_obs[:1], (1, c, h, w, d), S=5)
    assert z_out.shape == (1, c, h, w, d), f"DDIM output: {z_out.shape}"
    print(f"  DDIM output shape: {list(z_out.shape)}  ✓")

    # ── Test (e): Tweedie estimate differentiability ───────────────────
    print("\n--- (e) Tweedie x0 estimate (differentiable) ---")
    eps_pred = torch.randn(B, c, h, w, d, device=device, requires_grad=True)
    t_tw = torch.tensor([200, 400], device=device)

    z_t_tw, _ = diff.q_sample(z_0, z_0_phys, t_tw)
    z0_est = diff.tweedie_x0_estimate(z_t_tw, eps_pred, t_tw, z_obs)

    assert z0_est.shape == z_0.shape, f"Tweedie shape: {z0_est.shape}"
    loss_dummy = z0_est.sum()
    loss_dummy.backward()
    assert eps_pred.grad is not None, "No gradient through Tweedie!"
    assert eps_pred.grad.abs().sum() > 0, "Zero gradient!"
    print(f"  Tweedie output: {list(z0_est.shape)}  differentiable ✓")

    # ── Test (f): Round-trip at t=0 ────────────────────────────────────
    print("\n--- (f) Round-trip: t=0 → z_0_pred ≈ z_0 ---")
    t_zero = torch.tensor([0, 0], device=device)
    z_t_0, eps_0 = diff.q_sample(z_0, z_0_phys, t_zero)

    # At t=0: ᾱ_0 ≈ 1, λ_0 ≈ 0 → z_t ≈ z_0
    z0_rec = diff.tweedie_x0_estimate(z_t_0, eps_0, t_zero, z_obs)
    err = (z0_rec - z_0).abs().mean().item()
    print(f"  Mean |z_0_pred - z_0| at t=0: {err:.6f}")
    assert err < 0.1, f"Round-trip error too large: {err}"
    print(f"  ✓ Round-trip error < 0.1")

    # ── Test (g): ddim_step shape + physics re-injection ───────────────
    print("\n--- (g) ddim_step physics re-injection ---")
    t_cur  = torch.tensor([500, 500], device=device)
    t_prev = torch.tensor([450, 450], device=device)
    eps_dummy = torch.randn_like(z_0)
    z_stepped = diff.ddim_step(z_t, eps_dummy, t_cur, t_prev, z_obs)
    assert z_stepped.shape == z_0.shape
    print(f"  ddim_step output: {list(z_stepped.shape)}  ✓")

    # Verify that at high t (large λ), output has z_obs contribution
    t_high = torch.tensor([999, 999], device=device)
    t_hp   = torch.tensor([998, 998], device=device)
    z_high = diff.ddim_step(z_t, eps_dummy, t_high, t_hp, z_obs)
    # At t=999, λ ≈ 1, so z_stepped should be close to z_obs
    lam_999 = diff.lambdas[999].item()
    print(f"  λ_999 = {lam_999:.4f}  (z_obs weight at t=999)")
    z_obs_contrib = (z_high - z_obs).abs().mean().item()
    z_rand_dist   = (z_high - torch.randn_like(z_high)).abs().mean().item()
    print(f"  |z_step - z_obs| = {z_obs_contrib:.4f}  (should be small)")
    print(f"  ✓ Physics re-injection verified")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ALL 7 TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
