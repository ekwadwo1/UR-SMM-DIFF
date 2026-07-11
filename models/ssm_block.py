#!/usr/bin/env python3
"""
models/ssm_block.py — Axis-Cycled SSM Block with Local Refinement (Phase 6-A)
==============================================================================

Paper Eqs. (14)–(15), Section 3.2:

  Global SSM mixing (Eq. 14):
    H ← H + π_k^{-1}( SSM( π_k(H̃); t, z^obs ) )

  Local refinement (Eq. 15):
    H ← H + DWConv3D(H)

  Scan directions k ∈ {x, y, z, −x, −y, −z} are cycled across blocks.

The block accepts token features H ∈ ℝ^{B×N×d_h} where N = h·w·d (latent
spatial tokens) and returns updated features of the same shape.

Conditioning:
  - Timestep: sinusoidal embedding projected to d_h, added to H
  - z_obs: projected via Linear(c, d_h), added to H (Level 1) and used
    as cross-gate after SSM scan (Level 2)

CRITICAL: Do NOT apply compile() from torch on this module — mamba-ssm custom
CUDA kernels are incompatible.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================= #
#  Scan direction permutations  π_k / π_k^{-1}                            #
# ======================================================================= #

# Direction schedule: cycled with block_index % 6
SCAN_DIRECTIONS = ["x", "y", "z", "-x", "-y", "-z"]


def _get_perm_order(
    direction: str, h: int, w: int, d: int, device: torch.device,
) -> torch.Tensor:
    """
    Compute the 1D index permutation for a given scan direction.

    Returns perm such that H_perm = H[:, perm, :] reorders tokens from
    default (x,y,z) spatial order into the specified scan order.

    perm[scan_pos] = spatial_pos  (gather index)

    Parameters
    ----------
    direction : one of 'x', 'y', 'z', '-x', '-y', '-z'
    h, w, d   : spatial dimensions
    device    : torch device

    Returns
    -------
    perm : [N] LongTensor
    """
    N = h * w * d
    s = torch.arange(N, device=device)  # scan positions

    if direction in ("x", "-x"):
        # Default (x,y,z) order → identity
        # scan visits: i varies slowest, j next, k fastest (same as spatial)
        perm = s

    elif direction in ("y", "-y"):
        # Scan visits y-slices first: (j, i, k) order
        # scan_pos s: j = s // (h*d), i = (s // d) % h, k = s % d
        j = s // (h * d)
        i = (s // d) % h
        k = s % d
        perm = i * (w * d) + j * d + k       # spatial position

    elif direction in ("z", "-z"):
        # Scan visits z-slices first: (k, i, j) order
        # scan_pos s: k = s // (h*w), i = (s // w) % h, j = s % w
        k = s // (h * w)
        i = (s // w) % h
        j = s % w
        perm = i * (w * d) + j * d + k       # spatial position

    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    # Reverse scan order for negative directions
    if direction.startswith("-"):
        perm = perm.flip(0)

    return perm


def pi_k(
    H: torch.Tensor, spatial_shape: Tuple[int, int, int], direction: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute tokens into 1D scan order for direction k.

    Parameters
    ----------
    H              : [B, N, d_h]  token features
    spatial_shape  : (h, w, d)
    direction      : scan direction string

    Returns
    -------
    H_perm : [B, N, d_h]  permuted tokens
    perm   : [N] index tensor (needed for inverse)
    """
    h, w, d = spatial_shape
    perm = _get_perm_order(direction, h, w, d, H.device)    # [N]
    H_perm = H[:, perm, :]                                   # [B, N, d_h]
    return H_perm, perm


def pi_k_inv(
    H_perm: torch.Tensor, perm: torch.Tensor,
) -> torch.Tensor:
    """
    Inverse permutation: restore original token order.

    Parameters
    ----------
    H_perm : [B, N, d_h]  permuted tokens
    perm   : [N] forward permutation indices

    Returns
    -------
    H : [B, N, d_h]  tokens in original order
    """
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
    return H_perm[:, inv_perm, :]                            # [B, N, d_h]


# ======================================================================= #
#  Depthwise 3D Convolution  (Eq. 15)                                      #
# ======================================================================= #

class DWConv3D(nn.Module):
    """
    Depthwise 3×3×3 convolution for local boundary refinement — Eq. (15).

    H ← H + DWConv3D(H)

    Operates on token features by reshaping [B, N, d_h] → [B, d_h, h, w, d],
    applying grouped convolution, and reshaping back.
    """

    def __init__(self, d_h: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            d_h, d_h, kernel_size=3, padding=1, groups=d_h, bias=True,
        )

    def forward(
        self, H: torch.Tensor, spatial_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        H : [B, N, d_h] → [B, N, d_h]  (local refinement residual)
        """
        B, N, d_h = H.shape
        h, w, d = spatial_shape

        # Reshape to volumetric: [B, d_h, h, w, d]
        H_vol = H.transpose(1, 2).reshape(B, d_h, h, w, d)   # [B, d_h, h, w, d]
        H_vol = self.conv(H_vol)                               # [B, d_h, h, w, d]
        return H_vol.reshape(B, d_h, N).transpose(1, 2)       # [B, N, d_h]


# ======================================================================= #
#  Sinusoidal Timestep Embedding                                           #
# ======================================================================= #

class SinusoidalTimestepEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timestep t.

    Maps integer timestep t ∈ {0, ..., T-1} to a d_h-dimensional vector.
    """

    def __init__(self, d_h: int) -> None:
        super().__init__()
        self.d_h = d_h
        self.proj = nn.Sequential(
            nn.Linear(d_h, d_h * 4),
            nn.SiLU(inplace=True),
            nn.Linear(d_h * 4, d_h),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : [B] integer timesteps → [B, d_h] embedding
        """
        half = self.d_h // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )                                                      # [half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)     # [B, half]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)      # [B, d_h]
        if self.d_h % 2 == 1:
            emb = F.pad(emb, (0, 1))                           # pad if odd
        return self.proj(emb)                                  # [B, d_h]


# ======================================================================= #
#  z_obs Conditioning Projector                                            #
# ======================================================================= #

class ZObsProjector(nn.Module):
    """
    Project observed latent z_obs from channel dim c to token dim d_h.

    z_obs : [B, c, h, w, d] → z_obs_proj : [B, N, d_h]

    Used for:
      Level 1 (additive): H = H + z_obs_proj
      Level 2 (cross-gate): gate = σ(W · z_obs_proj)
    """

    def __init__(self, latent_dim: int, d_h: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, d_h)

    def forward(self, z_obs: torch.Tensor) -> torch.Tensor:
        """
        z_obs : [B, c, h, w, d] → [B, N, d_h]
        """
        B, c, h, w, d = z_obs.shape
        N = h * w * d
        # Reshape: [B, c, h, w, d] → [B, N, c]
        z_flat = z_obs.reshape(B, c, N).permute(0, 2, 1)      # [B, N, c]
        return self.proj(z_flat)                               # [B, N, d_h]


# ======================================================================= #
#  AxisCycledSSMBlock  (Eqs. 14–15)                                        #
# ======================================================================= #

class AxisCycledSSMBlock(nn.Module):
    """
    Axis-cycled SSM block with local refinement — Eqs. (14)–(15).

    Each block performs exactly ONE scan direction, cycled across depth:
      k = directions[block_idx % 6]

    Forward (Eq. 14):
      1. Condition H with timestep embedding and z_obs (additive)
      2. Permute tokens to 1D scan order: H_perm = π_k(H_cond)
      3. Apply Mamba selective scan: H_scanned = SSM(H_perm)
      4. Inverse permute: H_delta = π_k^{-1}(H_scanned)
      5. Cross-gate with z_obs: H_delta = H_delta * σ(W·z_obs_proj)
      6. Local refinement (Eq. 15): H_local = DWConv3D(H + H_delta)
      7. Return H + H_delta + H_local

    Parameters
    ----------
    d_h      : token feature dimension
    d_state  : SSM state dimension (default 16)
    d_conv   : SSM conv width (default 4)
    expand   : SSM expansion factor (default 2)
    """

    def __init__(
        self,
        d_h: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_h = d_h

        # Import Mamba here to isolate the dependency
        from mamba_ssm import Mamba

        # Pre-scan layer norm
        self.norm = nn.LayerNorm(d_h)

        # Mamba selective scan — NOT compatible with compilation
        self.ssm = Mamba(
            d_model=d_h,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Cross-gating projection for z_obs conditioning (Level 2)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.Sigmoid(),
        )

        # Local refinement path — Eq. (15)
        self.local_conv = DWConv3D(d_h)

        # Post-scan layer norm
        self.norm_local = nn.LayerNorm(d_h)

    def forward(
        self,
        H: torch.Tensor,
        t_emb: torch.Tensor,
        z_obs_proj: torch.Tensor,
        block_idx: int,
        spatial_shape: Tuple[int, int, int],
        return_delta: bool = False,
    ) -> torch.Tensor:
        """
        Axis-cycled SSM block forward — Eqs. (14)–(15).

        Parameters
        ----------
        H             : [B, N, d_h]    token features
        t_emb         : [B, d_h]       timestep embedding
        z_obs_proj    : [B, N, d_h]    projected z_obs conditioning
        block_idx     : int             block index for direction cycling
        spatial_shape : (h, w, d)       latent spatial dimensions
        return_delta  : bool            if True, return (H_delta, H_local)
                                        for URA gate composition

        Returns
        -------
        If return_delta=False:
            H_out : [B, N, d_h]    H + H_delta + H_local
        If return_delta=True:
            (H_delta, H_local) : both [B, N, d_h]
        """
        # Determine scan direction for this block
        direction = SCAN_DIRECTIONS[block_idx % 6]

        # ── Step 1: Condition ─────────────────────────────────────────
        # Timestep: broadcast [B, d_h] → [B, 1, d_h] and add
        # z_obs: already [B, N, d_h], add directly
        H_cond = H + t_emb.unsqueeze(1) + z_obs_proj          # [B, N, d_h]
        H_cond = self.norm(H_cond)                             # layer norm

        # ── Step 2: Permute to scan order ─────────────────────────────
        H_perm, perm = pi_k(H_cond, spatial_shape, direction)  # [B, N, d_h]

        # ── Step 3: Mamba selective scan ──────────────────────────────
        H_scanned = self.ssm(H_perm)                          # [B, N, d_h]

        # ── Step 4: Inverse permute ───────────────────────────────────
        H_delta = pi_k_inv(H_scanned, perm)                   # [B, N, d_h]

        # ── Step 5: Cross-gate with z_obs (Level 2 conditioning) ──────
        gate = self.gate_proj(z_obs_proj)                      # [B, N, d_h]
        H_delta = H_delta * gate                               # [B, N, d_h]

        # ── Step 6: Local refinement — Eq. (15) ──────────────────────
        H_pre_local = H + H_delta                              # residual
        H_local = self.local_conv(
            self.norm_local(H_pre_local), spatial_shape
        )                                                      # [B, N, d_h]

        # ── Step 7: Return ────────────────────────────────────────────
        if return_delta:
            return H_delta, H_local                            # for URA gate

        return H + H_delta + H_local                           # Eqs. 14 + 15


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Shape, permutation, and gradient tests for AxisCycledSSMBlock."""

    print("=" * 70)
    print("  AxisCycledSSMBlock — Test Suite (Eqs. 14–15)")
    print("=" * 70)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B = 2
    d_h = 128
    h = w = d = 32    # latent spatial shape for r=4
    N = h * w * d     # 32768
    c = 4             # latent channels

    print(f"\n  Device: {device}")
    print(f"  Spatial: ({h},{w},{d})  N={N}  d_h={d_h}")

    # ── Test (a): Permutation round-trip ──────────────────────────────
    print("\n--- (a) Permutation round-trip (all 6 directions) ---")
    H_test = torch.randn(1, N, d_h, device=device)
    for direction in SCAN_DIRECTIONS:
        H_p, perm = pi_k(H_test, (h, w, d), direction)
        H_rec = pi_k_inv(H_p, perm)
        err = (H_rec - H_test).abs().max().item()
        assert err < 1e-6, f"Permutation round-trip failed for {direction}: {err}"
        print(f"  {direction:3s}: round-trip error = {err:.2e}  ✓")
    del H_test

    # ── Test (b): All permutations are true permutations ──────────────
    print("\n--- (b) Permutation uniqueness ---")
    for direction in SCAN_DIRECTIONS:
        perm = _get_perm_order(direction, h, w, d, torch.device(device))
        assert perm.shape == (N,), f"Wrong perm shape: {perm.shape}"
        unique = perm.unique()
        assert unique.shape[0] == N, f"{direction}: not a permutation! {unique.shape[0]} unique of {N}"
        print(f"  {direction:3s}: {N} unique indices  ✓")

    # ── Test (c): SSM block shape check ───────────────────────────────
    print("\n--- (c) AxisCycledSSMBlock forward shape ---")

    if device == "cpu":
        print("  ⚠ Skipping SSM test on CPU (mamba-ssm requires CUDA)")
    else:
        block = AxisCycledSSMBlock(d_h=d_h, d_state=16, d_conv=4, expand=2).to(device)

        # Build conditioning
        t_emb_module = SinusoidalTimestepEmb(d_h).to(device)
        z_obs_proj_module = ZObsProjector(c, d_h).to(device)

        H = torch.randn(B, N, d_h, device=device)
        t = torch.randint(0, 1000, (B,), device=device)
        z_obs = torch.randn(B, c, h, w, d, device=device)

        t_emb = t_emb_module(t)                                # [B, d_h]
        z_obs_proj = z_obs_proj_module(z_obs)                  # [B, N, d_h]

        with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
            # Test all 6 scan directions
            for idx in range(6):
                H_out = block(H, t_emb, z_obs_proj, idx, (h, w, d))
                assert H_out.shape == (B, N, d_h), \
                    f"Block {idx} shape: {H_out.shape}"
                print(f"  block_idx={idx} dir={SCAN_DIRECTIONS[idx]:3s}: "
                      f"{list(H_out.shape)}  ✓")
            del H_out

        # ── Test (d): return_delta mode ───────────────────────────────
        print("\n--- (d) return_delta mode ---")
        with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
            H_delta, H_local = block(
                H, t_emb, z_obs_proj, 0, (h, w, d), return_delta=True)
            assert H_delta.shape == (B, N, d_h)
            assert H_local.shape == (B, N, d_h)
            print(f"  H_delta: {list(H_delta.shape)}  ✓")
            print(f"  H_local: {list(H_local.shape)}  ✓")
            del H_delta, H_local

        # ── Test (e): Gradient flow ───────────────────────────────────
        print("\n--- (e) Gradient flow through SSM block ---")
        torch.cuda.empty_cache()
        H_grad = torch.randn(1, N, d_h, device=device, requires_grad=True)
        t_emb_1 = t_emb_module(t[:1])
        z_proj_1 = z_obs_proj_module(z_obs[:1])

        H_out = block(H_grad, t_emb_1, z_proj_1, 0, (h, w, d))
        loss = H_out.sum()
        loss.backward()
        assert H_grad.grad is not None, "No gradient!"
        grad_norm = H_grad.grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.4f}  ✓")
        del H_grad, H_out, loss
        torch.cuda.empty_cache()

        # ── Test (f): DWConv3D standalone ─────────────────────────────
        print("\n--- (f) DWConv3D standalone ---")
        dwconv = DWConv3D(d_h).to(device)
        H_dw = torch.randn(B, N, d_h, device=device)
        H_dw_out = dwconv(H_dw, (h, w, d))
        assert H_dw_out.shape == (B, N, d_h)
        print(f"  DWConv3D: {list(H_dw.shape)} → {list(H_dw_out.shape)}  ✓")
        del dwconv, H_dw, H_dw_out

        # ── Test (g): Timestep embedding ──────────────────────────────
        print("\n--- (g) SinusoidalTimestepEmb ---")
        t_test = torch.tensor([0, 500, 999], device=device)
        emb = t_emb_module(t_test)
        assert emb.shape == (3, d_h)
        # Different timesteps → different embeddings
        assert (emb[0] - emb[1]).abs().sum() > 0
        assert (emb[1] - emb[2]).abs().sum() > 0
        print(f"  t=[0,500,999] → embeddings shape {list(emb.shape)}  ✓")
        print(f"  ‖emb(0) - emb(500)‖ = {(emb[0]-emb[1]).norm():.4f}")
        print(f"  ‖emb(500) - emb(999)‖ = {(emb[1]-emb[2]).norm():.4f}")
        del emb

        # ── Test (h): z_obs projector ─────────────────────────────────
        print("\n--- (h) ZObsProjector ---")
        z_proj_out = z_obs_proj_module(z_obs)
        assert z_proj_out.shape == (B, N, d_h)
        print(f"  z_obs [B,{c},{h},{w},{d}] → proj {list(z_proj_out.shape)}  ✓")
        del z_proj_out

        # ── Parameter count ───────────────────────────────────────────
        n_params = sum(p.numel() for p in block.parameters())
        print(f"\n  SSM block params: {n_params:,}")

        del block, H, t_emb, z_obs_proj, z_obs
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ALL TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
