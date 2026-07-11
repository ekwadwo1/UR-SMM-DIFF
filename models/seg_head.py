#!/usr/bin/env python3
"""
models/seg_head.py — Lightweight 3D Segmentation Head (Phase 8-A)
=================================================================

Paper Eq. (19), Section 3.3:
    ŝ₀ = S_φ(ẑ₀) ∈ ℝ^{H×W×D×K}

Lightweight head (~200K params) that takes the denoised latent ẑ₀
at reduced resolution and predicts full-resolution tumor subregion
segmentation maps.

Architecture:
  Conv3d(c, 64, 3)  + GN + GELU
  Conv3d(64, 64, 3) + GN + GELU
  Conv3d(64, 32, 3) + GN + GELU
  Trilinear upsample ×r
  Conv3d(32, K, 1)

K = 4 classes: {BG, NCR/NET, ED, ET}

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead3D(nn.Module):
    """
    Lightweight 3D segmentation head — Eq. (19).

    ŝ₀ = S_φ(ẑ₀)

    Takes denoised latent at reduced resolution [B, c, h, w, d]
    and outputs full-resolution logits [B, K, H, W, D].

    Parameters
    ----------
    latent_dim      : c (input channels from latent, default 4)
    n_classes       : K (number of segmentation classes, default 4)
    hidden_dim      : intermediate feature channels (default 64)
    downsample_factor : r (VQGAN spatial downsampling, default 4)
    """

    def __init__(
        self,
        latent_dim: int = 4,
        n_classes: int = 4,
        hidden_dim: int = 64,
        downsample_factor: int = 4,
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor

        self.layers = nn.Sequential(
            # Block 1: c → 64
            nn.Conv3d(latent_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),

            # Block 2: 64 → 64
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),

            # Block 3: 64 → 32
            nn.Conv3d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim // 2), hidden_dim // 2),
            nn.GELU(),
        )

        # Final classification: 32 → K (after upsampling)
        self.classifier = nn.Conv3d(hidden_dim // 2, n_classes, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : [B, c, h, w, d]  denoised latent (reduced resolution)

        Returns
        -------
        logits : [B, K, H, W, D]  segmentation logits at full resolution
                 H = h·r, W = w·r, D = d·r
        """
        h = self.layers(z)                                     # [B, 32, h, w, d]

        # Upsample to full resolution
        r = self.downsample_factor
        h = F.interpolate(
            h, scale_factor=float(r), mode="trilinear",
            align_corners=False,
        )                                                      # [B, 32, H, W, D]

        return self.classifier(h)                              # [B, K, H, W, D]


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Shape and parameter count tests for SegmentationHead3D."""

    print("=" * 70)
    print("  SegmentationHead3D — Test Suite (Eq. 19)")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    for r in [4, 8]:
        print(f"\n--- r={r} ---")
        h = w = d = 128 // r
        H = W = D = 128
        c, K = 4, 4

        head = SegmentationHead3D(
            latent_dim=c, n_classes=K, hidden_dim=64,
            downsample_factor=r,
        ).to(device)

        n_params = sum(p.numel() for p in head.parameters())
        print(f"  Params: {n_params:,}")

        # Shape test
        z = torch.randn(1, c, h, w, d, device=device)
        with torch.no_grad():
            logits = head(z)
        assert logits.shape == (1, K, H, W, D), f"Shape: {logits.shape}"
        print(f"  Input:  [{1},{c},{h},{w},{d}]")
        print(f"  Output: {list(logits.shape)}  ✓")

        # Gradient test
        z_g = torch.randn(1, c, h, w, d, device=device, requires_grad=True)
        logits_g = head(z_g)
        logits_g.sum().backward()
        assert z_g.grad is not None and z_g.grad.abs().sum() > 0
        print(f"  Gradient flow: ✓")

        # Batch test
        z_b = torch.randn(2, c, h, w, d, device=device)
        with torch.no_grad():
            logits_b = head(z_b)
        assert logits_b.shape == (2, K, H, W, D)
        print(f"  Batch=2: {list(logits_b.shape)}  ✓")

        del head, z, logits, z_g, logits_g, z_b, logits_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Param count check
    head_check = SegmentationHead3D(4, 4, 64, 4)
    n = sum(p.numel() for p in head_check.parameters())
    print(f"\n  Target: ~200K params. Actual: {n:,}")
    assert n < 500_000, f"Too many params: {n:,}"
    print(f"  ✓ Under 500K limit")

    print(f"\n{'=' * 70}")
    print("  ALL TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
