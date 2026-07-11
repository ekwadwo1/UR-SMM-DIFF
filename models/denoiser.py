#!/usr/bin/env python3
"""
models/denoiser.py — UR-SSM-Diff Full Denoiser f_θ  (Phase 6-C)
================================================================

Paper Eq. (10), Section 3.2:
    (ε_θ, v_θ) = f_θ(z_t, t; z^obs)
    v_θ = log σ²_θ ∈ ℝ^N

U-Net-style architecture:
  Encoder   — 3 stages of plain AxisCycledSSMBlocks (no URA)
  Bottleneck — 4 URASSMBlocks at lowest resolution
  Decoder   — 3 stages of URASSMBlocks with skip connections
  Dual heads — eps_head (Conv3d→c) + var_head (Conv3d→1)

Design choice: URA gating applied ONLY in bottleneck + decoder.
Encoder extracts features without uncertainty bias; the variance head
output from the current pass provides v_tilde to decoder blocks.
During training (single pass): v_tilde_ext=None → zeros (g≈1).
During DDIM inference: v_tilde from previous step.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8 · bfloat16 AMP
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn

from models.ssm_block import (
    AxisCycledSSMBlock,
    SinusoidalTimestepEmb,
)
from models.ura_gate import URASSMBlock


# ======================================================================= #
#  Helpers                                                                 #
# ======================================================================= #

def _vol_to_tok(x: torch.Tensor) -> torch.Tensor:
    """[B, C, h, w, d] → [B, N, C] where N=h·w·d."""
    B, C = x.shape[:2]
    return x.reshape(B, C, -1).permute(0, 2, 1)


def _tok_to_vol(x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    """[B, N, C] → [B, C, h, w, d]."""
    B, N, C = x.shape
    h, w, d = shape
    return x.permute(0, 2, 1).reshape(B, C, h, w, d)


class Downsample3D(nn.Module):
    """Strided conv ×2 downsample with channel expansion."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """Trilinear ×2 upsample + Conv3d channel reduction."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="trilinear",
                          align_corners=False)
        return self.conv(x)


# ======================================================================= #
#  Condition Encoder  (multi-scale z_obs features)                         #
# ======================================================================= #

class ConditionEncoder(nn.Module):
    """
    Extract multi-scale features from z_obs for cross-gating.

    Produces features at 3 resolutions matching encoder/decoder stages:
      feat0: [B, d_h,   h,   w,   d  ]
      feat1: [B, 2·d_h, h/2, w/2, d/2]
      feat2: [B, 4·d_h, h/4, w/4, d/4]
    """

    def __init__(self, latent_dim: int, d_h: int) -> None:
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv3d(latent_dim, d_h, 3, padding=1),
            nn.GroupNorm(min(32, d_h), d_h), nn.SiLU(inplace=True))
        self.stage1 = nn.Sequential(
            nn.Conv3d(d_h, 2 * d_h, 3, stride=2, padding=1),
            nn.GroupNorm(min(32, 2*d_h), 2*d_h), nn.SiLU(inplace=True))
        self.stage2 = nn.Sequential(
            nn.Conv3d(2 * d_h, 4 * d_h, 3, stride=2, padding=1),
            nn.GroupNorm(min(32, 4*d_h), 4*d_h), nn.SiLU(inplace=True))

    def forward(self, z_obs: torch.Tensor) -> List[torch.Tensor]:
        f0 = self.stage0(z_obs)                                # [B, d_h, h, w, d]
        f1 = self.stage1(f0)                                   # [B, 2d_h, h/2, ...]
        f2 = self.stage2(f1)                                   # [B, 4d_h, h/4, ...]
        return [f0, f1, f2]


# ======================================================================= #
#  Timestep Embedding (multi-scale projections)                            #
# ======================================================================= #

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal timestep embedding with per-stage linear projections.

    Base embedding: sinusoidal(t) → MLP → base_dim
    Per-stage: Linear(base_dim, stage_dim)
    """

    def __init__(self, d_h: int, n_stages: int = 3) -> None:
        super().__init__()
        base_dim = 4 * d_h  # rich base representation
        self.base = SinusoidalTimestepEmb(base_dim)

        dims = [d_h * (2 ** i) for i in range(n_stages)]      # [d_h, 2d_h, 4d_h]
        self.projs = nn.ModuleList([nn.Linear(base_dim, dim) for dim in dims])

    def forward(self, t: torch.Tensor) -> List[torch.Tensor]:
        """t: [B] → list of [B, stage_dim] for each stage."""
        base = self.base(t)                                    # [B, base_dim]
        return [proj(base) for proj in self.projs]


# ======================================================================= #
#  Encoder / Decoder Stage Runners                                         #
# ======================================================================= #

class EncoderStage(nn.Module):
    """Stage with plain SSM blocks (no URA gating). Eqs. (14)–(15)."""

    def __init__(self, d_model: int, n_blocks: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            AxisCycledSSMBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_blocks)
        ])

    def forward(
        self, H_vol: torch.Tensor, t_emb: torch.Tensor,
        cond_feat: torch.Tensor, block_idx_start: int,
    ) -> torch.Tensor:
        """
        H_vol: [B, C, h, w, d] → [B, C, h, w, d]
        t_emb: [B, C]
        cond_feat: [B, C, h, w, d]
        """
        B, C, h, w, d = H_vol.shape
        spatial = (h, w, d)
        H = _vol_to_tok(H_vol)                                # [B, N, C]
        z_proj = _vol_to_tok(cond_feat)                        # [B, N, C]

        for i, block in enumerate(self.blocks):
            H = block(H, t_emb, z_proj, block_idx_start + i, spatial)

        return _tok_to_vol(H, spatial)                         # [B, C, h, w, d]


class DecoderStage(nn.Module):
    """Stage with URA-SSM blocks. Eqs. (16)–(18)."""

    def __init__(self, d_model: int, n_blocks: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 init_kappa_raw: float = -3.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            URASSMBlock(d_model, d_state, d_conv, expand, init_kappa_raw)
            for _ in range(n_blocks)
        ])

    def forward(
        self, H_vol: torch.Tensor, v_tilde: torch.Tensor,
        t_emb: torch.Tensor, cond_feat: torch.Tensor,
        block_idx_start: int,
    ) -> torch.Tensor:
        """
        H_vol: [B, C, h, w, d]
        v_tilde: [B, N'] — may need to be at this resolution
        """
        B, C, h, w, d = H_vol.shape
        spatial = (h, w, d)
        H = _vol_to_tok(H_vol)                                # [B, N, C]
        z_proj = _vol_to_tok(cond_feat)                        # [B, N, C]

        for i, block in enumerate(self.blocks):
            H = block(H, v_tilde, t_emb, z_proj,
                      block_idx_start + i, spatial)

        return _tok_to_vol(H, spatial)


# ======================================================================= #
#  URSSMDenoiser  f_θ — Eq. (10)                                           #
# ======================================================================= #

class URSSMDenoiser(nn.Module):
    """
    Full UR-SSM-Diff denoiser f_θ — Eq. (10).

    (ε_θ, v_θ) = f_θ(z_t, t; z^obs)

    Architecture:
      Input:     cat([z_t, z_obs], dim=1) → Conv3d(2c, d_h)
      Encoder:   3 stages of plain SSM blocks  [2, 2, 4] blocks
      Bottleneck: 4 URA-SSM blocks at lowest resolution
      Decoder:   3 stages of URA-SSM blocks [4, 2, 2] with skip connections
      Heads:     eps_head Conv3d(d_h, c, 1)
                 var_head Conv3d(d_h, 1, 1) → [B, N]

    URA gating in bottleneck + decoder only. Encoder extracts features
    without uncertainty bias.

    Parameters
    ----------
    latent_dim   : c (4 for our VQGAN)
    d_h          : base hidden dim (128)
    stage_depths : blocks per encoder stage (2, 2, 4)
    bottleneck_depth : URA blocks in bottleneck (4)
    d_state, d_conv, expand : Mamba SSM params
    log_var_range : (min, max) for v_tilde clipping (Eq. 11)
    use_checkpoint : gradient checkpointing per stage
    """

    def __init__(
        self,
        latent_dim: int = 4,
        d_h: int = 128,
        stage_depths: Tuple[int, ...] = (2, 2, 4),
        bottleneck_depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        log_var_range: Tuple[float, float] = (-10.0, 10.0),
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_h = d_h
        self.use_checkpoint = use_checkpoint
        self.log_var_min, self.log_var_max = log_var_range
        self.n_stages = len(stage_depths)

        # Channel dims per stage: [d_h, 2*d_h, 4*d_h]
        self.stage_dims = [d_h * (2 ** i) for i in range(self.n_stages)]

        # ── Input conditioning ────────────────────────────────────────
        # cat([z_t, z_obs], dim=1) → [B, 2c, h, w, d]
        self.input_conv = nn.Sequential(
            nn.Conv3d(2 * latent_dim, d_h, 3, padding=1),
            nn.GroupNorm(min(32, d_h), d_h),
            nn.SiLU(inplace=True),
        )

        # ── Multi-scale z_obs conditioning ────────────────────────────
        self.cond_enc = ConditionEncoder(latent_dim, d_h)

        # ── Timestep embedding (per-stage projections) ────────────────
        self.time_embed = TimestepEmbedder(d_h, self.n_stages)

        # ── Encoder stages (plain SSM, no URA) ────────────────────────
        self.enc_stages = nn.ModuleList()
        for i, depth in enumerate(stage_depths):
            self.enc_stages.append(
                EncoderStage(self.stage_dims[i], depth,
                             d_state, d_conv, expand))

        # ── Encoder downsampling ──────────────────────────────────────
        self.enc_downs = nn.ModuleList()
        for i in range(self.n_stages - 1):
            self.enc_downs.append(
                Downsample3D(self.stage_dims[i], self.stage_dims[i + 1]))

        # ── Bottleneck (URA-SSM) ──────────────────────────────────────
        bot_dim = self.stage_dims[-1]  # 4*d_h
        self.bottleneck = DecoderStage(
            bot_dim, bottleneck_depth, d_state, d_conv, expand)

        # ── Decoder upsampling ────────────────────────────────────────
        self.dec_ups = nn.ModuleList()
        for i in range(self.n_stages - 1, 0, -1):
            self.dec_ups.append(
                Upsample3D(self.stage_dims[i], self.stage_dims[i - 1]))

        # ── Decoder skip projections (concat → project) ──────────────
        self.dec_skip_proj = nn.ModuleList()
        # After bottleneck: concat with enc[-1] skip
        self.dec_skip_proj.append(
            nn.Conv3d(2 * bot_dim, bot_dim, 1))
        # After each upsample: concat with enc skip
        for i in range(self.n_stages - 2, -1, -1):
            self.dec_skip_proj.append(
                nn.Conv3d(2 * self.stage_dims[i], self.stage_dims[i], 1))

        # ── Decoder stages (URA-SSM) ─────────────────────────────────
        self.dec_stages = nn.ModuleList()
        # First decoder stage at bottleneck resolution
        self.dec_stages.append(
            DecoderStage(bot_dim, stage_depths[-1],
                         d_state, d_conv, expand))
        # Remaining decoder stages (mirrored)
        for i in range(self.n_stages - 2, -1, -1):
            self.dec_stages.append(
                DecoderStage(self.stage_dims[i], stage_depths[i],
                             d_state, d_conv, expand))

        # ── Output heads ──────────────────────────────────────────────
        self.out_norm = nn.Sequential(
            nn.GroupNorm(min(32, d_h), d_h), nn.SiLU(inplace=True))
        # Eq. 10: noise prediction
        self.eps_head = nn.Conv3d(d_h, latent_dim, 1)
        # Eq. 10: variance prediction — Conv3d at FULL decoder resolution
        self.var_head = nn.Conv3d(d_h, 1, 1)

        # ── Track total block count for axis cycling ──────────────────
        enc_blocks = sum(stage_depths)
        bot_blocks = bottleneck_depth
        dec_blocks = sum(stage_depths)  # mirror
        self._enc_blocks = enc_blocks
        self._bot_start = enc_blocks
        self._dec_start = enc_blocks + bot_blocks

    # ── v_tilde at multiple resolutions ───────────────────────────────

    def _downsample_vtilde(
        self, v_tilde: torch.Tensor,
        full_shape: Tuple[int, int, int],
        target_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Downsample v_tilde [B, N_full] to match target spatial resolution."""
        if full_shape == target_shape:
            return v_tilde
        B = v_tilde.shape[0]
        h, w, d = full_shape
        v_vol = v_tilde.reshape(B, 1, h, w, d)
        th, tw, td = target_shape
        ks = (h // th, w // tw, d // td)
        v_down = F.avg_pool3d(v_vol, kernel_size=ks, stride=ks)
        return v_down.reshape(B, -1)                           # [B, N_target]

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_obs: torch.Tensor,
        v_tilde_ext: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full denoiser forward — Eq. (10).

        Parameters
        ----------
        z_t          : [B, c, h, w, d]  noisy latent at timestep t
        t            : [B]              integer timesteps
        z_obs        : [B, c, h, w, d]  observed latent conditioning
        v_tilde_ext  : [B, N] or None   clipped log-var from previous step
                       None → zeros (training default)

        Returns
        -------
        eps_theta : [B, c, h, w, d]  predicted noise
        v_theta   : [B, N]           predicted log-variance (raw, unclipped)
        """
        B, c, h, w, d = z_t.shape
        N = h * w * d
        device = z_t.device
        full_shape = (h, w, d)

        # Default v_tilde: zeros → gate ≈ 1
        if v_tilde_ext is None:
            v_tilde_ext = torch.zeros(B, N, device=device, dtype=z_t.dtype)

        # ── Input: cat z_t and z_obs ──────────────────────────────────
        x = torch.cat([z_t, z_obs], dim=1)                    # [B, 2c, h, w, d]
        x = self.input_conv(x)                                 # [B, d_h, h, w, d]

        # ── Multi-scale conditioning features ─────────────────────────
        cond_feats = self.cond_enc(z_obs)                      # [f0, f1, f2]

        # ── Timestep embeddings per stage ─────────────────────────────
        t_embs = self.time_embed(t)                            # [t0, t1, t2]

        # ── Encoder (plain SSM, no URA) ───────────────────────────────
        skips: List[torch.Tensor] = []
        block_idx = 0

        for i in range(self.n_stages):
            n_blocks = len(self.enc_stages[i].blocks)
            if self.use_checkpoint and x.requires_grad:
                x = ckpt_fn(
                    self.enc_stages[i],
                    x, t_embs[i], cond_feats[i], block_idx,
                    use_reentrant=False,
                )
            else:
                x = self.enc_stages[i](x, t_embs[i], cond_feats[i], block_idx)
            block_idx += n_blocks
            skips.append(x)

            if i < self.n_stages - 1:
                x = self.enc_downs[i](x)                       # downsample

        # ── Bottleneck (URA-SSM at lowest resolution) ─────────────────
        bot_shape = x.shape[2:]                                # (h/4, w/4, d/4)
        v_tilde_bot = self._downsample_vtilde(
            v_tilde_ext, full_shape, bot_shape)

        n_bot = len(self.bottleneck.blocks)
        if self.use_checkpoint and x.requires_grad:
            x = ckpt_fn(
                self.bottleneck,
                x, v_tilde_bot, t_embs[-1], cond_feats[-1], block_idx,
                use_reentrant=False,
            )
        else:
            x = self.bottleneck(x, v_tilde_bot, t_embs[-1],
                                cond_feats[-1], block_idx)
        block_idx += n_bot

        # ── Decoder (URA-SSM with skip connections) ───────────────────
        skip_proj_idx = 0

        # First decoder stage at bottleneck resolution + skip
        enc_skip = skips.pop()                                 # enc stage[-1]
        x = torch.cat([x, enc_skip], dim=1)                   # [B, 2·dim, ...]
        x = self.dec_skip_proj[skip_proj_idx](x)               # [B, dim, ...]
        skip_proj_idx += 1

        cur_shape = x.shape[2:]
        v_tilde_cur = self._downsample_vtilde(
            v_tilde_ext, full_shape, cur_shape)

        n_dec0 = len(self.dec_stages[0].blocks)
        if self.use_checkpoint and x.requires_grad:
            x = ckpt_fn(
                self.dec_stages[0],
                x, v_tilde_cur, t_embs[-1], cond_feats[-1], block_idx,
                use_reentrant=False,
            )
        else:
            x = self.dec_stages[0](x, v_tilde_cur, t_embs[-1],
                                   cond_feats[-1], block_idx)
        block_idx += n_dec0

        # Remaining decoder stages
        for dec_i in range(1, self.n_stages):
            # Reverse stage index: dec_i=1→enc_stage[-2], dec_i=2→enc_stage[-3]
            enc_stage_idx = self.n_stages - 1 - dec_i          # 1, 0

            # Upsample
            x = self.dec_ups[dec_i - 1](x)

            # Skip connection
            enc_skip = skips.pop()
            x = torch.cat([x, enc_skip], dim=1)
            x = self.dec_skip_proj[skip_proj_idx](x)
            skip_proj_idx += 1

            # v_tilde at this resolution
            cur_shape = x.shape[2:]
            v_tilde_cur = self._downsample_vtilde(
                v_tilde_ext, full_shape, cur_shape)

            n_blk = len(self.dec_stages[dec_i].blocks)
            if self.use_checkpoint and x.requires_grad:
                x = ckpt_fn(
                    self.dec_stages[dec_i],
                    x, v_tilde_cur, t_embs[enc_stage_idx],
                    cond_feats[enc_stage_idx], block_idx,
                    use_reentrant=False,
                )
            else:
                x = self.dec_stages[dec_i](
                    x, v_tilde_cur, t_embs[enc_stage_idx],
                    cond_feats[enc_stage_idx], block_idx)
            block_idx += n_blk

        # ── Output heads ──────────────────────────────────────────────
        x = self.out_norm(x)                                   # [B, d_h, h, w, d]
        eps_theta = self.eps_head(x)                           # [B, c, h, w, d]
        v_theta = self.var_head(x)                             # [B, 1, h, w, d]
        v_theta = v_theta.reshape(B, -1)                       # [B, N]

        return eps_theta, v_theta


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Shape, VRAM, and gradient tests for URSSMDenoiser."""

    print("=" * 70)
    print("  URSSMDenoiser — Test Suite (Eq. 10)")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ⚠ Skipping — requires CUDA for Mamba SSM blocks")
        return

    torch.manual_seed(42)

    for r, d_h, use_ckpt in [(8, 128, True), (4, 128, True)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        h = w = d = 128 // r
        N = h * w * d
        c = 4
        B = 1

        print(f"\n{'─' * 60}")
        print(f"  r={r}  latent=({h},{w},{d})  N={N}  d_h={d_h}  ckpt={use_ckpt}")
        print(f"{'─' * 60}")

        model = URSSMDenoiser(
            latent_dim=c, d_h=d_h,
            stage_depths=(2, 2, 4), bottleneck_depth=4,
            use_checkpoint=use_ckpt,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # ── Shape test (no grad, bf16) ────────────────────────────────
        print("\n  --- Shape test (no_grad + bf16) ---")
        z_t   = torch.randn(B, c, h, w, d, device=device)
        t     = torch.randint(0, 1000, (B,), device=device)
        z_obs = torch.randn(B, c, h, w, d, device=device)

        with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
            eps, v = model(z_t, t, z_obs)

        assert eps.shape == (B, c, h, w, d), f"eps shape: {eps.shape}"
        assert v.shape == (B, N), f"v shape: {v.shape}"
        print(f"  eps: {list(eps.shape)}  ✓")
        print(f"  v:   {list(v.shape)}  ✓")
        del eps, v

        # ── Shape test with external v_tilde ──────────────────────────
        print("\n  --- With v_tilde_ext ---")
        v_ext = torch.randn(B, N, device=device)
        with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16):
            eps2, v2 = model(z_t, t, z_obs, v_tilde_ext=v_ext)
        assert eps2.shape == (B, c, h, w, d) and v2.shape == (B, N)
        print(f"  With v_tilde_ext: shapes OK  ✓")
        del eps2, v2, v_ext

        # ── Gradient test ─────────────────────────────────────────────
        print("\n  --- Gradient flow ---")
        torch.cuda.empty_cache()
        z_t_g = torch.randn(B, c, h, w, d, device=device, requires_grad=True)

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            eps_g, v_g = model(z_t_g, t, z_obs)
            loss = eps_g.sum() + v_g.sum()
        loss.backward()

        assert z_t_g.grad is not None, "No gradient!"
        print(f"  z_t grad norm: {z_t_g.grad.norm().item():.4f}  ✓")
        del z_t_g, eps_g, v_g, loss
        torch.cuda.empty_cache()

        # ── VRAM measurement ──────────────────────────────────────────
        print("\n  --- VRAM measurement (training-like) ---")
        torch.cuda.reset_peak_memory_stats()

        z_t_v = torch.randn(B, c, h, w, d, device=device, requires_grad=True)
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            eps_v, v_v = model(z_t_v, t, z_obs)
            loss_v = eps_v.mean() + v_v.mean()
        loss_v.backward()

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"  Peak VRAM: {peak_gb:.2f} GB")
        if r == 8:
            assert peak_gb < 35, f"r=8 VRAM {peak_gb:.1f} GB exceeds 35 GB!"
            print(f"  ✓ Under 35 GB limit for r=8")
        else:
            print(f"  (r=4 — no strict limit, but should fit in 48 GB)")

        del z_t_v, eps_v, v_v, loss_v, z_t, z_obs
        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("  ALL TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
