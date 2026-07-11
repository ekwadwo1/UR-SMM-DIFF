#!/usr/bin/env python3
"""
experiments/new_baselines.py — Additional Baselines for UR-SSM-Diff
====================================================================

Three additional baselines to strengthen the comparative evaluation:

  6. SegMamba        — SSM-based discriminative segmenter
  7. Restore→nnU-Net — Sequential two-stage pipeline
  8. TransBTS        — Hybrid CNN-Transformer discriminative segmenter

All baselines:
  - Same 5-fold CV splits, same PhysicsCorruptionOperator
  - Segmentation metrics: DSC (ET/TC/WT), HD95 (mm), Surface Dice τ=1mm
  - Restoration metrics: PSNR (dB), SSIM (brain-masked 3D)
  - DDP or single-GPU, bfloat16 AMP, AdamW, warmup+cosine
  - 100 epochs, Dice+CE loss

Usage (single GPU — when GPU 0 is busy):
  CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
      experiments/new_baselines.py --baseline transbts --fold 0

Usage (2 GPUs):
  torchrun --nproc_per_node=2 experiments/new_baselines.py \
      --baseline segmamba --fold 0

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# ── Ensure project root is on sys.path ───────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Project imports ───────────────────────────────────────────────────
from data.brats_dataset import BraTSDataset
from losses.segmentation_loss import SegmentationLoss, map_brats_labels
from physics.corruption_operator import PhysicsCorruptionOperator

logger = logging.getLogger("new_baselines")


# ######################################################################
#  Metrics: DSC, HD95, Surface Dice, PSNR, SSIM                        #
# ######################################################################

def _binary_to_surface_points(
    mask: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Extract surface voxel coordinates from a binary 3D mask."""
    from scipy import ndimage
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    eroded = ndimage.binary_erosion(mask, iterations=1)
    border = mask & ~eroded
    coords = np.argwhere(border).astype(np.float64)
    if coords.size == 0:
        return coords
    coords *= np.array(spacing)
    return coords


def compute_hd95(
    pred: np.ndarray, target: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """95th-percentile Hausdorff Distance in mm."""
    from scipy.spatial.distance import cdist

    pred_b = pred.astype(bool)
    tgt_b = target.astype(bool)
    p_sum, t_sum = pred_b.sum(), tgt_b.sum()
    if p_sum == 0 and t_sum == 0:
        return 0.0
    if p_sum == 0 or t_sum == 0:
        return float("nan")

    sp = _binary_to_surface_points(pred_b, spacing)
    st = _binary_to_surface_points(tgt_b, spacing)
    if sp.shape[0] == 0 or st.shape[0] == 0:
        return float("nan")

    max_pts = 10000
    if sp.shape[0] > max_pts:
        sp = sp[np.random.choice(sp.shape[0], max_pts, replace=False)]
    if st.shape[0] > max_pts:
        st = st[np.random.choice(st.shape[0], max_pts, replace=False)]

    d_p2t = cdist(sp, st).min(axis=1)
    d_t2p = cdist(st, sp).min(axis=1)
    return float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))


def compute_surface_dice(
    pred: np.ndarray, target: np.ndarray,
    tau: float = 1.0,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """Surface Dice at tolerance τ (mm). Nikolov et al. 2021."""
    from scipy.spatial.distance import cdist

    pred_b, tgt_b = pred.astype(bool), target.astype(bool)
    p_sum, t_sum = pred_b.sum(), tgt_b.sum()
    if p_sum == 0 and t_sum == 0:
        return 1.0
    if p_sum == 0 or t_sum == 0:
        return 0.0

    sp = _binary_to_surface_points(pred_b, spacing)
    st = _binary_to_surface_points(tgt_b, spacing)
    if sp.shape[0] == 0 or st.shape[0] == 0:
        return 0.0

    max_pts = 10000
    sp_full, st_full = sp.shape[0], st.shape[0]
    if sp.shape[0] > max_pts:
        sp = sp[np.random.choice(sp.shape[0], max_pts, replace=False)]
    if st.shape[0] > max_pts:
        st = st[np.random.choice(st.shape[0], max_pts, replace=False)]

    d_p2t = cdist(sp, st).min(axis=1)
    d_t2p = cdist(st, sp).min(axis=1)

    n_p_close = (d_p2t <= tau).sum() * (sp_full / sp.shape[0])
    n_t_close = (d_t2p <= tau).sum() * (st_full / st.shape[0])
    return float((n_p_close + n_t_close) / (sp_full + st_full))


def compute_psnr(
    pred: np.ndarray, target: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
) -> float:
    """PSNR (dB) within brain mask in 3D."""
    if brain_mask is None:
        if target.ndim == 4:
            brain_mask = np.any(np.abs(target) > 1e-6, axis=0)
        else:
            brain_mask = np.abs(target) > 1e-6

    if pred.ndim == 4 and brain_mask.ndim == 3:
        mask_4d = np.broadcast_to(brain_mask[None], pred.shape)
    else:
        mask_4d = brain_mask

    p_m = pred[mask_4d]
    t_m = target[mask_4d]
    if t_m.size == 0:
        return 0.0

    data_range = float(t_m.max() - t_m.min())
    if data_range < 1e-8:
        return 0.0
    mse = float(np.mean((p_m - t_m) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_ssim_3d(
    pred: np.ndarray, target: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
) -> float:
    """SSIM within brain mask in 3D (Wang et al. 2004)."""
    from scipy.ndimage import gaussian_filter

    if brain_mask is None:
        if target.ndim == 4:
            brain_mask = np.any(np.abs(target) > 1e-6, axis=0)
        else:
            brain_mask = np.abs(target) > 1e-6

    sigma = 1.5

    def _ssim_ch(p, t, mask):
        if mask.sum() == 0:
            return 0.0
        dr = float(t[mask].max() - t[mask].min())
        if dr < 1e-8:
            dr = 1.0
        c1 = (0.01 * dr) ** 2
        c2 = (0.03 * dr) ** 2
        mu_p = gaussian_filter(p, sigma)
        mu_t = gaussian_filter(t, sigma)
        s_pp = gaussian_filter(p * p, sigma) - mu_p * mu_p
        s_tt = gaussian_filter(t * t, sigma) - mu_t * mu_t
        s_pt = gaussian_filter(p * t, sigma) - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + c1) * (2 * s_pt + c2)) / \
                   ((mu_p ** 2 + mu_t ** 2 + c1) * (s_pp + s_tt + c2))
        return float(ssim_map[mask].mean())

    if pred.ndim == 4:
        return float(np.mean([
            _ssim_ch(pred[c], target[c], brain_mask)
            for c in range(pred.shape[0])]))
    return _ssim_ch(pred, target, brain_mask)


def compute_all_region_metrics(
    pred_labels: np.ndarray,
    target_labels: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """DSC, HD95, Surface Dice for WT, TC, ET."""
    regions = {
        "wt": lambda x: (x >= 1),
        "tc": lambda x: (x == 1) | (x == 3),
        "et": lambda x: (x == 3),
    }
    results = {}
    for name, fn in regions.items():
        p_bin = fn(pred_labels).astype(bool)
        t_bin = fn(target_labels).astype(bool)

        inter = (p_bin & t_bin).sum()
        union = p_bin.sum() + t_bin.sum()
        results[f"dsc_{name}"] = float(
            1.0 if union == 0 else 2.0 * inter / union)
        results[f"hd95_{name}"] = compute_hd95(p_bin, t_bin, spacing)
        results[f"sd_{name}"] = compute_surface_dice(
            p_bin, t_bin, tau=1.0, spacing=spacing)
    return results


# ######################################################################
#  Paths and Config                                                     #
# ######################################################################

DRIVE_ROOT = (
    "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/"
    "NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS"
)
DEFAULTS = {
    "data_root": os.path.join(
        DRIVE_ROOT, "UR_SSM_Diff_Outputs/preprocessed/BraTS2021"),
    "vqgan_ckpt": os.path.join(
        DRIVE_ROOT, "UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"),
    "training_dir": os.path.join(
        DRIVE_ROOT, "UR_SSM_Diff_Outputs/training"),
    "output_dir": os.path.join(
        DRIVE_ROOT, "UR_SSM_Diff_Outputs/baselines"),
}


@dataclass
class BaselineConfig:
    data_root: str = DEFAULTS["data_root"]
    output_dir: str = DEFAULTS["output_dir"]
    vqgan_ckpt: str = DEFAULTS["vqgan_ckpt"]
    training_dir: str = DEFAULTS["training_dir"]
    epochs: int = 100
    lr: float = 2e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    micro_batch: int = 1
    grad_accum: int = 4
    warmup_steps: int = 5000
    n_classes: int = 4
    in_channels: int = 4
    num_workers: int = 4
    val_every_epochs: int = 5
    log_every: int = 50


# ######################################################################
#  DDP Utilities (graceful single-GPU support)                          #
# ######################################################################

def setup_ddp() -> Tuple[int, int, torch.device]:
    """Initialize DDP. Handles single-GPU (CUDA_VISIBLE_DEVICES=1 +
    --nproc_per_node=1) gracefully."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    n_visible = torch.cuda.device_count()
    if local_rank >= n_visible:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {n_visible} GPU(s) visible. "
            f"With CUDA_VISIBLE_DEVICES selecting a single GPU, "
            f"use --nproc_per_node=1 (not 2).")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ######################################################################
#  LR Scheduler                                                         #
# ######################################################################

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0
        self.current_lr = self.base_lrs[0]

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            frac = self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            frac = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, blr * frac)
        self.current_lr = self.optimizer.param_groups[0]["lr"]


# ######################################################################
#  Model 6: SegMamba                                                    #
# ######################################################################

class MambaEncoderBlock(nn.Module):
    """Mamba block with tri-orientated scanning (Xing et al., MICCAI 2024)."""

    def __init__(self, dim, axis=0, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.axis = axis
        self.dim = dim
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.SiLU(inplace=True)
        self.dwconv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.dwconv_norm = nn.GroupNorm(min(8, dim), dim)

        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.has_mamba = True
        except ImportError:
            logger.warning("mamba_ssm not found; using conv1d fallback")
            inner = dim * expand
            self.proj_in = nn.Linear(dim, inner * 2)
            self.conv1d = nn.Conv1d(
                inner, inner, d_conv, padding=d_conv - 1, groups=inner)
            self.proj_out = nn.Linear(inner, dim)
            self.has_mamba = False

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        residual = x
        h = self.act(self.norm(x))
        local_out = self.dwconv_norm(self.act(self.dwconv(h)))

        if self.axis == 0:
            h_perm = h.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        elif self.axis == 1:
            h_perm = h.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, C)
        else:
            h_perm = h.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, C)

        if self.has_mamba:
            h_scan = self.mamba(h_perm)
        else:
            g = self.proj_in(h_perm)
            gate, val = g.chunk(2, dim=-1)
            val = self.conv1d(val.transpose(1, 2))[:, :, :h_perm.shape[1]]
            h_scan = self.proj_out(F.silu(gate) * val.transpose(1, 2))

        h_scan = self.proj(h_scan)

        if self.axis == 0:
            h_scan = h_scan.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)
        elif self.axis == 1:
            h_scan = h_scan.reshape(B, D, W, H, C).permute(0, 4, 1, 3, 2)
        else:
            h_scan = h_scan.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        return residual + local_out + h_scan


class ConvBlock3D(nn.Module):
    """Residual 3×3×3 conv block for high-resolution stages."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.GroupNorm(min(8, dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.GroupNorm(min(8, dim), dim),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return x + self.block(x)


class SegMamba3D(nn.Module):
    """SegMamba-style 3D segmentation (memory-efficient variant).

    Conv at high-res (128³, 64³), Mamba at low-res (32³, 16³, 8³).
    Gradient checkpointing on Mamba stages. ~16M params with base_dim=64.
    """

    def __init__(self, in_channels=4, n_classes=4, base_dim=64,
                 n_stages=4, d_state=16, mamba_start_stage=2):
        super().__init__()
        self.n_stages = n_stages
        self.mamba_start = mamba_start_stage
        dims = [base_dim * (2 ** i) for i in range(n_stages)]

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], 3, padding=1),
            nn.GroupNorm(8, dims[0]), nn.SiLU(inplace=True),
            nn.Conv3d(dims[0], dims[0], 3, padding=1),
            nn.GroupNorm(8, dims[0]), nn.SiLU(inplace=True))

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(n_stages):
            if i < mamba_start_stage:
                blk = nn.Sequential(ConvBlock3D(dims[i]), ConvBlock3D(dims[i]))
            else:
                blk = nn.Sequential(
                    MambaEncoderBlock(dims[i], axis=(2*i) % 3, d_state=d_state),
                    MambaEncoderBlock(dims[i], axis=(2*i+1) % 3, d_state=d_state))
            self.enc_blocks.append(blk)
            if i < n_stages - 1:
                self.downsamples.append(
                    nn.Conv3d(dims[i], dims[i+1], 2, stride=2))

        self.bottleneck = nn.Sequential(
            MambaEncoderBlock(dims[-1], axis=0, d_state=d_state),
            MambaEncoderBlock(dims[-1], axis=1, d_state=d_state))

        self.upsamples = nn.ModuleList()
        self.dec_projs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(n_stages - 2, -1, -1):
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear",
                             align_corners=False),
                nn.Conv3d(dims[i+1], dims[i], 1)))
            self.dec_projs.append(nn.Conv3d(dims[i]*2, dims[i], 1))
            if i < mamba_start_stage:
                blk = nn.Sequential(ConvBlock3D(dims[i]), ConvBlock3D(dims[i]))
            else:
                blk = nn.Sequential(
                    MambaEncoderBlock(dims[i], axis=(2*i) % 3, d_state=d_state),
                    MambaEncoderBlock(dims[i], axis=(2*i+1) % 3, d_state=d_state))
            self.dec_blocks.append(blk)

        self.head = nn.Conv3d(dims[0], n_classes, 1)
        self._use_ckpt = True

    def _ckpt(self, fn, *args):
        if self._use_ckpt and self.training:
            return torch.utils.checkpoint.checkpoint(
                fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for i in range(self.n_stages):
            h = self._ckpt(self.enc_blocks[i], h) if i >= self.mamba_start \
                else self.enc_blocks[i](h)
            skips.append(h)
            if i < self.n_stages - 1:
                h = self.downsamples[i](h)

        h = self._ckpt(self.bottleneck, h)

        dec_idx = 0
        for i in range(self.n_stages - 2, -1, -1):
            h = self.upsamples[dec_idx](h)
            h = self.dec_projs[dec_idx](torch.cat([h, skips[i]], dim=1))
            h = self._ckpt(self.dec_blocks[dec_idx], h) if i >= self.mamba_start \
                else self.dec_blocks[dec_idx](h)
            dec_idx += 1
        return self.head(h)


# ######################################################################
#  Model 8: TransBTS                                                    #
# ######################################################################

class TransBTS3D(nn.Module):
    """TransBTS: CNN encoder → Transformer bottleneck → CNN decoder.
    Wang et al., MICCAI 2021."""

    def __init__(self, in_channels=4, n_classes=4, base_dim=32,
                 n_heads=8, n_transformer_layers=4):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        self.enc1 = self._cb(in_channels, dims[0])
        self.enc2 = self._cb(dims[0], dims[1])
        self.enc3 = self._cb(dims[1], dims[2])
        self.enc4 = self._cb(dims[2], dims[3])
        self.pool = nn.MaxPool3d(2, stride=2)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dims[3], nhead=n_heads, dim_feedforward=dims[3]*4,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_transformer_layers)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 8*8*8, dims[3]) * 0.02)

        self.up4 = self._ub(dims[3], dims[2])
        self.dec4 = self._cb(dims[2]+dims[3], dims[2])
        self.up3 = self._ub(dims[2], dims[1])
        self.dec3 = self._cb(dims[1]+dims[2], dims[1])
        self.up2 = self._ub(dims[1], dims[0])
        self.dec2 = self._cb(dims[0]+dims[1], dims[0])
        self.up1 = self._ub(dims[0], dims[0])
        self.dec1 = self._cb(dims[0]+in_channels, dims[0])
        self.head = nn.Conv3d(dims[0], n_classes, 1)

    @staticmethod
    def _cb(ic, oc):
        return nn.Sequential(
            nn.Conv3d(ic, oc, 3, padding=1),
            nn.GroupNorm(min(8, oc), oc), nn.SiLU(inplace=True),
            nn.Conv3d(oc, oc, 3, padding=1),
            nn.GroupNorm(min(8, oc), oc), nn.SiLU(inplace=True))

    @staticmethod
    def _ub(ic, oc):
        return nn.Sequential(
            nn.ConvTranspose3d(ic, oc, 2, stride=2),
            nn.GroupNorm(min(8, oc), oc), nn.SiLU(inplace=True))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.pool(e4)
        B, C, D, H, W = b.shape
        tok = b.flatten(2).transpose(1, 2)
        tok = tok + self.pos_embed[:, :tok.shape[1], :]
        tok = self.transformer(tok)
        b = tok.transpose(1, 2).reshape(B, C, D, H, W)

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x], 1))
        return self.head(d1)


# ######################################################################
#  Discriminative Training Loop                                         #
# ######################################################################

def train_discriminative(
    baseline_name: str,
    model: nn.Module,
    cfg: BaselineConfig,
    fold: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> str:
    """Train a discriminative segmentation baseline on corrupted inputs.
    Validates with DSC/HD95/SD + PSNR/SSIM. Returns best checkpoint path."""

    tag = f"{baseline_name}_fold{fold}"
    out_dir = os.path.join(cfg.output_dir, baseline_name, f"fold_{fold}")
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"=== Train {baseline_name} | Fold {fold} ===")

    amp_dt = torch.bfloat16

    # ── Data ──────────────────────────────────────────────────────────
    fold_json = os.path.join(cfg.data_root, f"fold_{fold}.json")
    if not os.path.exists(fold_json):
        raise FileNotFoundError(
            f"Fold JSON not found: {fold_json}\n"
            f"  --data-root must point to preprocessed BraTS2021 dir "
            f"containing fold_0.json … fold_4.json")

    train_ds = BraTSDataset(
        cfg.data_root, augment=True, fold_json=fold_json, split="train")
    val_ds = BraTSDataset(
        cfg.data_root, augment=False, fold_json=fold_json, split="val")

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.micro_batch, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(
        val_ds, batch_size=1, sampler=val_sampler,
        num_workers=2, pin_memory=True)

    # ── Model / optim ────────────────────────────────────────────────
    model = model.to(device)
    model = DDP(model, device_ids=[device.index],
                find_unused_parameters=True)
    corruption = PhysicsCorruptionOperator(device=str(device))
    seg_loss_fn = SegmentationLoss(n_classes=cfg.n_classes).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_dl) // cfg.grad_accum)
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.warmup_steps, total_steps, cfg.min_lr)

    if is_main():
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Params: {n:,}  Train: {len(train_ds)}  "
                    f"Val: {len(val_ds)}")

    best_dsc = -1.0
    best_ckpt = os.path.join(out_dir, "best_model.pt")
    metrics_log = []
    gstep = 0

    for epoch in range(cfg.epochs):
        train_dl.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            x_0 = batch["image"].to(device, non_blocking=True)
            s_0 = batch["label"].to(device, non_blocking=True)

            with autocast("cuda", dtype=amp_dt):
                with torch.no_grad():
                    y = corruption(x_0)
                s_0_mapped = map_brats_labels(s_0)
                logits = model(y)
                if logits.dim() == 6:
                    logits = logits[:, 0]
                l_seg, seg_logs = seg_loss_fn(logits, s_0_mapped)
                loss = l_seg / cfg.grad_accum

            loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                gstep += 1

                if is_main() and gstep % cfg.log_every == 0:
                    logger.info(
                        f"  [{tag}] ep={epoch} s={gstep} "
                        f"l_seg={seg_logs['l_seg']:.4f} "
                        f"lr={scheduler.current_lr:.2e}")

        # ── Validation ────────────────────────────────────────────────
        if (epoch + 1) % cfg.val_every_epochs == 0:
            model.eval()
            metric_keys = [
                "dsc_wt", "dsc_tc", "dsc_et",
                "hd95_wt", "hd95_tc", "hd95_et",
                "sd_wt", "sd_tc", "sd_et",
                "psnr", "ssim"]
            accum: Dict[str, List[float]] = {k: [] for k in metric_keys}

            with torch.no_grad():
                for vb in val_dl:
                    xv = vb["image"].to(device)
                    sv = vb["label"].to(device)
                    with autocast("cuda", dtype=amp_dt):
                        yv = corruption(xv)
                        logits_v = model(yv)
                    if logits_v.dim() == 6:
                        logits_v = logits_v[:, 0]

                    pred_np = logits_v.argmax(1)[0].cpu().numpy()
                    tgt_np = map_brats_labels(sv)[0].cpu().numpy()

                    seg_m = compute_all_region_metrics(pred_np, tgt_np)
                    for k, v in seg_m.items():
                        if not np.isnan(v):
                            accum[k].append(v)

                    # PSNR/SSIM: corrupted vs clean (corruption baseline)
                    accum["psnr"].append(compute_psnr(
                        yv[0].cpu().float().numpy(), xv[0].cpu().numpy()))
                    accum["ssim"].append(compute_ssim_3d(
                        yv[0].cpu().float().numpy(), xv[0].cpu().numpy()))

            avg = {}
            for k, vals in accum.items():
                m = float(np.mean(vals)) if vals else 0.0
                t = torch.tensor(m, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                avg[k] = t.item()

            if is_main():
                logger.info(
                    f"  [{tag} Val] ep={epoch+1} "
                    f"DSC_WT={avg['dsc_wt']:.4f} "
                    f"DSC_TC={avg['dsc_tc']:.4f} "
                    f"DSC_ET={avg['dsc_et']:.4f} "
                    f"HD95_WT={avg['hd95_wt']:.2f} "
                    f"SD_WT={avg['sd_wt']:.4f}")
                metrics_log.append({"epoch": epoch + 1, **avg})

                if avg["dsc_wt"] > best_dsc:
                    best_dsc = avg["dsc_wt"]
                    torch.save({
                        "epoch": epoch + 1,
                        "model": model.module.state_dict(),
                        "best_dsc": best_dsc,
                        "metrics": avg,
                    }, best_ckpt)
                    logger.info(f"  ✓ Best DSC_WT={best_dsc:.4f} saved")

    if is_main():
        log_path = os.path.join(out_dir, "metrics_log.json")
        with open(log_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        logger.info(f"  {baseline_name} done. Best DSC_WT={best_dsc:.4f}")
    return best_ckpt


# ######################################################################
#  Baseline 7: Restore→nnU-Net                                         #
# ######################################################################

@torch.no_grad()
def restore_volumes(cfg, fold, device):
    """Step 1: Restore volumes using UR-SSM-Diff (γ=0)."""
    from models.ur_ssm_diff import build_ur_ssm_diff

    out_dir = os.path.join(
        cfg.output_dir, "restore_nnunet", f"fold_{fold}", "restored")
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"=== Restore volumes | Fold {fold} ===")

    model_ckpt = os.path.join(
        cfg.training_dir, f"fold_{fold}", "phase3", "best_model.pt")
    if not os.path.exists(model_ckpt):
        model_ckpt = os.path.join(
            cfg.training_dir, f"fold_{fold}", "phase2", "best_model.pt")

    model = build_ur_ssm_diff(
        vqgan_ckpt=cfg.vqgan_ckpt, latent_dim=4, d_h=128,
        downsample_factor=4, n_classes=4, T=1000, device=str(device))
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=True)
    if "denoiser" in ckpt:
        model.denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    model.eval()

    fold_json = os.path.join(cfg.data_root, f"fold_{fold}.json")
    for split in ["train", "val"]:
        ds = BraTSDataset(cfg.data_root, augment=False,
                          fold_json=fold_json, split=split)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        if is_main():
            logger.info(f"  Restoring {split}: {len(ds)} subjects")

        for idx, batch in enumerate(loader):
            sid = batch["id"][0]
            x_0 = batch["image"].to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                y = model.corruption_op(x_0)
                x_phys = model.corruption_op(x_0)
                z_obs = model.vqgan(y, mode="encode")
                z_0 = model.vqgan(x_0, mode="encode")
                z_0_phys = model.vqgan(x_phys, mode="encode")
                t = torch.tensor([250], device=device)
                z_t, _ = model.diffusion.q_sample(z_0, z_0_phys, t)
                eps_theta, _ = model.denoiser(z_t, t, z_obs)
                z_0_pred = model.diffusion.tweedie_x0_estimate(
                    z_t, eps_theta, t, z_obs)
                x_restored = model.vqgan(z_0_pred.float(), mode="decode")

            np.save(os.path.join(out_dir, f"{sid}.npy"),
                    x_restored[0].cpu().float().numpy())
            if is_main() and (idx + 1) % 50 == 0:
                logger.info(f"    [{idx+1}/{len(ds)}] {sid}")

    del model
    torch.cuda.empty_cache()
    if is_main():
        logger.info(f"  Restored volumes → {out_dir}")
    return out_dir


class RestoredBraTSDataset(Dataset):
    """Loads restored volumes + original labels."""

    def __init__(self, original_root, restored_dir, fold_json,
                 split="train", augment=True):
        # Wrap BraTSDataset which already knows the correct file layout
        self.base_ds = BraTSDataset(
            original_root, augment=augment,
            fold_json=fold_json, split=split)
        self.restored_dir = restored_dir

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        # Load using BraTSDataset (correct label paths)
        sample = self.base_ds[idx]
        sid = sample["id"]

        # Swap image with restored version if available
        rpath = os.path.join(self.restored_dir, f"{sid}.npy")
        if os.path.exists(rpath):
            restored = np.load(rpath).astype(np.float32)
            sample["image"] = torch.from_numpy(restored)

        return sample


def train_restore_nnunet(cfg, fold, rank, world_size, device):
    """Step 2: Train nnU-Net on restored volumes."""
    from monai.networks.nets import DynUNet

    tag = f"restore_nnunet_fold{fold}"
    out_dir = os.path.join(cfg.output_dir, "restore_nnunet", f"fold_{fold}")
    if is_main():
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"=== Train Restore→nnU-Net | Fold {fold} ===")

    amp_dt = torch.bfloat16
    restored_dir = os.path.join(out_dir, "restored")
    fold_json = os.path.join(cfg.data_root, f"fold_{fold}.json")

    # Check if step 1 (restore) was run
    if not os.path.isdir(restored_dir) or not any(
            f.endswith(".npy") for f in os.listdir(restored_dir)
            if os.path.isfile(os.path.join(restored_dir, f))):
        logger.warning(
            f"  ⚠ Restored directory is empty or missing: {restored_dir}\n"
            f"    You must run --phase restore FIRST to generate restored volumes.\n"
            f"    Falling back to original (uncorrupted) volumes for training.")
    else:
        n_restored = len([f for f in os.listdir(restored_dir) if f.endswith(".npy")])
        if is_main():
            logger.info(f"  Found {n_restored} restored volumes in {restored_dir}")

    train_ds = RestoredBraTSDataset(
        cfg.data_root, restored_dir, fold_json, "train", True)
    val_ds = RestoredBraTSDataset(
        cfg.data_root, restored_dir, fold_json, "val", False)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.micro_batch,
                          sampler=train_sampler, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1, sampler=val_sampler,
                        num_workers=2, pin_memory=True)

    strides = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
    model = DynUNet(
        spatial_dims=3, in_channels=cfg.in_channels,
        out_channels=cfg.n_classes,
        kernel_size=[[3,3,3]]*5, strides=strides,
        upsample_kernel_size=strides[1:],
        deep_supervision=True, deep_supr_num=3).to(device)
    model = DDP(model, device_ids=[device.index],
                find_unused_parameters=True)
    seg_loss_fn = SegmentationLoss(n_classes=cfg.n_classes).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_dl) // cfg.grad_accum)
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.warmup_steps,
        cfg.epochs * steps_per_epoch, cfg.min_lr)

    if is_main():
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Params: {n:,}  Train: {len(train_ds)}  Val: {len(val_ds)}")

    best_dsc = -1.0
    best_ckpt = os.path.join(out_dir, "best_model.pt")
    metrics_log = []
    gstep = 0

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            xr = batch["image"].to(device, non_blocking=True)
            s0 = batch["label"].to(device, non_blocking=True)
            with autocast("cuda", dtype=amp_dt):
                s0m = map_brats_labels(s0)
                logits = model(xr)
                if logits.dim() == 6:
                    logits = logits[:, 0]
                l_seg, seg_logs = seg_loss_fn(logits, s0m)
                loss = l_seg / cfg.grad_accum
            loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                gstep += 1
                if is_main() and gstep % cfg.log_every == 0:
                    logger.info(
                        f"  [{tag}] ep={epoch} s={gstep} "
                        f"l_seg={seg_logs['l_seg']:.4f} "
                        f"lr={scheduler.current_lr:.2e}")

        if (epoch + 1) % cfg.val_every_epochs == 0:
            model.eval()
            metric_keys = [
                "dsc_wt","dsc_tc","dsc_et",
                "hd95_wt","hd95_tc","hd95_et",
                "sd_wt","sd_tc","sd_et","psnr","ssim"]
            accum = {k: [] for k in metric_keys}

            val_orig = BraTSDataset(
                cfg.data_root, augment=False,
                fold_json=fold_json, split="val")
            val_orig_dl = DataLoader(
                val_orig, batch_size=1, shuffle=False, num_workers=2)

            with torch.no_grad():
                for vb, vb_orig in zip(val_dl, val_orig_dl):
                    xv = vb["image"].to(device)
                    sv = vb["label"].to(device)
                    with autocast("cuda", dtype=amp_dt):
                        logits_v = model(xv)
                    if logits_v.dim() == 6:
                        logits_v = logits_v[:, 0]

                    pred_np = logits_v.argmax(1)[0].cpu().numpy()
                    tgt_np = map_brats_labels(sv)[0].cpu().numpy()
                    seg_m = compute_all_region_metrics(pred_np, tgt_np)
                    for k, v in seg_m.items():
                        if not np.isnan(v):
                            accum[k].append(v)

                    restored_np = xv[0].cpu().numpy()
                    clean_np = vb_orig["image"][0].numpy()
                    accum["psnr"].append(compute_psnr(restored_np, clean_np))
                    accum["ssim"].append(
                        compute_ssim_3d(restored_np, clean_np))

            avg = {}
            for k, vals in accum.items():
                m = float(np.mean(vals)) if vals else 0.0
                t = torch.tensor(m, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                avg[k] = t.item()

            if is_main():
                logger.info(
                    f"  [{tag} Val] ep={epoch+1} "
                    f"DSC_WT={avg['dsc_wt']:.4f} "
                    f"HD95_WT={avg['hd95_wt']:.2f} "
                    f"PSNR={avg['psnr']:.2f} SSIM={avg['ssim']:.4f}")
                metrics_log.append({"epoch": epoch+1, **avg})
                if avg["dsc_wt"] > best_dsc:
                    best_dsc = avg["dsc_wt"]
                    torch.save({
                        "epoch": epoch+1,
                        "model": model.module.state_dict(),
                        "best_dsc": best_dsc, "metrics": avg,
                    }, best_ckpt)
                    logger.info(f"  ✓ Best DSC_WT={best_dsc:.4f} saved")

    if is_main():
        with open(os.path.join(out_dir, "metrics_log.json"), "w") as f:
            json.dump(metrics_log, f, indent=2)
        logger.info(f"  restore_nnunet done. Best DSC_WT={best_dsc:.4f}")
    return best_ckpt


# ######################################################################
#  Builders                                                             #
# ######################################################################

def build_segmamba(cfg):
    return SegMamba3D(
        in_channels=cfg.in_channels, n_classes=cfg.n_classes,
        base_dim=64, n_stages=4, d_state=16, mamba_start_stage=2)

def build_transbts(cfg):
    return TransBTS3D(
        in_channels=cfg.in_channels, n_classes=cfg.n_classes,
        base_dim=32, n_heads=8, n_transformer_layers=4)


# ######################################################################
#  CLI                                                                  #
# ######################################################################

def main():
    parser = argparse.ArgumentParser("UR-SSM-Diff Additional Baselines")
    parser.add_argument("--baseline", required=True,
                        choices=["segmamba", "restore_nnunet", "transbts"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root", default=DEFAULTS["data_root"])
    parser.add_argument("--vqgan-ckpt", default=DEFAULTS["vqgan_ckpt"])
    parser.add_argument("--training-dir", default=DEFAULTS["training_dir"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--phase", default="all",
                        choices=["all", "restore", "train"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    rank, world_size, device = setup_ddp()
    if is_main():
        logger.info(f"DDP: rank={rank} world={world_size} device={device}")

    cfg = BaselineConfig(
        data_root=args.data_root, output_dir=args.output_dir,
        vqgan_ckpt=args.vqgan_ckpt, training_dir=args.training_dir,
        epochs=args.epochs, lr=args.lr)

    if args.baseline == "segmamba":
        model = build_segmamba(cfg)
        if is_main():
            logger.info(f"SegMamba: {sum(p.numel() for p in model.parameters()):,} params")
        train_discriminative("segmamba", model, cfg, args.fold,
                             rank, world_size, device)

    elif args.baseline == "restore_nnunet":
        if args.phase in ("all", "restore"):
            restore_volumes(cfg, args.fold, device)
            dist.barrier()
        if args.phase in ("all", "train"):
            train_restore_nnunet(cfg, args.fold, rank, world_size, device)

    elif args.baseline == "transbts":
        model = build_transbts(cfg)
        if is_main():
            logger.info(f"TransBTS: {sum(p.numel() for p in model.parameters()):,} params")
        train_discriminative("transbts", model, cfg, args.fold,
                             rank, world_size, device)

    cleanup_ddp()
    if is_main():
        logger.info("Done.")


if __name__ == "__main__":
    main()
