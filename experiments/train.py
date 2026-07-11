#!/usr/bin/env python3
"""
train.py — UR-SSM-Diff DDP Training Pipeline (Phase 9-B)
=========================================================

Launch:
    torchrun --nproc_per_node=2 train.py --config config.yaml
    torchrun --nproc_per_node=2 train.py --phase 2 --fold 0
    torchrun --nproc_per_node=2 train.py --phase 3 --fold 0

Phases:
  1 — VQGAN pre-training  (use train_vqgan.py, already complete)
  2 — Denoiser pre-train  (L_diff only, 100 epochs/fold)
  3 — Fine-tune Regime A  (L_diff + γ·L_seg, 50 epochs/fold)

5-fold CV: VQGAN shared across folds; denoiser + seg head per fold.

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8 · NCCL backend
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from data.brats_dataset import build_brats_dataloaders
from diffusion.physics_forward import PhysicsConsistentDiffusion
from losses.diffusion_loss import HeteroscedasticDiffusionLoss
from losses.segmentation_loss import (
    GammaScheduler,
    SegmentationLoss,
    map_brats_labels,
)
from models.denoiser import URSSMDenoiser
from models.seg_head import SegmentationHead3D
from models.vqgan3d import VQGAN3D, compute_psnr
from physics.corruption_operator import PhysicsCorruptionOperator

logger = logging.getLogger("train")


# ======================================================================= #
#  Configuration                                                           #
# ======================================================================= #

@dataclass
class TrainingConfig:
    """All training hyperparameters. Serialisable to/from YAML."""

    # ── Paths ─────────────────────────────────────────────────────────
    data_root: str = ""
    output_dir: str = ""
    vqgan_ckpt: str = ""                   # Phase 1 output

    # ── Model ─────────────────────────────────────────────────────────
    latent_dim: int = 4
    downsample_factor: int = 4             # r ∈ {4, 8}
    d_h: int = 128
    stage_depths: Tuple[int, ...] = (2, 2, 4)
    bottleneck_depth: int = 4
    n_classes: int = 4

    # ── Diffusion ─────────────────────────────────────────────────────
    T: int = 1000
    ddim_steps: int = 50
    rho: float = 2.0

    # ── Phase 2: Denoiser pre-train ───────────────────────────────────
    phase2_epochs: int = 100
    phase2_lr: float = 2e-4
    phase2_warmup: int = 5000
    phase2_min_lr: float = 1e-6

    # ── Phase 3: Fine-tune (Regime A) ─────────────────────────────────
    phase3_epochs: int = 50
    phase3_denoiser_lr: float = 2e-5
    phase3_seg_lr: float = 1e-3
    phase3_warmup: int = 2000
    phase3_min_lr: float = 1e-6
    gamma_target: float = 0.1
    gamma_warmup: int = 2000
    early_stop_patience: int = 5000        # steps on val DSC-WT

    # ── Shared training ───────────────────────────────────────────────
    micro_batch: int = 1
    grad_accum: int = 4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    amp_dtype: str = "bfloat16"
    num_workers: int = 4
    use_checkpoint: bool = True

    # ── Variance bounds ───────────────────────────────────────────────
    log_var_min: float = -10.0
    log_var_max: float = 10.0

    # ── CV ────────────────────────────────────────────────────────────
    n_folds: int = 5

    # ── Logging ───────────────────────────────────────────────────────
    log_every: int = 50
    val_every_epochs: int = 5
    save_every_epochs: int = 10

    def save(self, path: str) -> None:
        d = asdict(self)
        # Convert tuples to lists for YAML safe_load compatibility
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        # Convert lists back to tuples
        if "stage_depths" in d and isinstance(d["stage_depths"], list):
            d["stage_depths"] = tuple(d["stage_depths"])
        if "betas" in d and isinstance(d["betas"], list):
            d["betas"] = tuple(d["betas"])
        return cls(**d)


# ======================================================================= #
#  DDP Utilities                                                           #
# ======================================================================= #

def setup_ddp() -> Tuple[int, int, torch.device]:
    """Initialise DDP and return (rank, world_size, device)."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, dist.get_world_size(), device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ======================================================================= #
#  LR Scheduler: warmup + cosine                                           #
# ======================================================================= #

class WarmupCosineScheduler:
    """Linear warmup then cosine annealing to min_lr."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 1e-6) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self._step <= self.warmup_steps:
                lr = base_lr * self._step / max(self.warmup_steps, 1)
            else:
                progress = (self._step - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1)
                progress = min(progress, 1.0)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress))
            pg["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ======================================================================= #
#  Validation: compute DSC                                                 #
# ======================================================================= #

@torch.no_grad()
def compute_dice_scores(
    seg_pred: torch.Tensor, target: torch.Tensor, n_classes: int = 4,
) -> Dict[str, float]:
    """
    Compute per-region Dice scores (BraTS convention).

    BraTS regions from mapped labels {0,1,2,3}:
      WT (Whole Tumor):     classes {1, 2, 3}
      TC (Tumor Core):      classes {1, 3}
      ET (Enhancing Tumor): class {3}
    """
    pred = seg_pred.long()
    tgt = target.long()

    def _dice(p_mask, t_mask):
        inter = (p_mask & t_mask).sum().float()
        union = p_mask.sum().float() + t_mask.sum().float()
        if union < 1:
            return 1.0 if inter < 1 else 0.0
        return (2.0 * inter / union).item()

    # BraTS regions
    wt_pred = (pred >= 1)
    wt_tgt  = (tgt >= 1)
    tc_pred = (pred == 1) | (pred == 3)
    tc_tgt  = (tgt == 1) | (tgt == 3)
    et_pred = (pred == 3)
    et_tgt  = (tgt == 3)

    return {
        "dsc_wt": _dice(wt_pred, wt_tgt),
        "dsc_tc": _dice(tc_pred, tc_tgt),
        "dsc_et": _dice(et_pred, et_tgt),
    }


# ======================================================================= #
#  Phase 2: Denoiser Pre-training (L_diff only)                            #
# ======================================================================= #

def train_phase2(
    cfg: TrainingConfig,
    fold: int,
    rank: int, world_size: int, device: torch.device,
) -> str:
    """
    Phase 2 — Denoiser pre-training with L_diff only.

    Returns path to best checkpoint.
    """
    if is_main():
        logger.info(f"=== Phase 2: Denoiser Pre-train | Fold {fold} ===")

    fold_dir = os.path.join(cfg.output_dir, f"fold_{fold}", "phase2")
    if is_main():
        os.makedirs(fold_dir, exist_ok=True)

    amp_dt = getattr(torch, cfg.amp_dtype, torch.bfloat16)

    # ── Data ──────────────────────────────────────────────────────────
    fold_json = os.path.join(cfg.data_root, f"fold_{fold}.json")
    train_dl, val_dl = build_brats_dataloaders(
        cfg.data_root, fold_json, cfg.micro_batch, cfg.num_workers)

    # ── Models ────────────────────────────────────────────────────────
    vqgan = VQGAN3D(
        in_channels=4, latent_dim=cfg.latent_dim,
        downsample_factor=cfg.downsample_factor,
    ).to(device)
    if cfg.vqgan_ckpt and os.path.exists(cfg.vqgan_ckpt):
        ckpt = torch.load(cfg.vqgan_ckpt, map_location=device, weights_only=True)
        vqgan.load_state_dict(ckpt.get("gen", ckpt), strict=False)
        if is_main():
            logger.info(f"  Loaded VQGAN: {cfg.vqgan_ckpt}")
    vqgan.eval()
    for p in vqgan.parameters():
        p.requires_grad_(False)

    denoiser = URSSMDenoiser(
        latent_dim=cfg.latent_dim, d_h=cfg.d_h,
        stage_depths=cfg.stage_depths,
        bottleneck_depth=cfg.bottleneck_depth,
        use_checkpoint=cfg.use_checkpoint,
    ).to(device)
    denoiser = DDP(denoiser, device_ids=[device.index])

    diffusion = PhysicsConsistentDiffusion(
        T=cfg.T, schedule="cosine", rho=cfg.rho).to(device)
    corruption = PhysicsCorruptionOperator(device=str(device))
    hetero_loss = HeteroscedasticDiffusionLoss(
        sigma_min=math.exp(0.5 * cfg.log_var_min),
        sigma_max=math.exp(0.5 * cfg.log_var_max),
    ).to(device)

    # ── Optimizer + scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.phase2_lr,
        weight_decay=cfg.weight_decay, betas=cfg.betas)

    steps_per_epoch = len(train_dl) // cfg.grad_accum
    total_steps = cfg.phase2_epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.phase2_warmup, total_steps, cfg.phase2_min_lr)

    if is_main():
        logger.info(f"  Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")
        logger.info(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps}")
        n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        logger.info(f"  Denoiser params: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────
    best_loss = float("inf")
    best_ckpt = os.path.join(fold_dir, "best_denoiser.pt")
    gstep = 0

    for epoch in range(cfg.phase2_epochs):
        train_dl.sampler.set_epoch(epoch)
        denoiser.train()
        optimizer.zero_grad()
        epoch_losses = []

        for step, batch in enumerate(train_dl):
            x_0 = batch["image"].to(device, non_blocking=True) # [B,4,128,128,128]

            with autocast("cuda", dtype=amp_dt):
                # Dual independent corruptions
                with torch.no_grad():
                    y = corruption(x_0)
                    x_phys = corruption(x_0)
                    z_0 = vqgan(x_0, mode="encode")
                    z_obs = vqgan(y, mode="encode")
                    z_0_phys = vqgan(x_phys, mode="encode")

                t = diffusion.sample_timesteps(x_0.shape[0], device)
                z_t, eps = diffusion.q_sample(z_0, z_0_phys, t)

                eps_theta, v_theta = denoiser(z_t, t, z_obs)
                l_diff, v_tilde, logs = hetero_loss(eps, eps_theta, v_theta)
                loss = l_diff / cfg.grad_accum

            loss.backward()
            epoch_losses.append(l_diff.item())

            if (step + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                gstep += 1

                if is_main() and gstep % cfg.log_every == 0:
                    logger.info(
                        f"  [P2 f{fold}] ep={epoch} s={gstep} "
                        f"l_diff={logs['l_diff']:.4f} "
                        f"lr={scheduler.current_lr:.2e}")

        # ── Validation ────────────────────────────────────────────────
        if (epoch + 1) % cfg.val_every_epochs == 0:
            denoiser.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_dl:
                    xv = vb["image"].to(device)
                    with autocast("cuda", dtype=amp_dt):
                        with torch.no_grad():
                            yv = corruption(xv)
                            xpv = corruption(xv)
                            zv = vqgan(xv, mode="encode")
                            zov = vqgan(yv, mode="encode")
                            zpv = vqgan(xpv, mode="encode")
                        tv = diffusion.sample_timesteps(xv.shape[0], device)
                        ztv, epsv = diffusion.q_sample(zv, zpv, tv)
                        ep, vp = denoiser(ztv, tv, zov)
                        ld, _, _ = hetero_loss(epsv, ep, vp)
                    val_losses.append(ld.item())

            avg_val = np.mean(val_losses) if val_losses else float("inf")

            # All-reduce for consistent metric
            val_t = torch.tensor(avg_val, device=device)
            dist.all_reduce(val_t, op=dist.ReduceOp.AVG)
            avg_val = val_t.item()

            if is_main():
                logger.info(f"  [P2 Val] ep={epoch+1} l_diff={avg_val:.4f}")
                if avg_val < best_loss:
                    best_loss = avg_val
                    torch.save({
                        "epoch": epoch + 1, "step": gstep,
                        "denoiser": denoiser.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "config": asdict(cfg),
                    }, best_ckpt)
                    logger.info(f"  ✓ Best l_diff={best_loss:.4f} saved")

        # Periodic save
        if is_main() and (epoch + 1) % cfg.save_every_epochs == 0:
            torch.save({
                "epoch": epoch + 1,
                "denoiser": denoiser.module.state_dict(),
            }, os.path.join(fold_dir, f"denoiser_ep{epoch+1:04d}.pt"))

    if is_main():
        logger.info(f"  Phase 2 done. Best l_diff={best_loss:.4f}")
    return best_ckpt


# ======================================================================= #
#  Phase 3: Fine-tune Regime A (L_diff + γ·L_seg)                         #
# ======================================================================= #

def train_phase3(
    cfg: TrainingConfig,
    fold: int,
    phase2_ckpt: str,
    rank: int, world_size: int, device: torch.device,
) -> str:
    """
    Phase 3 — Joint fine-tuning with Regime A.

    Returns path to best checkpoint.
    """
    if is_main():
        logger.info(f"=== Phase 3: Fine-tune Regime A | Fold {fold} ===")

    fold_dir = os.path.join(cfg.output_dir, f"fold_{fold}", "phase3")
    if is_main():
        os.makedirs(fold_dir, exist_ok=True)

    amp_dt = getattr(torch, cfg.amp_dtype, torch.bfloat16)

    # ── Data ──────────────────────────────────────────────────────────
    fold_json = os.path.join(cfg.data_root, f"fold_{fold}.json")
    train_dl, val_dl = build_brats_dataloaders(
        cfg.data_root, fold_json, cfg.micro_batch, cfg.num_workers)

    # ── Models ────────────────────────────────────────────────────────
    vqgan = VQGAN3D(
        in_channels=4, latent_dim=cfg.latent_dim,
        downsample_factor=cfg.downsample_factor,
    ).to(device)
    if cfg.vqgan_ckpt and os.path.exists(cfg.vqgan_ckpt):
        ckpt = torch.load(cfg.vqgan_ckpt, map_location=device, weights_only=True)
        vqgan.load_state_dict(ckpt.get("gen", ckpt), strict=False)
    vqgan.eval()
    for p in vqgan.parameters():
        p.requires_grad_(False)

    denoiser = URSSMDenoiser(
        latent_dim=cfg.latent_dim, d_h=cfg.d_h,
        stage_depths=cfg.stage_depths,
        bottleneck_depth=cfg.bottleneck_depth,
        use_checkpoint=cfg.use_checkpoint,
    ).to(device)

    # Load Phase 2 checkpoint
    if phase2_ckpt and os.path.exists(phase2_ckpt):
        p2 = torch.load(phase2_ckpt, map_location=device, weights_only=True)
        denoiser.load_state_dict(p2.get("denoiser", p2), strict=False)
        if is_main():
            logger.info(f"  Loaded Phase 2: {phase2_ckpt}")

    seg_head = SegmentationHead3D(
        latent_dim=cfg.latent_dim, n_classes=cfg.n_classes,
        downsample_factor=cfg.downsample_factor,
    ).to(device)

    denoiser = DDP(denoiser, device_ids=[device.index])
    seg_head = DDP(seg_head, device_ids=[device.index])

    diffusion = PhysicsConsistentDiffusion(
        T=cfg.T, schedule="cosine", rho=cfg.rho).to(device)
    corruption = PhysicsCorruptionOperator(device=str(device))
    hetero_loss = HeteroscedasticDiffusionLoss(
        sigma_min=math.exp(0.5 * cfg.log_var_min),
        sigma_max=math.exp(0.5 * cfg.log_var_max),
    ).to(device)
    seg_loss_fn = SegmentationLoss(n_classes=cfg.n_classes).to(device)
    gamma_sched = GammaScheduler(cfg.gamma_target, cfg.gamma_warmup)

    # ── Optimizers (different LR for denoiser vs seg head) ────────────
    optimizer = torch.optim.AdamW([
        {"params": denoiser.parameters(), "lr": cfg.phase3_denoiser_lr},
        {"params": seg_head.parameters(), "lr": cfg.phase3_seg_lr},
    ], weight_decay=cfg.weight_decay, betas=cfg.betas)

    steps_per_epoch = len(train_dl) // cfg.grad_accum
    total_steps = cfg.phase3_epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.phase3_warmup, total_steps, cfg.phase3_min_lr)

    if is_main():
        logger.info(f"  Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")
        logger.info(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps}")

    # ── Training loop ─────────────────────────────────────────────────
    best_dsc_wt = -1.0
    best_ckpt = os.path.join(fold_dir, "best_model.pt")
    gstep = 0
    patience_counter = 0

    for epoch in range(cfg.phase3_epochs):
        train_dl.sampler.set_epoch(epoch)
        denoiser.train()
        seg_head.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            x_0 = batch["image"].to(device, non_blocking=True)
            s_0 = batch["label"].to(device, non_blocking=True)

            with autocast("cuda", dtype=amp_dt):
                with torch.no_grad():
                    y = corruption(x_0)
                    x_phys = corruption(x_0)
                    z_0 = vqgan(x_0, mode="encode")
                    z_obs = vqgan(y, mode="encode")
                    z_0_phys = vqgan(x_phys, mode="encode")

                t = diffusion.sample_timesteps(x_0.shape[0], device)
                z_t, eps = diffusion.q_sample(z_0, z_0_phys, t)

                eps_theta, v_theta = denoiser(z_t, t, z_obs)
                l_diff, v_tilde, diff_logs = hetero_loss(
                    eps, eps_theta, v_theta)

                # Tweedie single-step x0 estimate (NOT full DDIM)
                z_0_pred = diffusion.tweedie_x0_estimate(
                    z_t, eps_theta, t, z_obs)

                seg_logits = seg_head(z_0_pred)
                s_0_mapped = map_brats_labels(s_0)
                l_seg, seg_logs = seg_loss_fn(seg_logits, s_0_mapped)

                gamma = gamma_sched(gstep)
                l_total = (l_diff + gamma * l_seg) / cfg.grad_accum

            l_total.backward()

            if (step + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                nn.utils.clip_grad_norm_(seg_head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                gstep += 1

                if is_main() and gstep % cfg.log_every == 0:
                    logger.info(
                        f"  [P3 f{fold}] ep={epoch} s={gstep} "
                        f"l_diff={diff_logs['l_diff']:.4f} "
                        f"l_seg={seg_logs['l_seg']:.4f} "
                        f"γ={gamma:.4f} "
                        f"lr_d={optimizer.param_groups[0]['lr']:.2e} "
                        f"lr_s={optimizer.param_groups[1]['lr']:.2e}")

        # ── Validation ────────────────────────────────────────────────
        if (epoch + 1) % cfg.val_every_epochs == 0:
            denoiser.eval()
            seg_head.eval()
            all_dsc = {"dsc_wt": [], "dsc_tc": [], "dsc_et": []}

            with torch.no_grad():
                for vb in val_dl:
                    xv = vb["image"].to(device)
                    sv = vb["label"].to(device)
                    sv_mapped = map_brats_labels(sv)

                    with autocast("cuda", dtype=amp_dt):
                        with torch.no_grad():
                            yv = corruption(xv)
                            zov = vqgan(yv, mode="encode")
                            zv = vqgan(xv, mode="encode")
                            zpv = vqgan(corruption(xv), mode="encode")
                        tv = diffusion.sample_timesteps(xv.shape[0], device)
                        ztv, epsv = diffusion.q_sample(zv, zpv, tv)
                        ep, vp = denoiser(ztv, tv, zov)
                        z0p = diffusion.tweedie_x0_estimate(ztv, ep, tv, zov)
                        seg_log = seg_head(z0p)

                    pred = seg_log.argmax(dim=1)
                    for b in range(xv.shape[0]):
                        scores = compute_dice_scores(pred[b], sv_mapped[b])
                        for k in all_dsc:
                            all_dsc[k].append(scores[k])

            # Average and all-reduce
            avg_dsc = {}
            for k in all_dsc:
                val = np.mean(all_dsc[k]) if all_dsc[k] else 0.0
                t_val = torch.tensor(val, device=device)
                dist.all_reduce(t_val, op=dist.ReduceOp.AVG)
                avg_dsc[k] = t_val.item()

            if is_main():
                logger.info(
                    f"  [P3 Val] ep={epoch+1} "
                    f"DSC_WT={avg_dsc['dsc_wt']:.4f} "
                    f"DSC_TC={avg_dsc['dsc_tc']:.4f} "
                    f"DSC_ET={avg_dsc['dsc_et']:.4f}")

                if avg_dsc["dsc_wt"] > best_dsc_wt:
                    best_dsc_wt = avg_dsc["dsc_wt"]
                    patience_counter = 0
                    torch.save({
                        "epoch": epoch + 1, "step": gstep,
                        "denoiser": denoiser.module.state_dict(),
                        "seg_head": seg_head.module.state_dict(),
                        "best_dsc_wt": best_dsc_wt,
                        "config": asdict(cfg),
                    }, best_ckpt)
                    logger.info(f"  ✓ Best DSC_WT={best_dsc_wt:.4f} saved")
                else:
                    patience_counter += steps_per_epoch * cfg.val_every_epochs

            # Broadcast early stopping decision
            stop = torch.tensor(
                1 if patience_counter >= cfg.early_stop_patience else 0,
                device=device)
            dist.broadcast(stop, src=0)
            if stop.item() == 1:
                if is_main():
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # Periodic save
        if is_main() and (epoch + 1) % cfg.save_every_epochs == 0:
            torch.save({
                "epoch": epoch + 1,
                "denoiser": denoiser.module.state_dict(),
                "seg_head": seg_head.module.state_dict(),
            }, os.path.join(fold_dir, f"model_ep{epoch+1:04d}.pt"))

    if is_main():
        logger.info(f"  Phase 3 done. Best DSC_WT={best_dsc_wt:.4f}")
    return best_ckpt


# ======================================================================= #
#  5-Fold CV Orchestrator                                                  #
# ======================================================================= #

def run_cv(cfg: TrainingConfig, phases: List[int],
           rank: int, world_size: int, device: torch.device) -> None:
    """Run specified phases across all folds."""
    for fold in range(cfg.n_folds):
        if is_main():
            logger.info(f"\n{'='*60}")
            logger.info(f"  FOLD {fold}/{cfg.n_folds - 1}")
            logger.info(f"{'='*60}")

        fold_json = os.path.join(cfg.data_root, f"fold_{fold}.json")
        if not os.path.exists(fold_json):
            if is_main():
                logger.warning(f"  fold_{fold}.json not found — skipping")
            continue

        phase2_ckpt = os.path.join(
            cfg.output_dir, f"fold_{fold}", "phase2", "best_denoiser.pt")

        if 2 in phases:
            phase2_ckpt = train_phase2(cfg, fold, rank, world_size, device)
        if 3 in phases:
            train_phase3(cfg, fold, phase2_ckpt, rank, world_size, device)

        dist.barrier()


# ======================================================================= #
#  Default config generator                                                #
# ======================================================================= #

def generate_default_config(output_path: str) -> None:
    """Generate a default config.yaml."""
    DR = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS"
    cfg = TrainingConfig(
        data_root=os.path.join(DR, "UR_SSM_Diff_Outputs/preprocessed/BraTS2021"),
        output_dir=os.path.join(DR, "UR_SSM_Diff_Outputs/training"),
        vqgan_ckpt=os.path.join(DR, "UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"),
    )
    cfg.save(output_path)
    print(f"Default config saved to: {output_path}")


# ======================================================================= #
#  CLI Entry Point                                                         #
# ======================================================================= #

def main() -> None:
    # Jupyter guard
    _in_jupyter = any("ipykernel" in a or "kernel" in a for a in sys.argv)
    if _in_jupyter:
        print("  ⚠ train.py requires DDP — run from terminal:")
        print("      torchrun --nproc_per_node=2 train.py --config config.yaml")
        print("      torchrun --nproc_per_node=2 train.py --phase 2 --fold 0")
        return

    parser = argparse.ArgumentParser("UR-SSM-Diff Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate default config.yaml and exit")
    parser.add_argument("--phase", type=int, nargs="+", default=[2, 3],
                        help="Training phases to run (2=pretrain, 3=finetune)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Single fold to run (None=all folds)")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--vqgan-ckpt", type=str, default=None)
    parser.add_argument("--r", type=int, default=None, choices=[4, 8])
    args = parser.parse_args()

    if args.generate_config:
        generate_default_config(args.config)
        return

    # Load or create config
    if os.path.exists(args.config):
        cfg = TrainingConfig.load(args.config)
    else:
        cfg = TrainingConfig()

    # CLI overrides
    if args.data_root:
        cfg.data_root = args.data_root
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.vqgan_ckpt:
        cfg.vqgan_ckpt = args.vqgan_ckpt
    if args.r:
        cfg.downsample_factor = args.r

    # Setup DDP
    rank, world_size, device = setup_ddp()

    if is_main():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger.info(f"DDP: rank={rank} world={world_size} device={device}")
        logger.info(f"Config: r={cfg.downsample_factor} d_h={cfg.d_h}")
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.save(os.path.join(cfg.output_dir, "config.yaml"))

    # Handle single fold or all folds
    if args.fold is not None:
        # Single fold
        fold_json = os.path.join(cfg.data_root, f"fold_{args.fold}.json")
        assert os.path.exists(fold_json), f"Not found: {fold_json}"

        phase2_ckpt = os.path.join(
            cfg.output_dir, f"fold_{args.fold}", "phase2", "best_denoiser.pt")

        if 2 in args.phase:
            phase2_ckpt = train_phase2(
                cfg, args.fold, rank, world_size, device)
        if 3 in args.phase:
            train_phase3(
                cfg, args.fold, phase2_ckpt, rank, world_size, device)
    else:
        run_cv(cfg, args.phase, rank, world_size, device)

    cleanup_ddp()
    if is_main():
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
