#!/usr/bin/env python3
"""
train_vqgan.py — VQGAN3D DDP Pre-Training  (Phase 4, Paper Section 4.2)
========================================================================

Launch from TERMINAL (DDP requires torchrun):
    torchrun --nproc_per_node=2 train_vqgan.py --r 4
    torchrun --nproc_per_node=2 train_vqgan.py --r 8

Hardware: 2x NVIDIA RTX 5880 Ada 48 GB, CUDA 11.8, bfloat16 AMP
"""
from __future__ import annotations
import argparse, json, logging, math, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np, torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.vqgan3d import VQGAN3D, Discriminator3D, VQGANLoss, compute_psnr

logger = logging.getLogger("train_vqgan")

# ===================== Config =====================
@dataclass
class Cfg:
    in_channels:int=4; latent_dim:int=4; base_channels:int=64
    channel_mult:tuple=(1,2,4,8); n_res_blocks:int=2; downsample_factor:int=4
    n_embed:int=8192; commitment_cost:float=0.25
    disc_base_channels:int=64
    lambda_vq:float=1.0; lambda_perc:float=0.1; lambda_adv:float=0.1; disc_start_step:int=2000
    gen_lr:float=1e-4; disc_lr:float=4e-4; weight_decay:float=1e-2; betas:tuple=(0.5,0.9)
    epochs:int=200; micro_batch:int=1; grad_accum:int=4; num_workers:int=4
    amp_dtype:str="bfloat16"
    data_root:str=""; val_fold_json:str=""; output_dir:str=""
    wandb_project:str="ur-ssm-diff-vqgan"; wandb_name:str=""
    log_every:int=50; val_every:int=5; save_every:int=10

# ===================== Dataset =====================
class NpyDataset(Dataset):
    def __init__(self, files: list, augment: bool = True):
        self.files = files; self.augment = augment
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = torch.from_numpy(np.load(self.files[i])).float()  # [4,128,128,128]
        if self.augment:
            for d in [1, 2, 3]:  # spatial dims of [C, H, W, D]
                if torch.rand(1).item() > 0.5: img = img.flip(d)
            if torch.rand(1).item() > 0.5:
                img = torch.rot90(img, int(torch.randint(1, 4, (1,))), [1, 2])
        return img

def _file_lists(data_root, val_json):
    all_f = sorted(Path(data_root).glob("*_image.npy"))
    d = {f.stem.replace("_image",""):str(f) for f in all_f}
    if val_json and os.path.exists(val_json):
        fold = json.load(open(val_json))
        vi, ti = set(fold.get("val",[])), set(fold.get("train",[]))
    else:
        ids = sorted(d.keys()); n = max(1,len(ids)//10)
        vi, ti = set(ids[:n]), set(ids[n:])
    return [d[s] for s in sorted(ti) if s in d], [d[s] for s in sorted(vi) if s in d]

# ===================== Training =====================
def train(cfg: Cfg):
    dist.init_process_group("nccl")
    rank = dist.get_rank(); local_rank = int(os.environ.get("LOCAL_RANK",0))
    device = torch.device(f"cuda:{local_rank}"); torch.cuda.set_device(device)
    is_main = rank == 0
    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        logger.info(f"DDP rank={rank} world={dist.get_world_size()} r={cfg.downsample_factor}")

    tr_f, va_f = _file_lists(cfg.data_root, cfg.val_fold_json)
    if is_main: logger.info(f"Train={len(tr_f)} Val={len(va_f)}")

    tr_ds = NpyDataset(tr_f, True); va_ds = NpyDataset(va_f, False)
    tr_sam = DistributedSampler(tr_ds, shuffle=True)
    va_sam = DistributedSampler(va_ds, shuffle=False)
    tr_dl = DataLoader(tr_ds, cfg.micro_batch, sampler=tr_sam, num_workers=cfg.num_workers,
                       pin_memory=True, persistent_workers=True, drop_last=True)
    va_dl = DataLoader(va_ds, cfg.micro_batch, sampler=va_sam, num_workers=2, pin_memory=True)

    gen = DDP(VQGAN3D(cfg.in_channels, cfg.latent_dim, cfg.base_channels, cfg.channel_mult,
              cfg.n_res_blocks, cfg.downsample_factor, cfg.n_embed, cfg.commitment_cost
              ).to(device), device_ids=[local_rank])
    disc = DDP(Discriminator3D(cfg.in_channels, cfg.disc_base_channels
              ).to(device), device_ids=[local_rank])
    crit = VQGANLoss(cfg.lambda_vq, cfg.lambda_perc, cfg.lambda_adv, cfg.disc_start_step)

    opt_g = torch.optim.AdamW(gen.parameters(), cfg.gen_lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    opt_d = torch.optim.AdamW(disc.parameters(), cfg.disc_lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    total_steps = cfg.epochs * len(tr_dl) // cfg.grad_accum
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, total_steps, 1e-6)
    sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, total_steps, 1e-6)
    amp_dt = getattr(torch, cfg.amp_dtype, torch.bfloat16)

    wandb = None
    if is_main:
        try:
            import wandb as _wb; _wb.init(project=cfg.wandb_project,
                name=cfg.wandb_name or f"vqgan_r{cfg.downsample_factor}", config=vars(cfg))
            wandb = _wb
        except: pass

    ckpt_dir = os.path.join(cfg.output_dir, f"vqgan_r{cfg.downsample_factor}")
    if is_main: os.makedirs(ckpt_dir, exist_ok=True)
    best_psnr = -float("inf"); gstep = 0

    for ep in range(cfg.epochs):
        tr_sam.set_epoch(ep); gen.train(); disc.train()
        opt_g.zero_grad(); opt_d.zero_grad()
        acc_g, acc_d = {}, {}

        for step, batch in enumerate(tr_dl):
            x = batch.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=amp_dt):
                xr, zq, vl, _ = gen(x, mode="vqgan_train")
                dfk_l, dfk_f, drl_f = None, None, None
                if gstep >= cfg.disc_start_step:
                    dfk_l, dfk_f = disc(xr)
                    with torch.no_grad(): _, drl_f = disc(x)
                gl, gl_log = crit.generator_loss(x, xr, vl, dfk_l, dfk_f, drl_f, gstep)
                gl = gl / cfg.grad_accum
            gl.backward()

            if gstep >= cfg.disc_start_step:
                with autocast(device_type="cuda", dtype=amp_dt):
                    dr_l, _ = disc(x); df_l, _ = disc(xr.detach())
                    dl, dl_log = crit.discriminator_loss(dr_l, df_l)
                    dl = dl / cfg.grad_accum
                dl.backward()
            else:
                dl_log = {}

            for k,v in gl_log.items(): acc_g[k] = acc_g.get(k,0)+v/cfg.grad_accum
            for k,v in dl_log.items(): acc_d[k] = acc_d.get(k,0)+v/cfg.grad_accum

            if (step+1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(gen.parameters(), 1.0); opt_g.step(); opt_g.zero_grad(); sch_g.step()
                if gstep >= cfg.disc_start_step:
                    nn.utils.clip_grad_norm_(disc.parameters(), 1.0); opt_d.step(); opt_d.zero_grad(); sch_d.step()
                gstep += 1
                if is_main and gstep % cfg.log_every == 0:
                    logger.info(f"[{ep}/{cfg.epochs}] s={gstep} l1={acc_g.get('l1',0):.4f} "
                                f"ssim={acc_g.get('ssim',0):.4f} vq={acc_g.get('vq_loss',0):.4f}")
                    if wandb: wandb.log({**acc_g,**acc_d,"lr":opt_g.param_groups[0]["lr"],"epoch":ep,"step":gstep})
                acc_g, acc_d = {}, {}

        # Validation
        if (ep+1) % cfg.val_every == 0:
            gen.eval(); psnrs, l1s = [], []
            with torch.no_grad():
                for b in va_dl:
                    xv = b.to(device)
                    with autocast(device_type="cuda", dtype=amp_dt):
                        xvr,_,_,_ = gen(xv, mode="vqgan_train")
                    psnrs.append(compute_psnr(xv, xvr)); l1s.append(F.l1_loss(xv,xvr).item())
            avg_p = sum(psnrs)/max(len(psnrs),1)
            pt = torch.tensor(avg_p, device=device); dist.all_reduce(pt, dist.ReduceOp.AVG)
            avg_p = pt.item()
            if is_main:
                logger.info(f"  [Val] ep={ep+1} PSNR={avg_p:.2f}dB")
                if wandb: wandb.log({"val_psnr":avg_p,"epoch":ep+1})
                if avg_p > best_psnr:
                    best_psnr = avg_p
                    torch.save({"epoch":ep+1,"step":gstep,"gen":gen.module.state_dict(),
                                "disc":disc.module.state_dict(),"opt_g":opt_g.state_dict(),
                                "opt_d":opt_d.state_dict(),"best_psnr":best_psnr,"cfg":vars(cfg)},
                               os.path.join(ckpt_dir,"best_vqgan.pt"))
                    logger.info(f"  ✓ Best PSNR={best_psnr:.2f}dB saved")
        if is_main and (ep+1) % cfg.save_every == 0:
            torch.save({"epoch":ep+1,"gen":gen.module.state_dict()},
                       os.path.join(ckpt_dir,f"vqgan_ep{ep+1:04d}.pt"))

    # Latent statistics
    if is_main:
        logger.info("Computing latent statistics...")
        gen.eval(); means, stds = [], []
        with torch.no_grad():
            for b in DataLoader(NpyDataset(tr_f,False), 1, num_workers=2):
                z = gen(b.to(device), mode="encode")
                means.append(z.mean().item()); stds.append(z.std().item())
                if len(means) >= 200: break
        stats = {"latent_mean":float(np.mean(means)),"latent_std":float(np.mean(stds)),
                 "n_samples":len(means),"r":cfg.downsample_factor}
        json.dump(stats, open(os.path.join(ckpt_dir,"latent_stats.json"),"w"), indent=2)
        logger.info(f"  mean={stats['latent_mean']:.4f} std={stats['latent_std']:.4f}")

    dist.destroy_process_group()
    if is_main:
        logger.info(f"Done. Best PSNR={best_psnr:.2f}dB")
        if wandb: wandb.finish()

# ===================== CLI (Jupyter-safe) =====================
def main():
    _in_jupyter = any("ipykernel" in a or "kernel" in a for a in sys.argv)
    if _in_jupyter:
        print("  ⚠  train_vqgan.py requires DDP — run from terminal:")
        print("      torchrun --nproc_per_node=2 train_vqgan.py --r 4")
        print("      torchrun --nproc_per_node=2 train_vqgan.py --r 8")
        return

    DR = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/UR_SSM_DIFF_DATASETS"
    p = argparse.ArgumentParser("VQGAN3D Training")
    p.add_argument("--r", type=int, default=4, choices=[4,8])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--gen-lr", type=float, default=1e-4)
    p.add_argument("--disc-lr", type=float, default=4e-4)
    p.add_argument("--data-root", type=str, default=os.path.join(DR,"UR_SSM_Diff_Outputs/preprocessed/BraTS2021"))
    p.add_argument("--val-fold-json", type=str, default="")
    p.add_argument("--output-dir", type=str, default=os.path.join(DR,"UR_SSM_Diff_Outputs/checkpoints"))
    p.add_argument("--wandb-project", type=str, default="ur-ssm-diff-vqgan")
    a = p.parse_args()
    train(Cfg(downsample_factor=a.r, epochs=a.epochs, base_channels=a.base_channels,
              gen_lr=a.gen_lr, disc_lr=a.disc_lr, data_root=a.data_root,
              val_fold_json=a.val_fold_json, output_dir=a.output_dir,
              wandb_project=a.wandb_project, wandb_name=f"vqgan_r{a.r}"))

if __name__ == "__main__":
    main()
