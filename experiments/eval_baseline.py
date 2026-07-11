#!/usr/bin/env python3
"""
Evaluate a trained SegMamba (or TransBTS) checkpoint with full metrics:
  DSC (WT/TC/ET), HD95 (WT/TC/ET), Surface Dice τ=1mm (WT/TC/ET),
  PSNR (dB), SSIM — all brain-masked 3D.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=1 python experiments/eval_baseline.py \
      --baseline segmamba --fold 0

  CUDA_VISIBLE_DEVICES=0 python experiments/eval_baseline.py \
      --baseline segmamba --fold 0
"""

import argparse, json, logging, os, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.brats_dataset import BraTSDataset
from losses.segmentation_loss import map_brats_labels
from physics.corruption_operator import PhysicsCorruptionOperator

logger = logging.getLogger("eval_baseline")


# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════

def _surface_points(mask, spacing=(1.0, 1.0, 1.0)):
    from scipy import ndimage
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64)
    border = mask & ~ndimage.binary_erosion(mask, iterations=1)
    coords = np.argwhere(border).astype(np.float64)
    if coords.size == 0:
        return coords
    return coords * np.array(spacing)


def compute_hd95(pred, target, spacing=(1.0, 1.0, 1.0)):
    from scipy.spatial.distance import cdist
    p, t = pred.astype(bool), target.astype(bool)
    if p.sum() == 0 and t.sum() == 0:
        return 0.0
    if p.sum() == 0 or t.sum() == 0:
        return float("nan")
    sp, st = _surface_points(p, spacing), _surface_points(t, spacing)
    if sp.shape[0] == 0 or st.shape[0] == 0:
        return float("nan")
    mx = 10000
    if sp.shape[0] > mx:
        sp = sp[np.random.choice(sp.shape[0], mx, replace=False)]
    if st.shape[0] > mx:
        st = st[np.random.choice(st.shape[0], mx, replace=False)]
    d1 = cdist(sp, st).min(axis=1)
    d2 = cdist(st, sp).min(axis=1)
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def compute_surface_dice(pred, target, tau=1.0, spacing=(1.0, 1.0, 1.0)):
    from scipy.spatial.distance import cdist
    p, t = pred.astype(bool), target.astype(bool)
    if p.sum() == 0 and t.sum() == 0:
        return 1.0
    if p.sum() == 0 or t.sum() == 0:
        return 0.0
    sp, st = _surface_points(p, spacing), _surface_points(t, spacing)
    if sp.shape[0] == 0 or st.shape[0] == 0:
        return 0.0
    mx = 10000
    sp_n, st_n = sp.shape[0], st.shape[0]
    if sp.shape[0] > mx:
        sp = sp[np.random.choice(sp.shape[0], mx, replace=False)]
    if st.shape[0] > mx:
        st = st[np.random.choice(st.shape[0], mx, replace=False)]
    d1 = cdist(sp, st).min(axis=1)
    d2 = cdist(st, sp).min(axis=1)
    n1 = (d1 <= tau).sum() * (sp_n / sp.shape[0])
    n2 = (d2 <= tau).sum() * (st_n / st.shape[0])
    return float((n1 + n2) / (sp_n + st_n))


def compute_psnr(pred, target, mask=None):
    if mask is None:
        mask = np.any(np.abs(target) > 1e-6, axis=0) if target.ndim == 4 \
            else (np.abs(target) > 1e-6)
    if pred.ndim == 4 and mask.ndim == 3:
        mask = np.broadcast_to(mask[None], pred.shape)
    pm, tm = pred[mask], target[mask]
    if tm.size == 0:
        return 0.0
    dr = float(tm.max() - tm.min())
    if dr < 1e-8:
        return 0.0
    mse = float(np.mean((pm - tm) ** 2))
    return 100.0 if mse < 1e-12 else float(10 * np.log10(dr ** 2 / mse))


def compute_ssim_3d(pred, target, mask=None):
    from scipy.ndimage import gaussian_filter
    if mask is None:
        mask = np.any(np.abs(target) > 1e-6, axis=0) if target.ndim == 4 \
            else (np.abs(target) > 1e-6)
    sigma = 1.5

    def _ch(p, t, m):
        if m.sum() == 0:
            return 0.0
        dr = float(t[m].max() - t[m].min())
        if dr < 1e-8:
            dr = 1.0
        c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
        mp = gaussian_filter(p, sigma)
        mt = gaussian_filter(t, sigma)
        spp = gaussian_filter(p * p, sigma) - mp * mp
        stt = gaussian_filter(t * t, sigma) - mt * mt
        spt = gaussian_filter(p * t, sigma) - mp * mt
        s = ((2*mp*mt + c1) * (2*spt + c2)) / \
            ((mp**2 + mt**2 + c1) * (spp + stt + c2))
        return float(s[m].mean())

    if pred.ndim == 4:
        return float(np.mean([_ch(pred[c], target[c], mask)
                              for c in range(pred.shape[0])]))
    return _ch(pred, target, mask)


def compute_all_metrics(pred_labels, target_labels, spacing=(1.0, 1.0, 1.0)):
    """DSC, HD95, Surface Dice for WT/TC/ET."""
    regions = {
        "wt": lambda x: (x >= 1),
        "tc": lambda x: (x == 1) | (x == 3),
        "et": lambda x: (x == 3),
    }
    out = {}
    for name, fn in regions.items():
        pb, tb = fn(pred_labels).astype(bool), fn(target_labels).astype(bool)
        inter = (pb & tb).sum()
        union = pb.sum() + tb.sum()
        out[f"dsc_{name}"] = float(1.0 if union == 0 else 2.0 * inter / union)
        out[f"hd95_{name}"] = compute_hd95(pb, tb, spacing)
        out[f"sd_{name}"] = compute_surface_dice(pb, tb, 1.0, spacing)
    return out


# ══════════════════════════════════════════════════════════════════════
#  Import model classes from new_baselines.py
# ══════════════════════════════════════════════════════════════════════

# We import the model classes directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from new_baselines import SegMamba3D, TransBTS3D


# ══════════════════════════════════════════════════════════════════════
#  Main evaluation
# ══════════════════════════════════════════════════════════════════════

DRIVE_ROOT = (
    "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/"
    "NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS"
)

def main():
    parser = argparse.ArgumentParser("Evaluate baseline checkpoint")
    parser.add_argument("--baseline", required=True,
                        choices=["segmamba", "transbts"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-root", default=os.path.join(
        DRIVE_ROOT, "UR_SSM_Diff_Outputs/preprocessed/BraTS2021"))
    parser.add_argument("--output-dir", default=os.path.join(
        DRIVE_ROOT, "UR_SSM_Diff_Outputs/baselines"))
    parser.add_argument("--ckpt", default=None,
                        help="Override checkpoint path (default: auto-detect)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build model ───────────────────────────────────────────────────
    if args.baseline == "segmamba":
        model = SegMamba3D(
            in_channels=4, n_classes=4, base_dim=64,
            n_stages=4, d_state=16, mamba_start_stage=2)
    elif args.baseline == "transbts":
        model = TransBTS3D(
            in_channels=4, n_classes=4, base_dim=32,
            n_heads=8, n_transformer_layers=4)

    # ── Load checkpoint ───────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(
            args.output_dir, args.baseline,
            f"fold_{args.fold}", "best_model.pt")

    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    best_epoch = ckpt.get("epoch", "?")
    best_dsc = ckpt.get("best_dsc", "?")
    logger.info(f"Loaded {args.baseline} fold {args.fold} | "
                f"epoch={best_epoch} best_dsc={best_dsc}")
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Data ──────────────────────────────────────────────────────────
    fold_json = os.path.join(args.data_root, f"fold_{args.fold}.json")
    val_ds = BraTSDataset(
        args.data_root, augment=False, fold_json=fold_json, split="val")
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    logger.info(f"Val subjects: {len(val_ds)}")

    corruption = PhysicsCorruptionOperator(device=str(device))

    # ── Evaluate ──────────────────────────────────────────────────────
    all_results: List[Dict] = []
    metric_keys = [
        "dsc_wt", "dsc_tc", "dsc_et",
        "hd95_wt", "hd95_tc", "hd95_et",
        "sd_wt", "sd_tc", "sd_et",
        "psnr_corrupted", "ssim_corrupted",
    ]

    with torch.no_grad():
        for idx, batch in enumerate(val_dl):
            sid = batch["id"][0]
            x_0 = batch["image"].to(device)  # clean
            s_0 = batch["label"].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                y = corruption(x_0)            # corrupted
                logits = model(y)              # predict from corrupted

            if logits.dim() == 6:
                logits = logits[:, 0]

            pred_np = logits.argmax(1)[0].cpu().numpy()
            tgt_np = map_brats_labels(s_0)[0].cpu().numpy()

            # Segmentation metrics
            seg_m = compute_all_metrics(pred_np, tgt_np)

            # Restoration metrics (corrupted vs clean — corruption severity)
            y_np = y[0].cpu().float().numpy()
            x_np = x_0[0].cpu().numpy()
            psnr_c = compute_psnr(y_np, x_np)
            ssim_c = compute_ssim_3d(y_np, x_np)

            result = {"subject": sid, **seg_m,
                      "psnr_corrupted": psnr_c, "ssim_corrupted": ssim_c}
            all_results.append(result)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(val_ds):
                logger.info(
                    f"  [{idx+1}/{len(val_ds)}] {sid}  "
                    f"DSC_WT={seg_m['dsc_wt']:.4f}  "
                    f"HD95_WT={seg_m['hd95_wt']:.2f}  "
                    f"SD_WT={seg_m['sd_wt']:.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"  {args.baseline.upper()} — Fold {args.fold} — "
                f"{len(all_results)} subjects")
    logger.info("=" * 70)

    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in all_results if not np.isnan(r[k])]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            summary[k] = {"mean": mean, "std": std, "n": len(vals)}
            logger.info(f"  {k:20s}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")
        else:
            summary[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            logger.info(f"  {k:20s}: N/A")

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = os.path.join(
        args.output_dir, args.baseline, f"fold_{args.fold}")
    os.makedirs(out_dir, exist_ok=True)

    # Per-subject results
    per_subj_path = os.path.join(out_dir, "eval_per_subject.json")
    with open(per_subj_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    summary_path = os.path.join(out_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nPer-subject results → {per_subj_path}")
    logger.info(f"Summary            → {summary_path}")

    # ── Print LaTeX-ready row ─────────────────────────────────────────
    def _fmt(k):
        s = summary[k]
        if np.isnan(s["mean"]):
            return "---"
        return f"{s['mean']:.3f}{{\\pm}}{s['std']:.3f}"

    logger.info(f"\nLaTeX table row:")
    logger.info(
        f"  {args.baseline} & "
        f"${_fmt('dsc_wt')}$ & ${_fmt('dsc_tc')}$ & ${_fmt('dsc_et')}$ & "
        f"${_fmt('hd95_wt')}$ & ${_fmt('hd95_tc')}$ & ${_fmt('hd95_et')}$ & "
        f"${_fmt('sd_wt')}$ & ${_fmt('sd_tc')}$ & ${_fmt('sd_et')}$ \\\\")


if __name__ == "__main__":
    main()
