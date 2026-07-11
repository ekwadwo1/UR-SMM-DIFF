#!/usr/bin/env python3
"""
evaluate_baselines.py — Full sub-region evaluation for discriminative baselines
================================================================================

Evaluates trained nnU-Net and SwinUNETR checkpoints on all BraTS sub-regions
(WT, TC, ET) + boundary metrics (HD95, Surface Dice) across all 5 folds.

Saves per-subject CSVs for downstream Wilcoxon signed-rank testing.

This script reuses your EXISTING infrastructure:
  - Model classes from experiments/baselines.py (NnUNetBaseline, SwinUNETRBaseline)
  - evaluate_subject() from evaluation/metrics.py
  - BraTSDataset from data/brats_dataset.py
  - map_brats_labels() from losses/segmentation_loss.py
  - PhysicsCorruptionOperator from physics/corruption_operator.py

NO new dependencies. NO retraining. Just inference + metrics on existing checkpoints.

Usage:
  # Quick test (3 subjects, fold 0):
  python evaluate_baselines.py --baseline nnunet --fold 0 --max-subjects 3

  # Full 5-fold evaluation:
  python evaluate_baselines.py --baseline nnunet --all-folds
  python evaluate_baselines.py --baseline swinunetr --all-folds

  # Both baselines, all folds (batch):
  for BL in nnunet swinunetr; do
      python evaluate_baselines.py --baseline $BL --all-folds
  done

Hardware: Single GPU (no DDP needed for eval). ~5-10 min per fold.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

# ── Resolve project root for imports ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR  # evaluate_baselines.py lives at project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.brats_dataset import BraTSDataset
from evaluation.metrics import (
    MetricsAggregator,
    evaluate_subject,
)
from losses.segmentation_loss import map_brats_labels
from physics.corruption_operator import PhysicsCorruptionOperator

# Import baseline model classes
from experiments.baselines import NnUNetBaseline, SwinUNETRBaseline

logger = logging.getLogger("evaluate_baselines")


# ============================================================================
#  Default paths (match your machine layout)
# ============================================================================

DRIVE_ROOT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS"
DEFAULTS = {
    "data_root": os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs/preprocessed/BraTS2021"),
    "baselines_dir": os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs/baselines"),
    "output_dir": os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs/eval_baselines"),
}

# Checkpoint naming convention from baselines.py training
# Pattern: {baselines_dir}/{baseline}/fold_{fold}/best_model.pt
BASELINE_CONFIGS = {
    "nnunet": {
        "class": NnUNetBaseline,
        "kwargs": {"in_channels": 4, "out_channels": 4, "img_size": 128},
        "ckpt_pattern": "{baselines_dir}/nnunet/fold_{fold}/best_model.pt",
    },
    "swinunetr": {
        "class": SwinUNETRBaseline,
        "kwargs": {"in_channels": 4, "out_channels": 4, "img_size": 128},
        "ckpt_pattern": "{baselines_dir}/swinunetr/fold_{fold}/best_model.pt",
    },
}


# ============================================================================
#  Model loading
# ============================================================================

def load_baseline(
    baseline: str,
    fold: int,
    baselines_dir: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained baseline checkpoint.

    Args:
        baseline: 'nnunet' or 'swinunetr'.
        fold: CV fold (0-4).
        baselines_dir: Root directory for baseline checkpoints.
        device: Target device.

    Returns:
        Model in eval mode.

    Reference: experiments/baselines.py checkpoint format.
    """
    cfg = BASELINE_CONFIGS[baseline]
    ckpt_path = cfg["ckpt_pattern"].format(
        baselines_dir=baselines_dir, fold=fold)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"  Loading {baseline} fold {fold}: {ckpt_path}")

    # Instantiate model
    model = cfg["class"](**cfg["kwargs"])

    # Load state dict (handle DDP-wrapped keys)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # baselines.py saves: {"model": model_state, "epoch": ..., "best_dsc": ...}
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # Strip DDP "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v
                      for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Loaded: {n_params:.1f}M params")

    return model


# ============================================================================
#  Core evaluation loop
# ============================================================================

@torch.no_grad()
def evaluate_baseline_fold(
    baseline: str,
    fold: int,
    baselines_dir: str,
    data_root: str,
    output_dir: str,
    device: torch.device,
    max_subjects: Optional[int] = None,
) -> MetricsAggregator:
    """Evaluate one baseline on one fold's validation set.

    Pipeline per subject:
      1. Load clean x_0 and ground truth seg
      2. Apply physics corruption → y (same as training)
      3. Forward pass: y → logits → argmax → seg_pred {0,1,2,3}
      4. Compute DSC_WT/TC/ET, HD95_WT/TC/ET, SD_WT/TC/ET
      5. Save per-subject CSV

    Args:
        baseline: 'nnunet' or 'swinunetr'.
        fold: CV fold (0-4).
        baselines_dir: Directory with baseline checkpoints.
        data_root: Preprocessed BraTS2021 root.
        output_dir: Where to save CSV results.
        device: CUDA device.
        max_subjects: Cap for quick testing.

    Returns:
        MetricsAggregator with per-subject results.
    """
    fold_dir = os.path.join(output_dir, baseline, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # ── Load model ──
    model = load_baseline(baseline, fold, baselines_dir, device)

    # ── Load validation dataset ──
    fold_json = os.path.join(data_root, f"fold_{fold}.json")
    if not os.path.isfile(fold_json):
        raise FileNotFoundError(f"Fold JSON not found: {fold_json}")

    val_ds = BraTSDataset(data_root, augment=False,
                          fold_json=fold_json, split="val")
    n_subjects = len(val_ds) if max_subjects is None else \
        min(max_subjects, len(val_ds))

    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

    logger.info(f"  Evaluating {n_subjects} subjects "
                f"| {baseline} fold {fold}")

    # ── Physics corruption operator (same as training) ──
    corruption_op = PhysicsCorruptionOperator().to(device)

    # ── Evaluation loop ──
    agg = MetricsAggregator()
    t_start = time.time()

    for idx, batch in enumerate(loader):
        if idx >= n_subjects:
            break

        subject_id = batch["id"][0]
        x_0 = batch["image"].to(device)         # [1, 4, 128, 128, 128]
        s_0_raw = batch["label"].to(device)      # [1, 128, 128, 128] {0,1,2,4}
        s_0_mapped = map_brats_labels(s_0_raw)   # [1, 128, 128, 128] {0,1,2,3}

        # Apply physics corruption (same as training-time)
        with autocast("cuda", dtype=torch.bfloat16):
            y = corruption_op(x_0)               # [1, 4, 128, 128, 128]

            # Forward pass → segmentation logits
            logits = model(y)                     # [1, 4, 128, 128, 128]

        # argmax → class prediction {0, 1, 2, 3}
        seg_pred = logits.float().argmax(dim=1).squeeze(0).cpu().numpy()
        # [128, 128, 128] int

        seg_gt = s_0_mapped[0].cpu().numpy()     # [128, 128, 128] int

        # ── Compute all segmentation metrics ──
        # evaluate_subject expects mapped labels {0,1,2,3}
        # It handles WT/TC/ET composite region computation internally
        metrics = evaluate_subject(
            pred_seg=seg_pred,
            target_seg=seg_gt,
            restored=None,          # no restoration for discriminative
            reference=None,         # no PSNR/SSIM needed
            sigma_sq=None,          # no uncertainty
            spacing=(1.0, 1.0, 1.0),
            tolerance_mm=1.0,
        )

        agg.add(subject_id, metrics)

        if (idx + 1) % 25 == 0 or idx == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (idx + 1) * (n_subjects - idx - 1)
            logger.info(
                f"  [{idx+1}/{n_subjects}] {subject_id} "
                f"DSC_WT={metrics.get('dsc_wt', 0):.4f} "
                f"DSC_TC={metrics.get('dsc_tc', 0):.4f} "
                f"DSC_ET={metrics.get('dsc_et', 0):.4f} "
                f"ETA={eta/60:.0f}min")

    # ── Save results ──
    tag = f"{baseline}_fold{fold}"
    csv_path = os.path.join(fold_dir, f"{tag}_per_subject.csv")
    agg.save_csv(csv_path)
    logger.info(f"  ✓ Saved: {csv_path}")

    summary_path = os.path.join(fold_dir, f"{tag}_summary.txt")
    agg.save_summary(summary_path)

    # Print fold summary
    summ = agg.summary()
    logger.info(f"\n  ── {baseline} fold {fold} summary ──")
    for metric in ["dsc_wt", "dsc_tc", "dsc_et", "hd95_wt", "sd_wt"]:
        if metric in summ:
            logger.info(f"    {metric:10s}: "
                        f"{summ[metric]['mean']:.4f} ± "
                        f"{summ[metric]['std']:.4f}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return agg


def evaluate_all_folds(
    baseline: str,
    baselines_dir: str,
    data_root: str,
    output_dir: str,
    device: torch.device,
    max_subjects: Optional[int] = None,
    n_folds: int = 5,
) -> None:
    """Evaluate one baseline across all 5 folds, aggregate results.

    Produces:
      - Per-fold CSVs: {output_dir}/{baseline}/fold_{i}/{baseline}_fold{i}_per_subject.csv
      - Combined CSV:  {output_dir}/{baseline}/{baseline}_all_folds.csv
      - Summary:       {output_dir}/{baseline}/{baseline}_5fold_summary.txt
    """
    logger.info("=" * 70)
    logger.info(f"  {baseline.upper()} — 5-Fold Evaluation (BraTS 2021)")
    logger.info("=" * 70)

    all_aggs = []
    fold_summaries = []

    for fold in range(n_folds):
        ckpt_path = BASELINE_CONFIGS[baseline]["ckpt_pattern"].format(
            baselines_dir=baselines_dir, fold=fold)

        if not os.path.isfile(ckpt_path):
            logger.warning(f"  Fold {fold}: checkpoint not found — skipping")
            continue

        fold_json = os.path.join(data_root, f"fold_{fold}.json")
        if not os.path.isfile(fold_json):
            logger.warning(f"  Fold {fold}: fold JSON not found — skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  FOLD {fold}")
        logger.info(f"{'='*60}")

        agg = evaluate_baseline_fold(
            baseline=baseline,
            fold=fold,
            baselines_dir=baselines_dir,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
            max_subjects=max_subjects,
        )

        all_aggs.append(agg)
        fold_summaries.append(agg.summary())

    # ── Aggregate across folds ──
    if all_aggs:
        combined = MetricsAggregator()
        for a in all_aggs:
            for rec in a.records:
                combined.add(rec["subject_id"],
                             {k: v for k, v in rec.items()
                              if k != "subject_id"})

        # Save combined CSV
        combined_dir = os.path.join(output_dir, baseline)
        os.makedirs(combined_dir, exist_ok=True)

        csv_path = os.path.join(combined_dir,
                                f"{baseline}_all_folds.csv")
        combined.save_csv(csv_path)
        logger.info(f"\n  ✓ Combined CSV: {csv_path}")

        summary_path = os.path.join(combined_dir,
                                    f"{baseline}_5fold_summary.txt")
        combined.save_summary(summary_path)

        # ── Per-fold summary table ──
        logger.info(f"\n{'='*70}")
        logger.info(f"  {baseline.upper()} — 5-Fold Summary")
        logger.info(f"{'='*70}")

        key_metrics = ["dsc_wt", "dsc_tc", "dsc_et",
                       "hd95_wt", "hd95_tc", "hd95_et",
                       "sd_wt", "sd_tc", "sd_et"]

        header = f"{'Fold':<6}" + "".join(f"{m:<12}" for m in key_metrics)
        logger.info(header)
        logger.info("-" * len(header))

        for fold_i, summ in enumerate(fold_summaries):
            vals = []
            for m in key_metrics:
                if m in summ:
                    vals.append(f"{summ[m]['mean']:.4f}")
                else:
                    vals.append("N/A")
            logger.info(f"{fold_i:<6}" + "".join(f"{v:<12}" for v in vals))

        # Mean ± std
        logger.info("-" * len(header))
        overall = combined.summary()
        for row_type, fn in [("Mean", lambda s: f"{s['mean']:.4f}"),
                              ("Std", lambda s: f"{s['std']:.4f}")]:
            vals = []
            for m in key_metrics:
                if m in overall:
                    vals.append(fn(overall[m]))
                else:
                    vals.append("N/A")
            logger.info(f"{row_type:<6}" + "".join(f"{v:<12}" for v in vals))

        logger.info(f"\n  Total subjects: {len(combined.records)}")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baselines on all BraTS sub-regions (5-fold).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test:
  python evaluate_baselines.py --baseline nnunet --fold 0 --max-subjects 5

  # Full 5-fold:
  python evaluate_baselines.py --baseline nnunet --all-folds
  python evaluate_baselines.py --baseline swinunetr --all-folds

  # Both baselines:
  for BL in nnunet swinunetr; do
      python evaluate_baselines.py --baseline $BL --all-folds
  done
        """)

    parser.add_argument("--baseline", type=str, required=True,
                        choices=list(BASELINE_CONFIGS.keys()),
                        help="Which baseline to evaluate")
    parser.add_argument("--fold", type=int, default=0,
                        help="Single fold to evaluate (ignored if --all-folds)")
    parser.add_argument("--all-folds", action="store_true",
                        help="Evaluate all 5 folds")
    parser.add_argument("--data-root", type=str,
                        default=DEFAULTS["data_root"])
    parser.add_argument("--baselines-dir", type=str,
                        default=DEFAULTS["baselines_dir"])
    parser.add_argument("--output-dir", type=str,
                        default=DEFAULTS["output_dir"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Cap subjects per fold for quick testing")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device(args.device)

    if args.all_folds:
        evaluate_all_folds(
            baseline=args.baseline,
            baselines_dir=args.baselines_dir,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=device,
            max_subjects=args.max_subjects,
        )
    else:
        evaluate_baseline_fold(
            baseline=args.baseline,
            fold=args.fold,
            baselines_dir=args.baselines_dir,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=device,
            max_subjects=args.max_subjects,
        )

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
