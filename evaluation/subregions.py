#!/usr/bin/env python3
"""
evaluate_baselines_subregions.py
================================
Re-evaluate trained nnU-Net and SwinUNETR checkpoints on all BraTS
sub-regions (WT, TC, ET) + boundary metrics (HD95, Surface Dice).

Saves per-subject scores as .csv for downstream statistical testing.

Usage:
    torchrun --nproc_per_node=1 evaluate_baselines_subregions.py \
        --method nnunet --fold 0 --output-dir ./eval_results

    # Or batch all folds:
    for FOLD in 0 1 2 3 4; do
        for METHOD in nnunet swinunetr; do
            torchrun --nproc_per_node=1 evaluate_baselines_subregions.py \
                --method $METHOD --fold $FOLD --output-dir ./eval_results
        done
    done
"""

import argparse
import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── MONAI metrics ──────────────────────────────────────────────────────
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
)

# ============================================================================
#  PATHS — adjust these to match your directory structure
# ============================================================================
DRIVE_ROOT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/UR_SSM_DIFF_DATASETS"
BRATS2021_ROOT = os.path.join(DRIVE_ROOT, "BraTS2021_Training_Data")
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs")

# Checkpoint paths — {method}_{fold}.pt or similar
CHECKPOINT_PATTERNS = {
    "nnunet": os.path.join(OUTPUT_DIR, "baselines", "nnunet_fold{fold}",
                           "best_model.pt"),
    "swinunetr": os.path.join(OUTPUT_DIR, "baselines", "swinunetr_fold{fold}",
                              "best_model.pt"),
}

# ============================================================================
#  BraTS SUB-REGION DEFINITIONS
# ============================================================================
# BraTS labels: 0=background, 1=NCR/NET, 2=ED, 4=ET
# Sub-regions are COMPOSITE:
#   WT (Whole Tumor)     = labels {1, 2, 4}
#   TC (Tumor Core)      = labels {1, 4}
#   ET (Enhancing Tumor) = labels {4}

def brats_to_subregions(seg: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert BraTS integer labels to binary sub-region masks.

    Args:
        seg: [B, 1, H, W, D] integer tensor with BraTS labels {0,1,2,4}.

    Returns:
        Dict with keys 'WT', 'TC', 'ET', each [B, 1, H, W, D] binary.

    Reference: BraTS 2021 challenge evaluation protocol.
    """
    # seg shape: [B, 1, H, W, D]
    return {
        "WT": ((seg == 1) | (seg == 2) | (seg == 4)).float(),
        "TC": ((seg == 1) | (seg == 4)).float(),
        "ET": (seg == 4).float(),
    }


def compute_metrics_single_subject(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute all metrics for a single subject across all sub-regions.

    Args:
        pred: [H, W, D] integer array with BraTS labels.
        gt:   [H, W, D] integer array with BraTS labels.
        spacing: Voxel spacing in mm for distance-based metrics.

    Returns:
        Dict with keys like 'DSC_WT', 'DSC_TC', 'DSC_ET',
        'HD95_WT', 'HD95_TC', 'HD95_ET', 'SD_WT', 'SD_TC', 'SD_ET'.
    """
    # Convert to tensors: [1, 1, H, W, D]
    pred_t = torch.from_numpy(pred.astype(np.int64)).unsqueeze(0).unsqueeze(0)
    gt_t   = torch.from_numpy(gt.astype(np.int64)).unsqueeze(0).unsqueeze(0)

    pred_regions = brats_to_subregions(pred_t)
    gt_regions   = brats_to_subregions(gt_t)

    results = {}

    for region in ["WT", "TC", "ET"]:
        p = pred_regions[region]  # [1, 1, H, W, D]
        g = gt_regions[region]    # [1, 1, H, W, D]

        # ── DSC ──
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        # DiceMetric expects one-hot or channel-per-class
        # For binary: just use the binary masks directly
        intersection = (p * g).sum()
        union = p.sum() + g.sum()
        dsc = (2.0 * intersection / (union + 1e-8)).item()
        results[f"DSC_{region}"] = dsc

        # ── HD95 ──
        # Only compute if both pred and gt have foreground
        if p.sum() > 0 and g.sum() > 0:
            try:
                hd_metric = HausdorffDistanceMetric(
                    include_background=False,
                    percentile=95,
                    reduction="mean",
                )
                # HausdorffDistanceMetric expects [B, C, H, W, D]
                hd_val = hd_metric(p, g)
                results[f"HD95_{region}"] = hd_val.item()
            except Exception:
                results[f"HD95_{region}"] = float("nan")
        else:
            # If either is empty, HD95 is undefined
            results[f"HD95_{region}"] = float("nan")

        # ── Surface Dice (τ = 1mm) ──
        if p.sum() > 0 and g.sum() > 0:
            try:
                sd_metric = SurfaceDiceMetric(
                    class_thresholds=[1.0],  # τ = 1mm
                    include_background=False,
                    reduction="mean",
                )
                sd_val = sd_metric(p, g)
                results[f"SD_{region}"] = sd_val.item()
            except Exception:
                results[f"SD_{region}"] = float("nan")
        else:
            results[f"SD_{region}"] = 0.0 if g.sum() > 0 else float("nan")

    return results


def load_fold_split(fold: int) -> List[str]:
    """Load validation subject IDs for a given fold.

    Adjust this to match your actual fold split file format.

    Returns:
        List of subject directory names for validation.
    """
    split_file = os.path.join(OUTPUT_DIR, "splits", f"fold{fold}_val.txt")
    if os.path.isfile(split_file):
        with open(split_file) as f:
            return [line.strip() for line in f if line.strip()]

    # Fallback: try to read from a JSON split file
    import json
    json_split = os.path.join(OUTPUT_DIR, "splits", "splits_5fold.json")
    if os.path.isfile(json_split):
        with open(json_split) as f:
            splits = json.load(f)
        return splits[fold]["val"]

    raise FileNotFoundError(
        f"No split file found. Looked for:\n"
        f"  {split_file}\n  {json_split}\n"
        f"Create one or adjust load_fold_split()."
    )


def load_baseline_model(method: str, fold: int, device: torch.device):
    """Load a trained baseline checkpoint.

    Adjust architecture instantiation to match your training code.

    Args:
        method: 'nnunet' or 'swinunetr'.
        fold: CV fold index (0-4).
        device: Target device.

    Returns:
        model in eval mode.
    """
    ckpt_path = CHECKPOINT_PATTERNS[method].format(fold=fold)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading {method} fold {fold}: {ckpt_path}")

    if method == "nnunet":
        # ── Adjust import to match your nnU-Net wrapper ──
        # from baselines.nnunet_wrapper import NNUNetBaseline
        # model = NNUNetBaseline(in_channels=4, out_channels=4)
        raise NotImplementedError(
            "Uncomment and adjust the model instantiation above "
            "to match your nnU-Net architecture class."
        )

    elif method == "swinunetr":
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=4,
            out_channels=4,  # background + 3 tumour classes
            feature_size=48,
            use_checkpoint=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    state = torch.load(ckpt_path, map_location=device)
    # Handle DDP-wrapped checkpoints
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_method(
    method: str,
    fold: int,
    output_dir: str,
    device: torch.device,
) -> None:
    """Run full sub-region evaluation for one method on one fold.

    Saves per-subject results as CSV for statistical testing.

    Args:
        method: 'nnunet' or 'swinunetr'.
        fold: CV fold (0-4).
        output_dir: Directory for CSV output.
        device: CUDA device.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{method}_fold{fold}_subjects.csv")

    # Load model
    model = load_baseline_model(method, fold, device)

    # Load validation subjects
    val_subjects = load_fold_split(fold)
    print(f"  Evaluating {len(val_subjects)} subjects...")

    # Metric column names
    metric_names = []
    for region in ["WT", "TC", "ET"]:
        for metric in ["DSC", "HD95", "SD"]:
            metric_names.append(f"{metric}_{region}")

    # ── Per-subject evaluation loop ──
    all_results = []
    for i, subject_id in enumerate(val_subjects):
        # Load input volume: [4, 128, 128, 128]
        input_path = os.path.join(BRATS2021_ROOT, subject_id,
                                  f"{subject_id}_input.npy")
        gt_path = os.path.join(BRATS2021_ROOT, subject_id,
                               f"{subject_id}_seg.npy")

        if not os.path.isfile(input_path) or not os.path.isfile(gt_path):
            # Try alternative naming conventions
            input_path = os.path.join(BRATS2021_ROOT, subject_id,
                                      "input.npy")
            gt_path = os.path.join(BRATS2021_ROOT, subject_id, "seg.npy")

        if not os.path.isfile(input_path):
            print(f"    ⚠ Skipping {subject_id}: input not found")
            continue

        vol = np.load(input_path).astype(np.float32)    # [4, H, W, D]
        gt  = np.load(gt_path).astype(np.int64)          # [H, W, D]

        # Forward pass: [1, 4, 128, 128, 128] → [1, 4, 128, 128, 128]
        x = torch.from_numpy(vol).unsqueeze(0).to(device)  # [1, 4, H, W, D]
        logits = model(x)  # [1, num_classes, H, W, D]

        # Convert to BraTS integer labels
        pred_classes = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W, D]

        # Map class indices back to BraTS labels if needed
        # Common mapping: 0→0, 1→1(NCR), 2→2(ED), 3→4(ET)
        pred_brats = np.zeros_like(pred_classes)
        pred_brats[pred_classes == 1] = 1   # NCR/NET
        pred_brats[pred_classes == 2] = 2   # ED
        pred_brats[pred_classes == 3] = 4   # ET

        # Compute all metrics
        metrics = compute_metrics_single_subject(pred_brats, gt)
        metrics["subject_id"] = subject_id
        all_results.append(metrics)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(val_subjects)}")

    # ── Save per-subject CSV ──
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id"] + metric_names)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in
                             ["subject_id"] + metric_names})

    # ── Print summary ──
    print(f"\n  ── {method} fold {fold} summary ──")
    for m in metric_names:
        vals = [r[m] for r in all_results
                if m in r and not np.isnan(r.get(m, float("nan")))]
        if vals:
            arr = np.array(vals)
            print(f"    {m:10s}: {arr.mean():.4f} ± {arr.std():.4f}")

    print(f"  ✓ Saved: {csv_path} ({len(all_results)} subjects)")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baselines on all BraTS sub-regions.")
    parser.add_argument("--method", type=str, required=True,
                        choices=["nnunet", "swinunetr"])
    parser.add_argument("--fold", type=int, required=True,
                        choices=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=str, default="./eval_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"{'='*60}")
    print(f"  Evaluating {args.method} | Fold {args.fold}")
    print(f"{'='*60}")

    evaluate_method(args.method, args.fold, args.output_dir, device)


if __name__ == "__main__":
    main()
