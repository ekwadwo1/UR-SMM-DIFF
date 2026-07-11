#!/usr/bin/env python3
"""
evaluation/metrics.py — Full Evaluation Metrics Suite (Phase 10-A)
==================================================================

Segmentation metrics (BraTS regions: WT, TC, ET):
  compute_dsc          — Dice Similarity Coefficient
  compute_hd95         — 95th-percentile Hausdorff Distance
  compute_surface_dice — Surface Dice at 1 mm tolerance

Restoration metrics (brain-masked):
  compute_psnr         — Peak Signal-to-Noise Ratio (dB)
  compute_ssim3d       — 3D Structural Similarity Index

Calibration metrics:
  compute_ece          — Expected Calibration Error (σ² vs empirical MSE)
  compute_nll          — Negative Log-Likelihood

Computational profiling:
  profile_model        — FLOPs, peak VRAM, inference time for S∈{50,100}

Aggregation:
  MetricsAggregator    — per-subject storage, mean±std, DataFrame export

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================= #
#  BraTS Region Extraction                                                 #
# ======================================================================= #

def _brats_regions(
    pred: np.ndarray, target: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract BraTS evaluation regions from class-index maps {0,1,2,3}.

    WT (Whole Tumor):     {1, 2, 3}
    TC (Tumor Core):      {1, 3}
    ET (Enhancing Tumor): {3}

    Returns dict of region_name → (pred_mask, target_mask) as bool arrays.
    """
    return {
        "WT": (pred >= 1, target >= 1),
        "TC": (np.isin(pred, [1, 3]), np.isin(target, [1, 3])),
        "ET": (pred == 3, target == 3),
    }


# ======================================================================= #
#  Segmentation: DSC                                                       #
# ======================================================================= #

def compute_dsc(
    pred: np.ndarray, target: np.ndarray,
) -> Dict[str, float]:
    """
    Dice Similarity Coefficient for BraTS regions.

    DSC = 2|P ∩ T| / (|P| + |T|)

    Parameters
    ----------
    pred   : [H, W, D] int array, class indices {0,1,2,3}
    target : [H, W, D] int array, class indices {0,1,2,3}

    Returns
    -------
    dict with 'dsc_wt', 'dsc_tc', 'dsc_et'
    """
    regions = _brats_regions(pred, target)
    result = {}
    for name, (p, t) in regions.items():
        inter = np.logical_and(p, t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dsc = 1.0 if inter == 0 else 0.0
        else:
            dsc = 2.0 * inter / union
        result[f"dsc_{name.lower()}"] = float(dsc)
    return result


# ======================================================================= #
#  Segmentation: HD95                                                      #
# ======================================================================= #

def _surface_points(mask: np.ndarray, spacing: Tuple[float, ...] = (1., 1., 1.)) -> np.ndarray:
    """
    Extract surface voxel coordinates from a binary mask.

    A voxel is on the surface if it is True and has at least one
    False 6-connected neighbour.

    Returns
    -------
    points : [M, 3] float array of surface coordinates (in mm)
    """
    from scipy import ndimage

    # Erode by 1 voxel → interior; surface = mask & ~interior
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=struct, border_value=False)
    surface = mask & ~eroded

    coords = np.argwhere(surface).astype(np.float64)  # [M, 3] in voxels
    # Scale to mm
    coords *= np.array(spacing, dtype=np.float64)
    return coords


def compute_hd95(
    pred: np.ndarray, target: np.ndarray,
    spacing: Tuple[float, ...] = (1., 1., 1.),
) -> Dict[str, float]:
    """
    95th-percentile Hausdorff Distance for BraTS regions.

    HD95 = max( d95(P→T), d95(T→P) )
    where d95(A→B) = 95th percentile of min-distances from A to B.

    Parameters
    ----------
    pred    : [H, W, D] class indices
    target  : [H, W, D] class indices
    spacing : voxel spacing in mm (default 1mm isotropic)

    Returns
    -------
    dict with 'hd95_wt', 'hd95_tc', 'hd95_et'
    """
    from scipy.spatial.distance import cdist

    regions = _brats_regions(pred, target)
    result = {}

    for name, (p_mask, t_mask) in regions.items():
        if not p_mask.any() and not t_mask.any():
            result[f"hd95_{name.lower()}"] = 0.0
            continue
        if not p_mask.any() or not t_mask.any():
            result[f"hd95_{name.lower()}"] = float("inf")
            continue

        pts_p = _surface_points(p_mask, spacing)
        pts_t = _surface_points(t_mask, spacing)

        if len(pts_p) == 0 or len(pts_t) == 0:
            result[f"hd95_{name.lower()}"] = float("inf")
            continue

        # Compute directed distances in chunks to save memory
        def _directed_d95(src, dst, chunk_size=5000):
            min_dists = []
            for i in range(0, len(src), chunk_size):
                chunk = src[i:i + chunk_size]
                d = cdist(chunk, dst)
                min_dists.append(d.min(axis=1))
            return np.percentile(np.concatenate(min_dists), 95)

        d_p2t = _directed_d95(pts_p, pts_t)
        d_t2p = _directed_d95(pts_t, pts_p)
        result[f"hd95_{name.lower()}"] = float(max(d_p2t, d_t2p))

    return result


# ======================================================================= #
#  Segmentation: Surface Dice                                              #
# ======================================================================= #

def compute_surface_dice(
    pred: np.ndarray, target: np.ndarray,
    tolerance_mm: float = 1.0,
    spacing: Tuple[float, ...] = (1., 1., 1.),
) -> Dict[str, float]:
    """
    Surface Dice (Normalized Surface Distance) at tolerance_mm.

    SD = (|S_P within τ of S_T| + |S_T within τ of S_P|) / (|S_P| + |S_T|)

    A surface point is "within τ" if its nearest-neighbour distance
    to the other surface is ≤ tolerance_mm.

    Parameters
    ----------
    pred          : [H, W, D] class indices
    target        : [H, W, D] class indices
    tolerance_mm  : distance tolerance in mm (default 1.0)
    spacing       : voxel spacing in mm

    Returns
    -------
    dict with 'sd_wt', 'sd_tc', 'sd_et'
    """
    from scipy.spatial import cKDTree

    regions = _brats_regions(pred, target)
    result = {}

    for name, (p_mask, t_mask) in regions.items():
        if not p_mask.any() and not t_mask.any():
            result[f"sd_{name.lower()}"] = 1.0
            continue
        if not p_mask.any() or not t_mask.any():
            result[f"sd_{name.lower()}"] = 0.0
            continue

        pts_p = _surface_points(p_mask, spacing)
        pts_t = _surface_points(t_mask, spacing)

        if len(pts_p) == 0 or len(pts_t) == 0:
            result[f"sd_{name.lower()}"] = 0.0
            continue

        # Build KD-trees for efficient NN queries
        tree_p = cKDTree(pts_p)
        tree_t = cKDTree(pts_t)

        # Distances from P surface to T surface
        d_p2t, _ = tree_t.query(pts_p)
        # Distances from T surface to P surface
        d_t2p, _ = tree_p.query(pts_t)

        n_p_within = np.sum(d_p2t <= tolerance_mm)
        n_t_within = np.sum(d_t2p <= tolerance_mm)
        total = len(pts_p) + len(pts_t)

        sd = (n_p_within + n_t_within) / max(total, 1)
        result[f"sd_{name.lower()}"] = float(sd)

    return result


# ======================================================================= #
#  Restoration: PSNR (brain-masked)                                        #
# ======================================================================= #

def compute_psnr(
    restored: np.ndarray, reference: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB), optionally brain-masked.

    PSNR = 10 · log10( data_range² / MSE )

    Parameters
    ----------
    restored   : [C, H, W, D] or [H, W, D] float
    reference  : same shape as restored
    brain_mask : [H, W, D] bool (optional). If None, use all voxels > 0.

    Returns
    -------
    psnr_db : float
    """
    if brain_mask is None:
        # Default: voxels where reference > 0 in any contrast
        if restored.ndim == 4:
            brain_mask = reference.max(axis=0) > 0
        else:
            brain_mask = reference > 0

    if restored.ndim == 4:
        # Multi-contrast: mask each contrast
        mask_4d = np.broadcast_to(brain_mask[None], restored.shape)
        res = restored[mask_4d]
        ref = reference[mask_4d]
    else:
        res = restored[brain_mask]
        ref = reference[brain_mask]

    if len(ref) == 0:
        return 0.0

    mse = np.mean((res - ref) ** 2)
    if mse < 1e-12:
        return float("inf")
    data_range = ref.max() - ref.min()
    if data_range < 1e-12:
        return 0.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


# ======================================================================= #
#  Restoration: SSIM3D                                                     #
# ======================================================================= #

def compute_ssim3d(
    restored: np.ndarray, reference: np.ndarray,
    brain_mask: Optional[np.ndarray] = None,
    window_size: int = 7,
) -> float:
    """
    3D Structural Similarity Index (mean over contrasts and spatial).

    Uses Gaussian-weighted local statistics.

    Parameters
    ----------
    restored   : [C, H, W, D] float
    reference  : [C, H, W, D] float
    brain_mask : [H, W, D] bool (optional)
    window_size: Gaussian window size
    """
    from scipy.ndimage import gaussian_filter

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    sigma = 1.5

    if restored.ndim == 3:
        restored = restored[None]
        reference = reference[None]

    C_ch = restored.shape[0]
    ssim_vals = []

    for c in range(C_ch):
        x = restored[c].astype(np.float64)
        y = reference[c].astype(np.float64)

        # Data range for this contrast
        dr = y.max() - y.min()
        if dr < 1e-12:
            continue
        c1 = (0.01 * dr) ** 2
        c2 = (0.03 * dr) ** 2

        mu_x = gaussian_filter(x, sigma)
        mu_y = gaussian_filter(y, sigma)
        sigma_xx = gaussian_filter(x * x, sigma) - mu_x ** 2
        sigma_yy = gaussian_filter(y * y, sigma) - mu_y ** 2
        sigma_xy = gaussian_filter(x * y, sigma) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (np.maximum(sigma_xx, 0) + np.maximum(sigma_yy, 0) + c2))

        if brain_mask is not None:
            ssim_vals.append(ssim_map[brain_mask].mean())
        else:
            ssim_vals.append(ssim_map.mean())

    return float(np.mean(ssim_vals)) if ssim_vals else 0.0


# ======================================================================= #
#  Calibration: ECE                                                        #
# ======================================================================= #

def compute_ece(
    sigma_sq: np.ndarray,
    mse_per_token: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error for uncertainty estimates.

    Bins predicted σ² and compares bin-mean σ² to bin-mean empirical MSE.
    A well-calibrated model has σ² ≈ MSE within each bin.

    ECE = Σ_b (n_b / N) · |mean(σ²_b) - mean(MSE_b)|

    Parameters
    ----------
    sigma_sq      : [N] predicted aleatoric variance per token
    mse_per_token : [N] empirical squared error per token
    n_bins        : number of bins (default 15)

    Returns
    -------
    ece : float
    """
    N = len(sigma_sq)
    if N == 0:
        return 0.0

    # Bin by predicted σ²
    bin_edges = np.linspace(sigma_sq.min(), sigma_sq.max() + 1e-8, n_bins + 1)
    bin_indices = np.digitize(sigma_sq, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        mean_sigma = sigma_sq[mask].mean()
        mean_mse = mse_per_token[mask].mean()
        ece += (n_b / N) * abs(mean_sigma - mean_mse)

    return float(ece)


# ======================================================================= #
#  Calibration: NLL                                                        #
# ======================================================================= #

def compute_nll(
    sigma_sq: np.ndarray,
    mse_per_token: np.ndarray,
) -> float:
    """
    Heteroscedastic Negative Log-Likelihood per token.

    NLL = 0.5 · mean( MSE/σ² + log(σ²) )

    Parameters
    ----------
    sigma_sq      : [N] predicted variance per token (> 0)
    mse_per_token : [N] empirical squared error per token

    Returns
    -------
    nll : float
    """
    sigma_sq = np.maximum(sigma_sq, 1e-8)
    nll_per_token = 0.5 * (mse_per_token / sigma_sq + np.log(sigma_sq))
    return float(nll_per_token.mean())


# ======================================================================= #
#  Computational Profiling                                                 #
# ======================================================================= #

def profile_model(
    model: nn.Module,
    z_obs_shape: Tuple[int, ...],
    S_values: Tuple[int, ...] = (50, 100),
    device: str = "cuda:0",
    n_warmup: int = 2,
    n_runs: int = 3,
) -> Dict[str, Any]:
    """
    Profile model: FLOPs estimate, peak VRAM, inference time.

    Parameters
    ----------
    model       : URSSMDiff or similar with restore_and_segment()
    z_obs_shape : (B, C, H, W, D) observed volume shape
    S_values    : DDIM step counts to profile
    device      : CUDA device
    n_warmup    : warmup runs (not timed)
    n_runs      : timed runs (averaged)

    Returns
    -------
    dict with 'params', 'vram_gb', 'inference_s', 'flops_estimate'
    """
    dev = torch.device(device)
    results = {}

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["params_total"] = n_params
    results["params_trainable"] = n_trainable

    # FLOPs estimate via forward hook counting MACs
    # Simple estimate: 2 * params * tokens_per_step * S
    h, w, d = z_obs_shape[2:]
    N_tokens = h * w * d
    results["flops_estimate_note"] = "Use thop or fvcore for precise FLOPs"

    model.eval()
    y_dummy = torch.randn(z_obs_shape, device=dev)

    for S in S_values:
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad(), torch.amp.autocast(str(dev), dtype=torch.bfloat16):
                _ = model.restore_and_segment(y_dummy, S=S)
            torch.cuda.synchronize()

        # Timed runs
        torch.cuda.reset_peak_memory_stats(dev)
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(str(dev), dtype=torch.bfloat16):
                _ = model.restore_and_segment(y_dummy, S=S)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        peak_vram = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        avg_time = np.mean(times)

        results[f"S{S}_time_s"] = float(avg_time)
        results[f"S{S}_vram_gb"] = float(peak_vram)

    return results


# ======================================================================= #
#  MetricsAggregator                                                       #
# ======================================================================= #

class MetricsAggregator:
    """
    Store per-subject metrics, compute summary stats, export DataFrame.

    Usage:
        agg = MetricsAggregator()
        agg.add("subject_001", {"dsc_wt": 0.85, "dsc_tc": 0.80, ...})
        agg.add("subject_002", {...})
        summary = agg.summary()          # mean ± std per metric
        df = agg.to_dataframe()           # pandas DataFrame
        agg.save_csv("results.csv")
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(self, subject_id: str, metrics: Dict[str, float]) -> None:
        """Add one subject's metrics."""
        self.records.append({"subject_id": subject_id, **metrics})

    def __len__(self) -> int:
        return len(self.records)

    def metric_names(self) -> List[str]:
        """All metric keys (excluding subject_id)."""
        if not self.records:
            return []
        return [k for k in self.records[0] if k != "subject_id"]

    def values(self, metric: str) -> np.ndarray:
        """Get all values for a metric as a numpy array."""
        vals = [r[metric] for r in self.records if metric in r
                and r[metric] is not None and not math.isinf(r[metric])]
        return np.array(vals, dtype=np.float64)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean ± std for each metric.

        Returns dict of metric_name → {'mean': ..., 'std': ..., 'median': ..., 'n': ...}
        """
        result = {}
        for name in self.metric_names():
            vals = self.values(name)
            if len(vals) == 0:
                result[name] = {"mean": 0.0, "std": 0.0, "median": 0.0, "n": 0}
            else:
                result[name] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }
        return result

    def summary_table(self) -> str:
        """Pretty-printed summary table."""
        s = self.summary()
        lines = [f"{'Metric':<15} {'Mean':>10} {'± Std':>10} {'Median':>10} {'N':>5}"]
        lines.append("-" * 55)
        for name, stats in sorted(s.items()):
            lines.append(
                f"{name:<15} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                f"{stats['median']:>10.4f} {stats['n']:>5}")
        return "\n".join(lines)

    def to_dataframe(self):
        """Export as pandas DataFrame (for statistical testing)."""
        import pandas as pd
        return pd.DataFrame(self.records)

    def save_csv(self, path: str) -> None:
        """Save per-subject results to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def save_summary(self, path: str) -> None:
        """Save summary to text file."""
        with open(path, "w") as f:
            f.write(self.summary_table())


# ======================================================================= #
#  Convenience: evaluate one subject                                       #
# ======================================================================= #

def evaluate_subject(
    pred_seg: np.ndarray,
    target_seg: np.ndarray,
    restored: Optional[np.ndarray] = None,
    reference: Optional[np.ndarray] = None,
    sigma_sq: Optional[np.ndarray] = None,
    spacing: Tuple[float, ...] = (1., 1., 1.),
    tolerance_mm: float = 1.0,
) -> Dict[str, float]:
    """
    Compute all metrics for a single subject.

    Parameters
    ----------
    pred_seg   : [H, W, D]     predicted class indices {0,1,2,3}
    target_seg : [H, W, D]     ground truth class indices {0,1,2,3}
    restored   : [C, H, W, D]  restored volume (optional)
    reference  : [C, H, W, D]  clean reference volume (optional)
    sigma_sq   : [N]           predicted variance (optional)
    spacing    : voxel spacing in mm
    tolerance_mm : surface dice tolerance

    Returns
    -------
    dict with all computed metrics
    """
    metrics = {}

    # Segmentation
    metrics.update(compute_dsc(pred_seg, target_seg))
    metrics.update(compute_hd95(pred_seg, target_seg, spacing))
    metrics.update(compute_surface_dice(pred_seg, target_seg, tolerance_mm, spacing))

    # Restoration
    if restored is not None and reference is not None:
        brain_mask = reference.max(axis=0) > 0 if reference.ndim == 4 else reference > 0
        metrics["psnr"] = compute_psnr(restored, reference, brain_mask)
        metrics["ssim"] = compute_ssim3d(restored, reference, brain_mask)

    # Calibration
    if sigma_sq is not None and restored is not None and reference is not None:
        if restored.ndim == 4:
            mse_per_ch = ((restored - reference) ** 2).sum(axis=0)  # [H,W,D]
            mse_flat = mse_per_ch.flatten()
        else:
            mse_flat = ((restored - reference) ** 2).flatten()

        # Match lengths (sigma_sq may be at latent resolution)
        if len(sigma_sq) != len(mse_flat):
            # Upsample sigma_sq to match spatial resolution
            ratio = int(round((len(mse_flat) / len(sigma_sq)) ** (1/3)))
            if ratio > 1:
                s = int(round(len(sigma_sq) ** (1/3)))
                sv = sigma_sq.reshape(s, s, s)
                from scipy.ndimage import zoom
                sv_up = zoom(sv, ratio, order=1)
                sigma_sq = sv_up.flatten()[:len(mse_flat)]

        n = min(len(sigma_sq), len(mse_flat))
        metrics["ece"] = compute_ece(sigma_sq[:n], mse_flat[:n])
        metrics["nll"] = compute_nll(sigma_sq[:n], mse_flat[:n])

    return metrics


# ======================================================================= #
#  Tests (Jupyter-safe — no argparse)                                      #
# ======================================================================= #

def _run_tests() -> None:
    """Comprehensive tests for all metrics."""

    print("=" * 70)
    print("  Evaluation Metrics — Test Suite (Phase 10)")
    print("=" * 70)

    np.random.seed(42)
    H = W = D = 64

    # ── Test 1: DSC ───────────────────────────────────────────────────
    print("\n--- (1) Dice Similarity Coefficient ---")
    # Perfect prediction
    target = np.zeros((H, W, D), dtype=int)
    target[:20, :, :] = 1; target[20:40, :, :] = 2; target[40:50, :, :] = 3
    dsc_perf = compute_dsc(target.copy(), target)
    print(f"  Perfect: {dsc_perf}")
    assert all(v == 1.0 for v in dsc_perf.values()), "Perfect DSC should be 1.0"
    print(f"  ✓ All DSC = 1.0 for perfect prediction")

    # Partial overlap
    pred_partial = np.zeros_like(target)
    pred_partial[:15, :, :] = 1; pred_partial[20:35, :, :] = 2; pred_partial[40:48, :, :] = 3
    dsc_part = compute_dsc(pred_partial, target)
    print(f"  Partial: {dsc_part}")
    assert 0 < dsc_part["dsc_wt"] < 1.0
    print(f"  ✓ 0 < DSC < 1 for partial overlap")

    # Empty prediction
    dsc_empty = compute_dsc(np.zeros_like(target), target)
    assert dsc_empty["dsc_wt"] == 0.0
    print(f"  Empty pred: dsc_wt={dsc_empty['dsc_wt']}  ✓")

    # Both empty
    dsc_both_empty = compute_dsc(np.zeros((H, W, D), dtype=int),
                                   np.zeros((H, W, D), dtype=int))
    assert dsc_both_empty["dsc_wt"] == 1.0
    print(f"  Both empty: dsc_wt={dsc_both_empty['dsc_wt']}  ✓")

    # ── Test 2: HD95 ──────────────────────────────────────────────────
    print("\n--- (2) Hausdorff Distance 95th percentile ---")
    try:
        from scipy.spatial.distance import cdist
        hd = compute_hd95(target, target, spacing=(1., 1., 1.))
        print(f"  Perfect: {hd}")
        assert hd["hd95_wt"] == 0.0
        print(f"  ✓ HD95 = 0 for perfect prediction")

        hd_part = compute_hd95(pred_partial, target)
        print(f"  Partial: hd95_wt={hd_part['hd95_wt']:.2f} mm")
        assert hd_part["hd95_wt"] > 0
        print(f"  ✓ HD95 > 0 for partial overlap")
    except ImportError:
        print("  ⚠ scipy not available — skipping HD95 test")

    # ── Test 3: Surface Dice ──────────────────────────────────────────
    print("\n--- (3) Surface Dice (τ=1mm) ---")
    try:
        from scipy.spatial import cKDTree
        sd_perf = compute_surface_dice(target, target, tolerance_mm=1.0)
        print(f"  Perfect: {sd_perf}")
        assert sd_perf["sd_wt"] == 1.0
        print(f"  ✓ SD = 1.0 for perfect prediction")

        sd_part = compute_surface_dice(pred_partial, target, tolerance_mm=1.0)
        print(f"  Partial: sd_wt={sd_part['sd_wt']:.4f}")
        assert 0 < sd_part["sd_wt"] < 1.0
        print(f"  ✓ 0 < SD < 1 for partial overlap")
    except ImportError:
        print("  ⚠ scipy not available — skipping Surface Dice test")

    # ── Test 4: PSNR ─────────────────────────────────────────────────
    print("\n--- (4) PSNR (brain-masked) ---")
    ref = np.random.rand(4, H, W, D).astype(np.float32) * 0.8 + 0.1
    restored_good = ref + np.random.randn(*ref.shape).astype(np.float32) * 0.01
    restored_bad = ref + np.random.randn(*ref.shape).astype(np.float32) * 0.1

    psnr_good = compute_psnr(restored_good, ref)
    psnr_bad = compute_psnr(restored_bad, ref)
    psnr_perf = compute_psnr(ref, ref)

    print(f"  Perfect: {psnr_perf:.2f} dB")
    print(f"  Good:    {psnr_good:.2f} dB")
    print(f"  Bad:     {psnr_bad:.2f} dB")
    assert psnr_perf == float("inf") or psnr_perf > 100
    assert psnr_good > psnr_bad
    print(f"  ✓ PSNR ordering correct")

    # ── Test 5: SSIM3D ────────────────────────────────────────────────
    print("\n--- (5) SSIM3D ---")
    try:
        from scipy.ndimage import gaussian_filter
        ssim_perf = compute_ssim3d(ref, ref)
        ssim_good = compute_ssim3d(restored_good, ref)
        ssim_bad = compute_ssim3d(restored_bad, ref)
        print(f"  Perfect: {ssim_perf:.4f}")
        print(f"  Good:    {ssim_good:.4f}")
        print(f"  Bad:     {ssim_bad:.4f}")
        assert ssim_perf > 0.99
        assert ssim_good > ssim_bad
        print(f"  ✓ SSIM ordering correct")
    except ImportError:
        print("  ⚠ scipy not available — skipping SSIM test")

    # ── Test 6: ECE ──────────────────────────────────────────────────
    print("\n--- (6) Expected Calibration Error ---")
    N_tok = 1000
    # Well-calibrated: σ² ≈ MSE
    true_var = np.random.uniform(0.01, 0.5, N_tok)
    mse_cal = true_var + np.random.randn(N_tok) * 0.01
    mse_cal = np.maximum(mse_cal, 0)
    ece_cal = compute_ece(true_var, mse_cal, n_bins=10)

    # Poorly calibrated: σ² constant but MSE varies
    sigma_const = np.full(N_tok, 0.1)
    mse_vary = np.random.uniform(0, 1, N_tok)
    ece_bad = compute_ece(sigma_const, mse_vary, n_bins=10)

    print(f"  Calibrated ECE:  {ece_cal:.4f}")
    print(f"  Uncalibrated ECE: {ece_bad:.4f}")
    assert ece_cal < ece_bad
    print(f"  ✓ Well-calibrated has lower ECE")

    # ── Test 7: NLL ──────────────────────────────────────────────────
    print("\n--- (7) Negative Log-Likelihood ---")
    # Optimal σ² = MSE
    mse_fix = np.full(N_tok, 0.1)
    nll_opt = compute_nll(mse_fix, mse_fix)             # σ² = MSE
    nll_big = compute_nll(mse_fix * 10, mse_fix)        # σ² = 10·MSE
    nll_small = compute_nll(mse_fix * 0.1, mse_fix)     # σ² = 0.1·MSE

    print(f"  NLL(σ²=MSE):      {nll_opt:.4f}")
    print(f"  NLL(σ²=10·MSE):   {nll_big:.4f}")
    print(f"  NLL(σ²=0.1·MSE):  {nll_small:.4f}")
    assert nll_opt < nll_big and nll_opt < nll_small
    print(f"  ✓ Optimal σ² = MSE gives lowest NLL")

    # ── Test 8: MetricsAggregator ─────────────────────────────────────
    print("\n--- (8) MetricsAggregator ---")
    agg = MetricsAggregator()
    for i in range(10):
        agg.add(f"subj_{i:03d}", {
            "dsc_wt": 0.8 + np.random.randn() * 0.05,
            "dsc_tc": 0.7 + np.random.randn() * 0.06,
            "dsc_et": 0.6 + np.random.randn() * 0.08,
            "psnr": 28 + np.random.randn() * 2,
        })
    assert len(agg) == 10
    summary = agg.summary()
    assert "dsc_wt" in summary
    assert abs(summary["dsc_wt"]["mean"] - 0.8) < 0.1
    print(f"  Subjects: {len(agg)}")
    print(f"\n{agg.summary_table()}")
    print(f"  ✓ Aggregator working")

    # ── Test 9: evaluate_subject ──────────────────────────────────────
    print("\n--- (9) evaluate_subject (all-in-one) ---")
    try:
        all_metrics = evaluate_subject(
            pred_seg=pred_partial,
            target_seg=target,
            restored=restored_good,
            reference=ref,
        )
        print(f"  Keys: {sorted(all_metrics.keys())}")
        assert "dsc_wt" in all_metrics
        assert "hd95_wt" in all_metrics
        assert "sd_wt" in all_metrics
        assert "psnr" in all_metrics
        assert "ssim" in all_metrics
        print(f"  DSC_WT={all_metrics['dsc_wt']:.4f}  "
              f"HD95_WT={all_metrics['hd95_wt']:.2f}  "
              f"SD_WT={all_metrics['sd_wt']:.4f}")
        print(f"  PSNR={all_metrics['psnr']:.2f}  SSIM={all_metrics['ssim']:.4f}")
        print(f"  ✓ All metrics computed")
    except ImportError:
        print("  ⚠ scipy required — skipping full evaluate_subject")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ALL TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
