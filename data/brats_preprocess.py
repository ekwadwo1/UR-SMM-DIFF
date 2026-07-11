#!/usr/bin/env python3
"""
preprocessing/brats_preprocess.py — BraTS Offline Preprocessing  (Phase 2-A)
=============================================================================

Paper Section 4.1: "All BraTS volumes underwent a standardized 3D pipeline:
skull-stripping, affine co-registration to the SRI24 template, resampling to
1.0 mm³ isotropic resolution, foreground-aware cropping to 128³, and
within-mask z-score normalization."

Pipeline steps
--------------
1. Skull-strip   (SimpleITK Otsu + morphology; skip if already stripped)
2. Register      (SimpleITK affine to SRI24 atlas; optional, BraTS is pre-aligned)
3. Resample      (GPU-accelerated torch.nn.functional.interpolate → 1 mm³ iso)
4. Crop / pad    (foreground-aware, centered on tumor/brain centroid → 128³)
5. Normalize     (per-contrast within-mask z-score, clip to [-5, 5])
6. Save          (.npy + metadata JSON)

Hardware:  2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
Speed:     GPU resampling/norm for n_workers=1, or CPU-parallel with n_workers>1.

Usage
-----
    # Full run
    python -m preprocessing.brats_preprocess

    # Dry run (2 subjects, shape checks)
    python -m preprocessing.brats_preprocess --dry-run

    # Create 5-fold splits only
    python -m preprocessing.brats_preprocess --folds-only
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
import traceback
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("brats_preprocess")

# ---------------------------------------------------------------------------
# Path constants  (match context block exactly)
# ---------------------------------------------------------------------------
DRIVE_ROOT: str = (
    "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/UR_SSM_DIFF_DATASETS"
)
BRATS2021_ROOT: str = os.path.join(DRIVE_ROOT, "BraTS2021_Training_Data")
BRATS2023_ADU_ROOT: str = os.path.join(
    DRIVE_ROOT, "BraTS2023-GLI-Challenge-TrainingData"
)
BRATS2023_PED_ROOT: str = os.path.join(
    DRIVE_ROOT, "BraTS2023-PED-TrainingData"
)
OUTPUT_DIR: str = os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs")
PREPROCESSED_DIR: str = os.path.join(OUTPUT_DIR, "preprocessed")

# ---------------------------------------------------------------------------
# BraTS naming conventions
# ---------------------------------------------------------------------------
# BraTS 2021:  {id}_t1.nii.gz  _t1ce  _t2  _flair  _seg
# BraTS 2023:  {id}-t1n.nii.gz -t1c  -t2w  -t2f    -seg
#
# Canonical channel order: [T1, T1ce, T2, FLAIR]  (C=4, paper Section 4.1)

_NAMING = {
    "BraTS2021": {
        "suffixes": ["_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz", "_flair.nii.gz"],
        "seg_suffix": "_seg.nii.gz",
    },
    "BraTS2023": {
        "suffixes": ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz"],
        "seg_suffix": "-seg.nii.gz",
    },
}

# Atlas download URL (MNI152 1 mm as SRI24 stand-in; both are AC-PC aligned)
_ATLAS_URL = (
    "https://templateflow.s3.amazonaws.com/"
    "tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
)


# ===================================================================== #
#  Standalone worker — must be module-level for ProcessPoolExecutor      #
# ===================================================================== #

def _process_subject_cpu(
    subject_dir: str,
    output_root: str,
    naming_key: str,
    skip_existing: bool,
    do_registration: bool,
    atlas_path: Optional[str],
) -> Dict[str, Any]:
    """
    Process one BraTS subject **entirely on CPU** (for multi-worker mode).

    Returns a result dict with keys: subject_id, status, error, elapsed.
    """
    subject_id = os.path.basename(subject_dir)
    t0 = time.time()

    try:
        # --- Check skip ---------------------------------------------------
        img_out = os.path.join(output_root, f"{subject_id}_image.npy")
        lbl_out = os.path.join(output_root, f"{subject_id}_label.npy")
        if skip_existing and os.path.exists(img_out) and os.path.exists(lbl_out):
            return {"subject_id": subject_id, "status": "skipped",
                    "error": None, "elapsed": time.time() - t0}

        # --- Discover files -----------------------------------------------
        paths = _resolve_contrast_paths(subject_dir, naming_key)
        if paths is None:
            return {"subject_id": subject_id, "status": "failed",
                    "error": "Could not resolve contrast files", "elapsed": time.time() - t0}

        # --- Load ----------------------------------------------------------
        images_sitk = {}   # key -> sitk.Image
        for key in ("t1", "t1ce", "t2", "flair"):
            images_sitk[key] = sitk.ReadImage(paths[key], sitk.sitkFloat32)

        label_sitk = None
        if paths.get("seg") and os.path.exists(paths["seg"]):
            label_sitk = sitk.ReadImage(paths["seg"], sitk.sitkUInt8)

        orig_spacing = images_sitk["t1"].GetSpacing()       # (sx, sy, sz)
        orig_size    = images_sitk["t1"].GetSize()           # (nx, ny, nz)
        orig_origin  = images_sitk["t1"].GetOrigin()
        orig_direction = images_sitk["t1"].GetDirection()

        # --- Step 1: Skull stripping (skip if already stripped) ------------
        t1_arr = sitk.GetArrayFromImage(images_sitk["t1"])   # (z, y, x)
        nonzero_ratio = np.count_nonzero(t1_arr) / t1_arr.size
        brain_mask_sitk = None

        if nonzero_ratio > 0.70:
            # Probably NOT skull-stripped → apply Otsu
            brain_mask_sitk = _skull_strip_otsu(images_sitk["t1"])
            for key in images_sitk:
                images_sitk[key] = sitk.Mask(images_sitk[key], brain_mask_sitk)
        else:
            # Already skull-stripped — derive mask from non-zero voxels
            mask_arr = (t1_arr > 0).astype(np.uint8)
            brain_mask_sitk = sitk.GetImageFromArray(mask_arr)
            brain_mask_sitk.CopyInformation(images_sitk["t1"])

        # --- Step 2: Registration to atlas (optional) ----------------------
        if do_registration and atlas_path and os.path.exists(atlas_path):
            atlas_sitk = sitk.ReadImage(atlas_path, sitk.sitkFloat32)
            transform = _register_affine(images_sitk["t1"], atlas_sitk)
            # Apply transform to all contrasts + label + mask
            for key in images_sitk:
                images_sitk[key] = sitk.Resample(
                    images_sitk[key], atlas_sitk, transform,
                    sitk.sitkBSpline, 0.0, sitk.sitkFloat32,
                )
            if label_sitk is not None:
                label_sitk = sitk.Resample(
                    label_sitk, atlas_sitk, transform,
                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
                )
            brain_mask_sitk = sitk.Resample(
                brain_mask_sitk, atlas_sitk, transform,
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
            )

        # --- Step 3: Resample to 1.0 mm³ isotropic (CPU fallback) ---------
        images_np = {}
        for key in ("t1", "t1ce", "t2", "flair"):
            images_np[key] = _resample_isotropic_cpu(images_sitk[key], is_label=False)
        mask_np = _resample_isotropic_cpu(brain_mask_sitk, is_label=True)
        label_np = None
        if label_sitk is not None:
            label_np = _resample_isotropic_cpu(label_sitk, is_label=True)

        # --- Stack channels → [4, D, H, W] --------------------------------
        stack = np.stack(
            [images_np["t1"], images_np["t1ce"], images_np["t2"], images_np["flair"]],
            axis=0,
        ).astype(np.float32)   # [4, D, H, W]
        mask_np = mask_np.astype(np.uint8)
        if label_np is None:
            label_np = np.zeros(stack.shape[1:], dtype=np.uint8)

        # --- Step 4: Crop / pad to 128³ ------------------------------------
        stack, label_np, mask_np, crop_coords = _crop_or_pad_128(
            stack, label_np, mask_np
        )

        # --- Step 5: Within-mask z-score per contrast ----------------------
        norm_stats = {}
        for c in range(4):
            ch = stack[c]
            m = mask_np > 0
            if m.sum() > 100:
                mu = float(ch[m].mean())
                sd = float(ch[m].std()) + 1e-8
            else:
                mu, sd = 0.0, 1.0
            stack[c] = np.clip((ch - mu) / sd, -5.0, 5.0)
            # Zero outside brain
            stack[c][~m] = 0.0
            norm_stats[f"ch{c}_mean"] = mu
            norm_stats[f"ch{c}_std"]  = sd

        # --- Step 6: Save --------------------------------------------------
        os.makedirs(output_root, exist_ok=True)
        np.save(img_out, stack.astype(np.float32))           # [4,128,128,128]
        np.save(lbl_out, label_np.astype(np.uint8))          # [128,128,128]

        meta = {
            "subject_id": subject_id,
            "original_spacing": list(orig_spacing),
            "original_size":    list(orig_size),
            "original_origin":  list(orig_origin),
            "crop_coords":      crop_coords,
            "norm_stats":       norm_stats,
            "naming":           naming_key,
            "skull_stripped":    nonzero_ratio <= 0.70,
            "registered":       do_registration and atlas_path is not None,
        }
        meta_out = os.path.join(output_root, f"{subject_id}_meta.json")
        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2)

        return {"subject_id": subject_id, "status": "ok",
                "error": None, "elapsed": time.time() - t0}

    except Exception as e:
        tb = traceback.format_exc()
        return {"subject_id": subject_id, "status": "failed",
                "error": f"{e}\n{tb}", "elapsed": time.time() - t0}


# ===================================================================== #
#  Module-level helpers  (pickle-able for ProcessPoolExecutor)           #
# ===================================================================== #

def _resolve_contrast_paths(
    subject_dir: str, naming_key: str,
) -> Optional[Dict[str, str]]:
    """Resolve the four contrast + seg paths for a subject directory."""
    subject_id = os.path.basename(subject_dir)
    # Filter out macOS resource fork files (._*) and other hidden files
    files = [f for f in os.listdir(subject_dir)
             if not f.startswith("._") and not f.startswith(".")]
    files_lower = {f.lower(): f for f in files}

    # Try the specified naming first, then the other
    for nk in [naming_key, "BraTS2021", "BraTS2023"]:
        if nk not in _NAMING:
            continue
        info = _NAMING[nk]
        found = {}
        ok = True
        canon = ["t1", "t1ce", "t2", "flair"]
        for i, suffix in enumerate(info["suffixes"]):
            matches = [f for f in files if f.lower().endswith(suffix.lower())]
            if matches:
                found[canon[i]] = os.path.join(subject_dir, matches[0])
            else:
                ok = False
                break
        if ok:
            # seg is optional
            seg_matches = [
                f for f in files if f.lower().endswith(info["seg_suffix"].lower())
            ]
            found["seg"] = (
                os.path.join(subject_dir, seg_matches[0]) if seg_matches else None
            )
            return found

    # Last resort: glob for common patterns
    result: Dict[str, Optional[str]] = {}
    patterns = {
        "t1":   ["*t1.nii*", "*t1n.nii*", "*T1.nii*"],
        "t1ce": ["*t1ce.nii*", "*t1c.nii*", "*T1ce.nii*", "*T1c.nii*"],
        "t2":   ["*t2.nii*", "*t2w.nii*", "*T2.nii*", "*T2w.nii*"],
        "flair": ["*flair.nii*", "*t2f.nii*", "*FLAIR.nii*", "*T2f.nii*"],
        "seg":  ["*seg.nii*", "*Seg.nii*"],
    }
    for key, pats in patterns.items():
        found_path = None
        for pat in pats:
            hits = glob.glob(os.path.join(subject_dir, pat))
            # Filter macOS resource forks (._*) from glob results
            hits = [h for h in hits if not os.path.basename(h).startswith("._")]
            # For t1, make sure we don't accidentally match t1ce
            if key == "t1" and hits:
                hits = [h for h in hits if "ce" not in h.lower() and "t1c" not in h.lower()]
            if hits:
                found_path = hits[0]
                break
        result[key] = found_path

    # Verify four contrasts found
    for key in ("t1", "t1ce", "t2", "flair"):
        if result.get(key) is None:
            logger.warning(f"  Missing {key} in {subject_dir}")
            return None
    return result


def _skull_strip_otsu(image_sitk: sitk.Image) -> sitk.Image:
    """
    Step 1 — Skull stripping via Otsu threshold + morphological ops.

    Reference: Paper Section 4.1 — "skull-stripping".
    We use SimpleITK (NOT antspyx) to avoid CUDA conflicts with mamba-ssm.
    """
    # Otsu threshold
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask = otsu_filter.Execute(image_sitk)

    # Morphological closing to fill holes
    closer = sitk.BinaryMorphologicalClosingImageFilter()
    closer.SetKernelRadius(3)
    closer.SetKernelType(sitk.sitkBall)
    mask = closer.Execute(mask)

    # Fill internal holes
    filler = sitk.BinaryFillholeImageFilter()
    filler.SetForegroundValue(1)
    mask = filler.Execute(mask)

    # Keep largest connected component
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled = cc_filter.Execute(sitk.Cast(mask, sitk.sitkUInt16))
    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(100)
    relabeled = relabel.Execute(labeled)
    # Largest component = label 1
    mask = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1)

    # Slight dilation for safety margin
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius(2)
    dilate.SetKernelType(sitk.sitkBall)
    mask = dilate.Execute(mask)

    return sitk.Cast(mask, sitk.sitkUInt8)


def _register_affine(
    moving_sitk: sitk.Image,
    atlas_sitk: sitk.Image,
) -> sitk.Transform:
    """
    Step 2 — Affine co-registration to atlas using SimpleITK.

    Reference: Paper Section 4.1 — "affine co-registration to the SRI24 template".
    """
    registration = sitk.ImageRegistrationMethod()

    # Metric: Mattes Mutual Information
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.10)

    registration.SetInterpolator(sitk.sitkLinear)

    # Initial alignment by geometry centres
    initial_transform = sitk.CenteredTransformInitializer(
        atlas_sitk, moving_sitk,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration.Execute(atlas_sitk, moving_sitk)
    return final_transform


def _resample_isotropic_cpu(
    image_sitk: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_label: bool = False,
) -> np.ndarray:
    """
    Step 3 (CPU path) — Resample to isotropic resolution using SimpleITK.

    Reference: Paper Section 4.1 — "resampling to 1.0 mm³ isotropic resolution".
    Uses BSpline for images, NearestNeighbor for labels.
    """
    orig_spacing = image_sitk.GetSpacing()
    orig_size    = image_sitk.GetSize()

    # Check if already at target spacing (within tolerance)
    if all(abs(os - ts) < 0.01 for os, ts in zip(orig_spacing, target_spacing)):
        return sitk.GetArrayFromImage(image_sitk)

    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(orig_size, orig_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetDefaultPixelValue(0.0)

    resampled = resampler.Execute(image_sitk)
    return sitk.GetArrayFromImage(resampled)


def _crop_or_pad_128(
    images: np.ndarray,
    label: np.ndarray,
    mask: np.ndarray,
    target: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Step 4 — Foreground-aware crop/pad to 128³.

    Reference: Paper Section 4.1 — "foreground-aware cropping to 128³".

    Strategy:
    - If brain fits inside 128³, pad symmetrically with zeros.
    - If brain is larger, centre crop on tumour centroid (if labels present)
      or brain centroid.
    """
    # images: [4, D, H, W],  label: [D, H, W],  mask: [D, H, W]
    _, D, H, W = images.shape

    # ---- Determine centroid -----------------------------------------------
    if label is not None and np.any(label > 0):
        nz = np.argwhere(label > 0)          # (N, 3)
        centroid = nz.mean(axis=0).astype(int)  # tumour centroid
    else:
        nz = np.argwhere(mask > 0)
        if len(nz) > 0:
            centroid = nz.mean(axis=0).astype(int)  # brain centroid
        else:
            centroid = np.array([D // 2, H // 2, W // 2])

    # ---- Compute crop window per axis ------------------------------------
    crop_slices = []
    pad_before  = []
    pad_after   = []
    crop_coords = []

    for ax, (dim_size, c) in enumerate(zip([D, H, W], centroid)):
        if dim_size <= target:
            # Pad — use the whole dimension
            start, end = 0, dim_size
            pb = (target - dim_size) // 2
            pa = target - dim_size - pb
        else:
            # Crop — centre on centroid
            half = target // 2
            start = max(0, c - half)
            end = start + target
            if end > dim_size:
                end = dim_size
                start = end - target
            pb, pa = 0, 0

        crop_slices.append(slice(start, end))
        pad_before.append(pb)
        pad_after.append(pa)
        crop_coords.append([int(start), int(end), int(pb), int(pa)])

    # ---- Apply crop -------------------------------------------------------
    ds, hs, ws = crop_slices
    images = images[:, ds, hs, ws]
    label  = label[ds, hs, ws]
    mask   = mask[ds, hs, ws]

    # ---- Apply pad --------------------------------------------------------
    if any(p > 0 for p in pad_before + pad_after):
        images = np.pad(
            images,
            [(0, 0),
             (pad_before[0], pad_after[0]),
             (pad_before[1], pad_after[1]),
             (pad_before[2], pad_after[2])],
            mode="constant", constant_values=0,
        )
        label = np.pad(
            label,
            [(pad_before[0], pad_after[0]),
             (pad_before[1], pad_after[1]),
             (pad_before[2], pad_after[2])],
            mode="constant", constant_values=0,
        )
        mask = np.pad(
            mask,
            [(pad_before[0], pad_after[0]),
             (pad_before[1], pad_after[1]),
             (pad_before[2], pad_after[2])],
            mode="constant", constant_values=0,
        )

    assert images.shape == (4, target, target, target), \
        f"Image shape {images.shape} != (4, {target}, {target}, {target})"
    assert label.shape == (target, target, target), \
        f"Label shape {label.shape} != ({target}, {target}, {target})"

    return images, label, mask, crop_coords


# ===================================================================== #
#  Main class                                                            #
# ===================================================================== #

class BraTSPreprocessor:
    """
    Offline BraTS preprocessing pipeline (Paper Section 4.1).

    Parameters
    ----------
    raw_root : str
        Path to the raw BraTS dataset root (e.g. BraTS2021_Training_Data).
    output_root : str
        Directory where preprocessed .npy files will be saved.
    dataset : str
        One of 'BraTS2021', 'BraTS2023_Adult', 'BraTS2023_Pediatric'.
    n_workers : int
        Number of parallel CPU workers.  Set 1 for GPU-accelerated mode.
    skip_existing : bool
        Skip subjects whose output files already exist.
    device : str
        CUDA device for GPU-accelerated resampling (n_workers=1 only).
    atlas_path : str or None
        Path to SRI24 / MNI152 atlas NIfTI.  If None, registration is skipped.
    do_registration : bool
        Whether to perform affine registration to atlas.
        Default False because BraTS data ships pre-registered.
    """

    def __init__(
        self,
        raw_root: str,
        output_root: str,
        dataset: str = "BraTS2021",
        n_workers: int = 8,
        skip_existing: bool = True,
        device: str = "cuda:0",
        atlas_path: Optional[str] = None,
        do_registration: bool = False,
    ) -> None:
        self.raw_root = raw_root
        self.output_root = output_root
        self.dataset = dataset
        self.n_workers = n_workers
        self.skip_existing = skip_existing
        self.device = device
        self.atlas_path = atlas_path
        self.do_registration = do_registration

        # Resolve naming convention
        if "2023" in dataset:
            self.naming_key = "BraTS2023"
        else:
            self.naming_key = "BraTS2021"

        os.makedirs(output_root, exist_ok=True)

        # Resolve atlas if registration requested
        if self.do_registration and self.atlas_path is None:
            self.atlas_path = self._ensure_atlas()

    # ------------------------------------------------------------------ #
    #  Atlas management                                                   #
    # ------------------------------------------------------------------ #

    def _ensure_atlas(self) -> Optional[str]:
        """Locate or download an atlas template for registration."""
        search_paths = [
            os.path.join(self.output_root, "atlas", "atlas_T1.nii.gz"),
            os.path.join(os.path.expanduser("~"), "templates", "sri24_T1.nii.gz"),
            "/usr/share/fsl/data/standard/MNI152_T1_1mm.nii.gz",
        ]
        for p in search_paths:
            if os.path.isfile(p):
                logger.info(f"Atlas found at {p}")
                return p

        # Attempt download
        atlas_dir = os.path.join(self.output_root, "atlas")
        os.makedirs(atlas_dir, exist_ok=True)
        atlas_path = os.path.join(atlas_dir, "atlas_T1.nii.gz")
        try:
            logger.info(f"Downloading atlas from {_ATLAS_URL} …")
            urllib.request.urlretrieve(_ATLAS_URL, atlas_path)
            logger.info(f"Atlas saved to {atlas_path}")
            return atlas_path
        except Exception as e:
            logger.warning(f"Atlas download failed ({e}). Registration will be skipped.")
            return None

    # ------------------------------------------------------------------ #
    #  Subject discovery                                                  #
    # ------------------------------------------------------------------ #

    def _discover_subjects(self) -> List[str]:
        """Return sorted list of subject directory paths."""
        if not os.path.isdir(self.raw_root):
            raise FileNotFoundError(f"Dataset root not found: {self.raw_root}")

        subjects = sorted([
            os.path.join(self.raw_root, d)
            for d in os.listdir(self.raw_root)
            if os.path.isdir(os.path.join(self.raw_root, d))
            and not d.startswith(".")
        ])
        logger.info(f"Discovered {len(subjects)} subjects in {self.raw_root}")
        return subjects

    # ------------------------------------------------------------------ #
    #  GPU-accelerated single-subject pipeline                            #
    # ------------------------------------------------------------------ #

    def _process_single_gpu(self, subject_dir: str) -> Dict[str, Any]:
        """
        Process one subject with GPU-accelerated resampling and normalisation.

        Used when n_workers ≤ 1 for maximum per-subject speed.
        """
        subject_id = os.path.basename(subject_dir)
        t0 = time.time()

        try:
            img_out = os.path.join(self.output_root, f"{subject_id}_image.npy")
            lbl_out = os.path.join(self.output_root, f"{subject_id}_label.npy")
            if self.skip_existing and os.path.exists(img_out) and os.path.exists(lbl_out):
                return {"subject_id": subject_id, "status": "skipped",
                        "error": None, "elapsed": time.time() - t0}

            # --- Discover files ---
            paths = _resolve_contrast_paths(subject_dir, self.naming_key)
            if paths is None:
                return {"subject_id": subject_id, "status": "failed",
                        "error": "Missing contrast files", "elapsed": time.time() - t0}

            # --- Load with SimpleITK ---
            images_sitk = {}
            for key in ("t1", "t1ce", "t2", "flair"):
                images_sitk[key] = sitk.ReadImage(paths[key], sitk.sitkFloat32)

            label_sitk = None
            if paths.get("seg") and os.path.exists(paths["seg"]):
                label_sitk = sitk.ReadImage(paths["seg"], sitk.sitkUInt8)

            orig_spacing = images_sitk["t1"].GetSpacing()

            # --- Step 1: skull strip (detect if needed) ---
            t1_arr = sitk.GetArrayFromImage(images_sitk["t1"])
            nonzero_ratio = np.count_nonzero(t1_arr) / max(t1_arr.size, 1)

            if nonzero_ratio > 0.70:
                brain_mask_sitk = _skull_strip_otsu(images_sitk["t1"])
                for key in images_sitk:
                    images_sitk[key] = sitk.Mask(images_sitk[key], brain_mask_sitk)
            else:
                mask_arr = (t1_arr > 0).astype(np.uint8)
                brain_mask_sitk = sitk.GetImageFromArray(mask_arr)
                brain_mask_sitk.CopyInformation(images_sitk["t1"])

            # --- Step 2: optional registration ---
            if (self.do_registration and self.atlas_path
                    and os.path.exists(self.atlas_path)):
                atlas_sitk = sitk.ReadImage(self.atlas_path, sitk.sitkFloat32)
                xfm = _register_affine(images_sitk["t1"], atlas_sitk)
                for key in images_sitk:
                    images_sitk[key] = sitk.Resample(
                        images_sitk[key], atlas_sitk, xfm,
                        sitk.sitkBSpline, 0.0, sitk.sitkFloat32)
                if label_sitk is not None:
                    label_sitk = sitk.Resample(
                        label_sitk, atlas_sitk, xfm,
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                brain_mask_sitk = sitk.Resample(
                    brain_mask_sitk, atlas_sitk, xfm,
                    sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

            # --- Step 3: GPU-accelerated isotropic resampling ---
            arrays = {}
            for key in ("t1", "t1ce", "t2", "flair"):
                arr = sitk.GetArrayFromImage(images_sitk[key])  # (z,y,x)
                sp  = images_sitk[key].GetSpacing()             # (sx,sy,sz)
                arrays[key] = self._resample_gpu(arr, sp, is_label=False)

            mask_arr = sitk.GetArrayFromImage(brain_mask_sitk)
            mask_sp  = brain_mask_sitk.GetSpacing()
            mask_np  = self._resample_gpu(mask_arr, mask_sp, is_label=True)

            label_np = None
            if label_sitk is not None:
                lbl_arr = sitk.GetArrayFromImage(label_sitk)
                lbl_sp  = label_sitk.GetSpacing()
                label_np = self._resample_gpu(lbl_arr, lbl_sp, is_label=True)

            # --- Stack [4, D, H, W] ---
            stack = np.stack(
                [arrays["t1"], arrays["t1ce"], arrays["t2"], arrays["flair"]],
                axis=0,
            ).astype(np.float32)
            mask_np = mask_np.astype(np.uint8)
            if label_np is None:
                label_np = np.zeros(stack.shape[1:], dtype=np.uint8)

            # --- Step 4: crop / pad 128³ ---
            stack, label_np, mask_np, crop_coords = _crop_or_pad_128(
                stack, label_np, mask_np
            )

            # --- Step 5: GPU-accelerated z-score normalisation ---
            stack, norm_stats = self._normalize_gpu(stack, mask_np)

            # --- Step 6: save ---
            np.save(img_out, stack.astype(np.float32))
            np.save(lbl_out, label_np.astype(np.uint8))

            meta = {
                "subject_id": subject_id,
                "original_spacing": list(orig_spacing),
                "crop_coords": crop_coords,
                "norm_stats": norm_stats,
                "naming": self.naming_key,
                "gpu_processed": True,
            }
            with open(os.path.join(self.output_root, f"{subject_id}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            return {"subject_id": subject_id, "status": "ok",
                    "error": None, "elapsed": time.time() - t0}

        except Exception as e:
            return {"subject_id": subject_id, "status": "failed",
                    "error": str(e), "elapsed": time.time() - t0}

    def _resample_gpu(
        self,
        arr: np.ndarray,
        spacing: Tuple[float, ...],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        is_label: bool = False,
    ) -> np.ndarray:
        """
        Step 3 (GPU path) — Resample to isotropic via torch.nn.functional.interpolate.

        Reference: Paper Section 4.1 — "resampling to 1.0 mm³ isotropic resolution".

        Parameters
        ----------
        arr : ndarray  (D, H, W)  from sitk.GetArrayFromImage
        spacing : tuple  (sx, sy, sz)  from SimpleITK  (x, y, z order)
        """
        # SimpleITK spacing = (sx, sy, sz);  array dims = (z, y, x)
        # scale_factor for F.interpolate dim order = (D, H, W) = (z, y, x)
        sx, sy, sz = spacing
        tx, ty, tz = target_spacing

        # If already isotropic, skip
        if all(abs(s - t) < 0.01 for s, t in zip(spacing, target_spacing)):
            return arr

        scale_z = sz / tz
        scale_y = sy / ty
        scale_x = sx / tx

        tensor = torch.from_numpy(arr.astype(np.float32))          # (D,H,W)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)   # (1,1,D,H,W)

        if is_label:
            # Nearest-neighbour for labels: use float for interpolate, then round
            resampled = F.interpolate(
                tensor, scale_factor=(scale_z, scale_y, scale_x),
                mode="nearest",
            )
            result = resampled.squeeze().cpu().numpy().astype(np.uint8)
        else:
            resampled = F.interpolate(
                tensor, scale_factor=(scale_z, scale_y, scale_x),
                mode="trilinear", align_corners=False,
            )
            result = resampled.squeeze().cpu().numpy()

        return result

    def _normalize_gpu(
        self, images: np.ndarray, mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Step 5 — Per-contrast within-mask z-score normalisation on GPU.

        Reference: Paper Section 4.1 — "within-mask z-score normalization".
        Clip to [-5, 5] after normalisation.
        """
        # images: [4,128,128,128]   mask: [128,128,128]
        device = self.device
        img_t = torch.from_numpy(images).to(device)                    # (4,D,H,W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)  # (D,H,W)
        mask_bool = mask_t > 0.5                                       # bool

        stats: Dict[str, float] = {}
        for c in range(4):
            ch = img_t[c]                   # (D, H, W)
            vals = ch[mask_bool]            # (M,)
            if vals.numel() > 100:
                mu = vals.mean()
                sd = vals.std() + 1e-8
            else:
                mu = torch.tensor(0.0, device=device)
                sd = torch.tensor(1.0, device=device)
            img_t[c] = ((ch - mu) / sd).clamp(-5.0, 5.0)
            img_t[c][~mask_bool] = 0.0
            stats[f"ch{c}_mean"] = mu.item()
            stats[f"ch{c}_std"]  = sd.item()

        return img_t.cpu().numpy(), stats

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def run(self, subject_list: Optional[List[str]] = None) -> None:
        """
        Process all (or selected) subjects.

        Parameters
        ----------
        subject_list : list of str, optional
            If provided, only these subject directory paths are processed.
            If None, discover all subjects under raw_root.
        """
        if subject_list is None:
            subject_list = self._discover_subjects()

        logger.info(
            f"Processing {len(subject_list)} subjects  |  dataset={self.dataset}"
            f"  |  workers={self.n_workers}  |  skip_existing={self.skip_existing}"
        )

        results: List[Dict] = []
        t_start = time.time()

        if self.n_workers <= 1:
            # ── Sequential with GPU acceleration ──────────────────────────
            logger.info("Mode: sequential + GPU-accelerated resampling/normalisation")
            for subj_dir in tqdm(subject_list, desc="Preprocessing"):
                r = self._process_single_gpu(subj_dir)
                results.append(r)
                if r["status"] == "failed":
                    logger.warning(f"  FAILED {r['subject_id']}: {r['error']}")
        else:
            # ── Parallel CPU workers ──────────────────────────────────────
            logger.info(f"Mode: {self.n_workers} parallel CPU workers")
            args_list = [
                (subj_dir, self.output_root, self.naming_key,
                 self.skip_existing, self.do_registration, self.atlas_path)
                for subj_dir in subject_list
            ]

            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {
                    pool.submit(_process_subject_cpu, *args): args[0]
                    for args in args_list
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Preprocessing"
                ):
                    r = future.result()
                    results.append(r)
                    if r["status"] == "failed":
                        logger.warning(f"  FAILED {r['subject_id']}: {r['error']}")

        # ── Summary ───────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        ok      = sum(1 for r in results if r["status"] == "ok")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed  = sum(1 for r in results if r["status"] == "failed")

        logger.info(
            f"\nDone in {elapsed:.1f}s  |  OK={ok}  Skipped={skipped}  Failed={failed}"
        )
        if failed > 0:
            logger.warning("Failed subjects:")
            for r in results:
                if r["status"] == "failed":
                    logger.warning(f"  {r['subject_id']}: {r['error'][:120]}")

        # Save run report
        report_path = os.path.join(self.output_root, "preprocess_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "dataset": self.dataset,
                "total": len(results),
                "ok": ok, "skipped": skipped, "failed": failed,
                "elapsed_sec": elapsed,
                "subjects": [
                    {k: v for k, v in r.items() if k != "error"}
                    for r in results
                ],
            }, f, indent=2)
        logger.info(f"Report saved to {report_path}")

    # ------------------------------------------------------------------ #
    #  Verification                                                       #
    # ------------------------------------------------------------------ #

    def verify(self, subject_id: str) -> bool:
        """
        Verify a preprocessed subject: shapes, dtypes, NaN/Inf, label dist.

        Returns True if all checks pass.
        """
        img_path = os.path.join(self.output_root, f"{subject_id}_image.npy")
        lbl_path = os.path.join(self.output_root, f"{subject_id}_label.npy")
        meta_path = os.path.join(self.output_root, f"{subject_id}_meta.json")

        ok = True

        # --- File existence ---
        for path, name in [(img_path, "image"), (lbl_path, "label"), (meta_path, "meta")]:
            if not os.path.exists(path):
                logger.error(f"  [{subject_id}] {name} file missing: {path}")
                ok = False
        if not ok:
            return False

        # --- Load ---
        img = np.load(img_path)
        lbl = np.load(lbl_path)

        # --- Shape ---
        if img.shape != (4, 128, 128, 128):
            logger.error(f"  [{subject_id}] Image shape {img.shape} != (4,128,128,128)")
            ok = False
        if lbl.shape != (128, 128, 128):
            logger.error(f"  [{subject_id}] Label shape {lbl.shape} != (128,128,128)")
            ok = False

        # --- Dtype ---
        if img.dtype != np.float32:
            logger.error(f"  [{subject_id}] Image dtype {img.dtype} != float32")
            ok = False
        if lbl.dtype != np.uint8:
            logger.error(f"  [{subject_id}] Label dtype {lbl.dtype} != uint8")
            ok = False

        # --- NaN / Inf ---
        if np.any(np.isnan(img)):
            logger.error(f"  [{subject_id}] Image contains NaN")
            ok = False
        if np.any(np.isinf(img)):
            logger.error(f"  [{subject_id}] Image contains Inf")
            ok = False

        # --- Value range (should be clipped to [-5, 5]) ---
        if img.min() < -5.01 or img.max() > 5.01:
            logger.warning(
                f"  [{subject_id}] Image range [{img.min():.2f}, {img.max():.2f}] "
                f"outside expected [-5, 5]"
            )

        # --- Label values ---
        unique = set(np.unique(lbl).tolist())
        valid_labels = {0, 1, 2, 4}
        invalid = unique - valid_labels
        if invalid:
            logger.warning(
                f"  [{subject_id}] Unexpected label values: {invalid}. "
                f"Expected subset of {valid_labels}"
            )
            # BraTS 2023 may use different label encoding — warn, don't fail
            if invalid - {3}:
                ok = False

        # --- Label distribution ---
        counts = Counter(lbl.flat)
        total = lbl.size
        dist_str = "  ".join(
            f"L{k}: {v} ({100*v/total:.1f}%)" for k, v in sorted(counts.items())
        )
        logger.info(f"  [{subject_id}] Labels: {dist_str}")

        if ok:
            logger.info(f"  [{subject_id}] Verification PASSED ✓")
        else:
            logger.error(f"  [{subject_id}] Verification FAILED ✗")

        return ok

    # ------------------------------------------------------------------ #
    #  Five-fold patient-level splits                                     #
    # ------------------------------------------------------------------ #

    def create_folds(self, n_folds: int = 5, seed: int = 42) -> None:
        """
        Create stratified k-fold patient-level splits for BraTS 2021.

        Saves fold_0.json … fold_{n_folds-1}.json to output_root.
        Each JSON: {'train': [subject_id, …], 'val': [subject_id, …]}

        Uses sklearn.model_selection.KFold with shuffle=True.
        """
        from sklearn.model_selection import KFold

        subjects = self._discover_subjects()
        subject_ids = [os.path.basename(s) for s in subjects]

        logger.info(
            f"Creating {n_folds}-fold CV splits for {len(subject_ids)} subjects "
            f"(seed={seed})"
        )

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(subject_ids)):
            fold = {
                "train": [subject_ids[i] for i in train_idx],
                "val":   [subject_ids[i] for i in val_idx],
            }
            fold_path = os.path.join(self.output_root, f"fold_{fold_idx}.json")
            with open(fold_path, "w") as f:
                json.dump(fold, f, indent=2)
            logger.info(
                f"  fold_{fold_idx}: train={len(fold['train'])}, "
                f"val={len(fold['val'])}  →  {fold_path}"
            )

        logger.info("Fold creation complete.")


# ===================================================================== #
#  __main__ — dry-run mode                                               #
# ===================================================================== #

def _print_label_histogram(lbl: np.ndarray, subject_id: str) -> None:
    """Print a text histogram of label distributions."""
    counts = Counter(lbl.flat)
    total = lbl.size
    print(f"\n  Label histogram for {subject_id}:")
    bar_max = 40
    max_count = max(counts.values()) if counts else 1
    for label_val in sorted(counts.keys()):
        cnt = counts[label_val]
        pct = 100.0 * cnt / total
        bar_len = int(bar_max * cnt / max_count)
        bar = "█" * bar_len
        name_map = {0: "BG", 1: "NCR/NET", 2: "ED", 4: "ET"}
        name = name_map.get(label_val, f"L{label_val}")
        print(f"    {name:>8s} (={label_val}): {bar}  {cnt:>8d} ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BraTS Offline Preprocessing (Phase 2-A, Paper Section 4.1)",
    )
    parser.add_argument(
        "--dataset", type=str, default="BraTS2021",
        choices=["BraTS2021", "BraTS2023_Adult", "BraTS2023_Pediatric"],
        help="Which BraTS dataset to preprocess.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 2 subjects and print diagnostics.")
    parser.add_argument("--folds-only", action="store_true",
                        help="Only create 5-fold splits (skip preprocessing).")
    parser.add_argument("--n-workers", type=int, default=8,
                        help="Parallel CPU workers (set 1 for GPU mode).")
    parser.add_argument("--register", action="store_true",
                        help="Enable affine registration to atlas.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device for GPU mode.")
    parser.add_argument("--verify", type=str, default=None, metavar="SUBJECT_ID",
                        help="Verify a single preprocessed subject and exit.")
    # --- Jupyter-safe argparse ------------------------------------------------
    _in_jupyter = any("ipykernel" in a or "kernel" in a for a in sys.argv)
    if _in_jupyter:
        args = argparse.Namespace(
            dataset="BraTS2021", dry_run=True, folds_only=False,
            n_workers=1, register=False, device="cuda:0", verify=None,
        )
        print("  ⚠  Jupyter detected — running in --dry-run mode.")
        print("     For full preprocessing, run from a terminal:")
        print("         python -m preprocessing.brats_preprocess\n")
    else:
        args = parser.parse_args()

    # Resolve dataset root
    dataset_roots = {
        "BraTS2021": BRATS2021_ROOT,
        "BraTS2023_Adult": BRATS2023_ADU_ROOT,
        "BraTS2023_Pediatric": BRATS2023_PED_ROOT,
    }
    raw_root = dataset_roots[args.dataset]
    output_root = os.path.join(PREPROCESSED_DIR, args.dataset)

    preprocessor = BraTSPreprocessor(
        raw_root=raw_root,
        output_root=output_root,
        dataset=args.dataset,
        n_workers=1 if args.dry_run else args.n_workers,
        skip_existing=not args.dry_run,
        device=args.device,
        do_registration=args.register,
    )

    # --- Verify mode ---
    if args.verify:
        ok = preprocessor.verify(args.verify)
        sys.exit(0 if ok else 1)

    # --- Folds-only mode ---
    if args.folds_only:
        preprocessor.create_folds(n_folds=5, seed=42)
        return

    # --- Dry run ---
    if args.dry_run:
        logger.info("=== DRY RUN MODE (2 subjects) ===")
        subjects = preprocessor._discover_subjects()
        if not subjects:
            logger.error("No subjects found. Check the dataset path.")
            return

        sample = subjects[:2]
        preprocessor.skip_existing = False
        preprocessor.run(subject_list=sample)

        # Shape checks and histograms
        print("\n" + "=" * 60)
        print("  DRY RUN DIAGNOSTICS")
        print("=" * 60)
        for subj_dir in sample:
            sid = os.path.basename(subj_dir)
            img_path = os.path.join(output_root, f"{sid}_image.npy")
            lbl_path = os.path.join(output_root, f"{sid}_label.npy")
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                img = np.load(img_path)
                lbl = np.load(lbl_path)
                print(f"\n  Subject: {sid}")
                print(f"    Image shape : {img.shape}  dtype={img.dtype}")
                print(f"    Label shape : {lbl.shape}  dtype={lbl.dtype}")
                print(f"    Image range : [{img.min():.3f}, {img.max():.3f}]")
                print(f"    NaN: {np.any(np.isnan(img))}  Inf: {np.any(np.isinf(img))}")

                for c in range(4):
                    ch = img[c]
                    nz = ch[ch != 0]
                    if len(nz) > 0:
                        print(
                            f"    Ch{c}: mean={nz.mean():.4f}  "
                            f"std={nz.std():.4f}  "
                            f"min={nz.min():.3f}  max={nz.max():.3f}"
                        )

                _print_label_histogram(lbl, sid)
                preprocessor.verify(sid)
            else:
                logger.warning(f"  Output files not found for {sid}")

        print("\n" + "=" * 60)
        print("  Dry run complete.")
        print("=" * 60)
        return

    # --- Full run ---
    preprocessor.run()

    # Create folds for BraTS2021
    if args.dataset == "BraTS2021":
        preprocessor.create_folds(n_folds=5, seed=42)

    # Quick verification on 3 random subjects
    subjects = preprocessor._discover_subjects()
    if len(subjects) >= 3:
        import random
        random.seed(42)
        sample = random.sample(subjects, 3)
        logger.info("\nVerifying 3 random subjects …")
        for s in sample:
            sid = os.path.basename(s)
            preprocessor.verify(sid)


if __name__ == "__main__":
    main()
