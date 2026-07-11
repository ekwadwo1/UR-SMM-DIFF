#!/usr/bin/env python3
"""
preprocessing/fastmri_preprocess.py — NYU fastMRI Brain Offline Preprocessing (Phase 2-B)
=========================================================================================

Paper Section 4.1: "To isolate and validate the generative prior against true
measurement structure, we utilized raw, multi-coil k-space from the NYU
fastMRI Brain dataset.  Reference images were reconstructed via standard
root-sum-of-squares (RSS) coil combination."

**CRITICAL**: fastMRI is used ONLY for physics validation — it is NOT used
for training the main model.  The workflow is:
  1. Preprocess here → reference_image.npy  [1, 128, 128, 128]  float32
  2. Apply PhysicsCorruptionOperator (Phase 3) to get corrupted volume
  3. Run UR-SSM-Diff restoration
  4. Compare restored vs. reference via PSNR / SSIM

**CRITICAL**: fastMRI is single-contrast.  The model expects C=4 channels.
This module saves [1, 128, 128, 128].  Channel replication to 4 is handled
at the DATASET level (Phase 9), NOT here.

Pipeline
--------
1. Load multi-coil k-space from .h5  (or DICOM → FFT fallback)
2. RSS coil combination per slice
3. Stack slices → 3D volume, resample to 1.0 mm iso, crop/pad to 128³
4. Normalise: divide by 99th percentile, clip [0, 1]
5. Save: reference_image.npy, raw_kspace.npy, meta.json

Hardware:  2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
Usage
-----
    python -m preprocessing.fastmri_preprocess                  # full run
    python -m preprocessing.fastmri_preprocess --dry-run        # 2 files only
    python -m preprocessing.fastmri_preprocess --verify <ID>    # check one file
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom as scipy_zoom
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fastmri_preprocess")

# ---------------------------------------------------------------------------
# Path constants (match master prompt exactly)
# ---------------------------------------------------------------------------
DRIVE_ROOT: str = (
    "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/UR_SSM_DIFF_DATASETS"
)
FASTMRI_ROOT: str = os.path.join(DRIVE_ROOT, "brain_fastMRI_DICOM")
OUTPUT_DIR: str = os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs")
PREPROCESSED_DIR: str = os.path.join(OUTPUT_DIR, "preprocessed", "fastMRI_Brain")

# ---------------------------------------------------------------------------
# Target geometry
# ---------------------------------------------------------------------------
TARGET_SIZE: int = 128
TARGET_SPACING_MM: float = 1.0


# ======================================================================= #
#  Standalone worker  (module-level for ProcessPoolExecutor pickling)      #
# ======================================================================= #

def _process_single_file(
    file_path: str,
    output_root: str,
    skip_existing: bool,
) -> Dict[str, Any]:
    """
    Process one fastMRI file end-to-end (Steps 1-5).

    Paper Section 4.1 — "Reference images were reconstructed via standard
    root-sum-of-squares (RSS) coil combination."

    Parameters
    ----------
    file_path : str
        Path to an .h5 file or a DICOM directory.
    output_root : str
        Where to write outputs.
    skip_existing : bool
        Skip if output files already exist.

    Returns
    -------
    dict with keys: file_id, status ('ok'|'skipped'|'failed'), error, elapsed.
    """
    file_id = Path(file_path).stem  # filename without extension
    t0 = time.time()

    try:
        # Paths
        ref_out   = os.path.join(output_root, f"{file_id}_reference.npy")
        ksp_out   = os.path.join(output_root, f"{file_id}_kspace.npy")
        meta_out  = os.path.join(output_root, f"{file_id}_meta.json")

        if skip_existing and os.path.exists(ref_out) and os.path.exists(meta_out):
            return {"file_id": file_id, "status": "skipped",
                    "error": None, "elapsed": time.time() - t0}

        os.makedirs(output_root, exist_ok=True)

        # ==============================================================
        # Step 1 — Load k-space
        # ==============================================================
        kspace, kspace_meta = _load_kspace(file_path)
        # kspace: [n_slices, n_coils, H, W] complex64

        n_slices, n_coils, kH, kW = kspace.shape

        # ==============================================================
        # Step 2 — RSS coil combination per slice
        #   rss = sqrt( sum_over_coils( |IFFT2(k)|² ) )
        # ==============================================================
        rss_slices = _rss_reconstruction(kspace)
        # rss_slices: [n_slices, H, W]  float32

        # ==============================================================
        # Step 3 — 3D volume assembly + resample + crop/pad to 128³
        # ==============================================================
        #  Stack slices: array order (slice, H, W) → volume (H, W, n_slices)
        #  Convention: first two dims are in-plane, third is slice/through-plane
        volume_3d = np.transpose(rss_slices, (1, 2, 0))  # [H, W, n_slices]

        # Infer spacing from metadata if available, else assume typical fastMRI
        in_plane_mm  = kspace_meta.get("in_plane_spacing_mm", 0.7)
        slice_gap_mm = kspace_meta.get("slice_spacing_mm", 5.0)

        volume_iso, zoom_factors = _resample_isotropic(
            volume_3d,
            current_spacing=(in_plane_mm, in_plane_mm, slice_gap_mm),
            target_spacing=TARGET_SPACING_MM,
        )
        # volume_iso: [D', H', W']  arbitrary sizes close to physical dimensions

        volume_128 = _crop_or_pad_128(volume_iso, target=TARGET_SIZE)
        # volume_128: [128, 128, 128]

        # ==============================================================
        # Step 4 — Normalize: divide by 99th percentile, clip [0, 1]
        # ==============================================================
        p99 = float(np.percentile(volume_128[volume_128 > 0], 99)) if np.any(volume_128 > 0) else 1.0
        p99 = max(p99, 1e-8)  # safety
        volume_norm = np.clip(volume_128 / p99, 0.0, 1.0).astype(np.float32)

        # Add channel dim → [1, 128, 128, 128]
        volume_out = volume_norm[np.newaxis, ...]  # [1, D, H, W]

        # ==============================================================
        # Step 5 — Save
        # ==============================================================
        np.save(ref_out, volume_out)                            # [1,128,128,128] f32
        np.save(ksp_out, kspace.astype(np.complex64))           # raw k-space
        meta = {
            "file_id":              file_id,
            "source_path":          file_path,
            "n_slices":             int(n_slices),
            "n_coils":              int(n_coils),
            "kspace_shape":         list(kspace.shape),
            "rss_shape":            list(rss_slices.shape),
            "volume_pre_resample":  list(volume_3d.shape),
            "volume_post_resample": list(volume_iso.shape),
            "zoom_factors":         [float(z) for z in zoom_factors],
            "in_plane_spacing_mm":  float(in_plane_mm),
            "slice_spacing_mm":     float(slice_gap_mm),
            "p99_value":            float(p99),
            "final_shape":          list(volume_out.shape),
        }
        meta.update(kspace_meta)
        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return {"file_id": file_id, "status": "ok",
                "error": None, "elapsed": time.time() - t0}

    except Exception as e:
        tb = traceback.format_exc()
        return {"file_id": file_id, "status": "failed",
                "error": f"{e}\n{tb}", "elapsed": time.time() - t0}


# ======================================================================= #
#  Module-level helpers  (pickle-safe)                                     #
# ======================================================================= #

def _load_kspace(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Step 1 — Load multi-coil k-space.

    Supports two input formats:
    - .h5 files (standard fastMRI HDF5):  key 'kspace' → [slices, coils, H, W] complex64
    - DICOM directory: load images with pydicom, derive k-space via 2D FFT

    Returns
    -------
    kspace : ndarray [n_slices, n_coils, H, W] complex64
    meta   : dict with any extracted metadata
    """
    meta: Dict[str, Any] = {}

    # ─── HDF5 path ────────────────────────────────────────────────────────
    if path.lower().endswith(".h5") or path.lower().endswith(".hdf5"):
        import h5py

        with h5py.File(path, "r") as f:
            if "kspace" in f:
                kspace = np.array(f["kspace"])  # [n_slices, n_coils, H, W] complex
            elif "reconstruction_rss" in f:
                # Some fastMRI files provide pre-computed RSS instead of k-space
                rss = np.array(f["reconstruction_rss"])  # [n_slices, H, W] float
                # Fake single-coil k-space by taking forward FFT
                kspace = np.fft.fft2(rss, norm="ortho")  # [n_slices, H, W] complex
                kspace = kspace[:, np.newaxis, :, :]      # [n_slices, 1, H, W]
                meta["fallback"] = "reconstruction_rss_to_kspace"
            else:
                available = list(f.keys())
                raise KeyError(
                    f"HDF5 file has no 'kspace' or 'reconstruction_rss' key. "
                    f"Available keys: {available}"
                )

            # Extract any available metadata attributes
            attrs = dict(f.attrs) if hasattr(f, "attrs") else {}
            for key in ("acquisition", "patient_id", "encoding_size",
                        "recon_size", "padding_left", "padding_right",
                        "ismrmrd_header"):
                if key in attrs:
                    val = attrs[key]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    meta[key] = val

            # Try to extract spacing from ismrmrd header
            if "ismrmrd_header" in meta:
                meta.update(_parse_ismrmrd_spacing(str(meta["ismrmrd_header"])))

        kspace = kspace.astype(np.complex64)
        if kspace.ndim == 3:
            # Single-coil: [n_slices, H, W] → [n_slices, 1, H, W]
            kspace = kspace[:, np.newaxis, :, :]

        return kspace, meta

    # ─── DICOM directory path ─────────────────────────────────────────────
    if os.path.isdir(path):
        return _load_dicom_to_kspace(path)

    # ─── Single DICOM file ───────────────────────────────────────────────
    if os.path.isfile(path) and not path.lower().endswith((".npy", ".npz")):
        # Treat as single DICOM or try anyway
        parent = os.path.dirname(path)
        return _load_dicom_to_kspace(parent)

    raise ValueError(f"Unsupported input: {path}")


def _load_dicom_to_kspace(dicom_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load DICOM series from a directory, reconstruct images, derive k-space via FFT.

    Returns [n_slices, 1, H, W] complex64  (single "virtual coil").
    """
    import pydicom

    meta: Dict[str, Any] = {"source_format": "dicom"}

    # Discover DICOM files (filter macOS resource forks ._*)
    dcm_files = []
    for entry in sorted(os.listdir(dicom_dir)):
        if entry.startswith("._") or entry.startswith("."):
            continue  # skip macOS resource forks and hidden files
        full = os.path.join(dicom_dir, entry)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in (".dcm", ".ima", ".dicom", ""):
            try:
                pydicom.dcmread(full, stop_before_pixels=True)
                dcm_files.append(full)
            except Exception:
                continue

    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    # Sort by instance number or filename
    def _sort_key(fp):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    dcm_files.sort(key=_sort_key)

    # Load pixel data — collect with shapes for filtering
    slices_with_shapes = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp)
            arr = ds.pixel_array.astype(np.float32)
            # Apply rescale if present
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            slices_with_shapes.append((arr, arr.shape))
        except Exception:
            continue  # skip unreadable slices

    if not slices_with_shapes:
        raise ValueError(f"No readable DICOM pixel data in {dicom_dir}")

    # Filter to the most common slice shape (handles mixed-resolution
    # directories where scout scans or localisers have different dimensions)
    from collections import Counter as _Counter
    shape_counts = _Counter(s[1] for s in slices_with_shapes)
    most_common_shape = shape_counts.most_common(1)[0][0]
    slices = [s[0] for s in slices_with_shapes if s[1] == most_common_shape]

    if len(slices) < len(slices_with_shapes):
        n_dropped = len(slices_with_shapes) - len(slices)
        meta["dropped_slices"] = n_dropped
        meta["dropped_reason"] = (
            f"Filtered {n_dropped} slices with non-standard shape "
            f"(kept {len(slices)} slices with shape {most_common_shape})"
        )

    # Extract spacing from first DICOM of the kept series
    ds0 = pydicom.dcmread(dcm_files[0])
    pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])
    slice_thickness = float(getattr(ds0, "SliceThickness", 5.0))
    meta["in_plane_spacing_mm"] = float(pixel_spacing[0])
    meta["slice_spacing_mm"] = slice_thickness
    meta["n_dicom_files"] = len(dcm_files)

    volume = np.stack(slices, axis=0)  # [n_slices, H, W]

    # Derive k-space via 2D FFT per slice (single virtual coil)
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(volume, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)

    kspace = kspace[:, np.newaxis, :, :]  # [n_slices, 1, H, W]
    return kspace, meta


def _parse_ismrmrd_spacing(header_str: str) -> Dict[str, float]:
    """
    Extract field-of-view and encoding matrix from ISMRMRD XML header
    to estimate voxel spacing.

    Returns dict with 'in_plane_spacing_mm' and 'slice_spacing_mm' if parseable.
    """
    result: Dict[str, float] = {}
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(header_str)
        ns = {"i": "http://www.ismrm.org/ISMRMRD"}

        # Field of view (mm)
        fov_x = root.find(".//i:reconSpace/i:fieldOfView_mm/i:x", ns)
        fov_y = root.find(".//i:reconSpace/i:fieldOfView_mm/i:y", ns)
        fov_z = root.find(".//i:reconSpace/i:fieldOfView_mm/i:z", ns)

        # Encoding matrix
        mat_x = root.find(".//i:reconSpace/i:matrixSize/i:x", ns)
        mat_y = root.find(".//i:reconSpace/i:matrixSize/i:y", ns)
        mat_z = root.find(".//i:reconSpace/i:matrixSize/i:z", ns)

        if fov_x is not None and mat_x is not None:
            result["in_plane_spacing_mm"] = float(fov_x.text) / float(mat_x.text)
        if fov_z is not None and mat_z is not None:
            nz = float(mat_z.text)
            if nz > 1:
                result["slice_spacing_mm"] = float(fov_z.text) / nz

    except Exception:
        pass  # parsing failure is non-critical, caller uses defaults

    return result


def _rss_reconstruction(kspace: np.ndarray) -> np.ndarray:
    """
    Step 2 — Root-Sum-of-Squares (RSS) coil combination.

    Paper Section 4.1: "Reference images were reconstructed via standard
    root-sum-of-squares (RSS) coil combination."

    For each slice:
        rss = sqrt( sum_c( |IFFT2(k_c)|² ) )

    Parameters
    ----------
    kspace : ndarray [n_slices, n_coils, H, W] complex64

    Returns
    -------
    rss : ndarray [n_slices, H, W] float32
    """
    n_slices, n_coils, H, W = kspace.shape

    rss = np.zeros((n_slices, H, W), dtype=np.float32)

    for s in range(n_slices):
        sum_sq = np.zeros((H, W), dtype=np.float64)
        for c in range(n_coils):
            # Centred IFFT: ifftshift → ifft2 → fftshift
            coil_ksp = kspace[s, c]  # [H, W] complex
            coil_img = np.fft.fftshift(
                np.fft.ifft2(
                    np.fft.ifftshift(coil_ksp),
                    norm="ortho",
                ),
            )
            sum_sq += np.abs(coil_img) ** 2
        rss[s] = np.sqrt(sum_sq).astype(np.float32)

    return rss


def _resample_isotropic(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: float = 1.0,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Step 3a — Resample a 3D volume to isotropic resolution.

    Paper Section 4.1: "resampling to 1.0 mm isotropic".

    Parameters
    ----------
    volume : ndarray [D, H, W]
    current_spacing : (sx, sy, sz) in mm
    target_spacing : target isotropic spacing in mm

    Returns
    -------
    resampled : ndarray with new shape
    zoom_factors : (zx, zy, zz) applied
    """
    sx, sy, sz = current_spacing

    # Skip if already close to target
    if all(abs(s - target_spacing) < 0.05 for s in (sx, sy, sz)):
        return volume, (1.0, 1.0, 1.0)

    zoom_factors = (
        sx / target_spacing,
        sy / target_spacing,
        sz / target_spacing,
    )

    resampled = scipy_zoom(
        volume, zoom_factors, order=3, mode="nearest", prefilter=True,
    ).astype(np.float32)

    return resampled, zoom_factors


def _crop_or_pad_128(volume: np.ndarray, target: int = 128) -> np.ndarray:
    """
    Step 3b — Centre-crop or zero-pad each axis to target size.

    Uses the same logic as the BraTS preprocessor (Phase 2-A, Step 4):
    centroid-aware cropping if volume exceeds target in any dimension.

    Parameters
    ----------
    volume : ndarray [D, H, W]
    target : int (default 128)

    Returns
    -------
    ndarray [target, target, target]
    """
    D, H, W = volume.shape

    # Compute centroid of non-zero region
    nz = np.argwhere(volume > 0)
    if len(nz) > 0:
        centroid = nz.mean(axis=0).astype(int)
    else:
        centroid = np.array([D // 2, H // 2, W // 2])

    slices_list = []
    pads_list   = []

    for ax, (dim_size, c) in enumerate(zip([D, H, W], centroid)):
        if dim_size <= target:
            # Pad
            start, end = 0, dim_size
            pad_b = (target - dim_size) // 2
            pad_a = target - dim_size - pad_b
        else:
            # Crop centred on non-zero centroid
            half = target // 2
            start = max(0, c - half)
            end = start + target
            if end > dim_size:
                end = dim_size
                start = end - target
            pad_b, pad_a = 0, 0

        slices_list.append(slice(start, end))
        pads_list.append((pad_b, pad_a))

    # Apply crop
    ds, hs, ws = slices_list
    volume = volume[ds, hs, ws]

    # Apply pad if needed
    if any(p > 0 for pair in pads_list for p in pair):
        volume = np.pad(volume, pads_list, mode="constant", constant_values=0)

    assert volume.shape == (target, target, target), (
        f"Expected ({target},{target},{target}), got {volume.shape}"
    )
    return volume


# ======================================================================= #
#  Main Class                                                              #
# ======================================================================= #

class FastMRIPreprocessor:
    """
    Offline NYU fastMRI Brain preprocessing (Paper Section 4.1).

    fastMRI data is used ONLY for physics validation:
    - Reconstruct reference images via RSS coil combination
    - Later: corrupt with PhysicsCorruptionOperator, restore, evaluate PSNR/SSIM

    Output: [1, 128, 128, 128] float32 reference volumes normalised to [0, 1].
    Channel replication to C=4 is done at the Dataset level (Phase 9).

    Parameters
    ----------
    raw_root : str
        Root directory containing .h5 files and/or DICOM sub-directories.
    output_root : str
        Where to save preprocessed .npy and metadata files.
    n_workers : int
        Number of parallel CPU workers.
    skip_existing : bool
        Skip files whose outputs already exist.
    """

    def __init__(
        self,
        raw_root: str,
        output_root: str,
        n_workers: int = 4,
        skip_existing: bool = True,
    ) -> None:
        self.raw_root = raw_root
        self.output_root = output_root
        self.n_workers = n_workers
        self.skip_existing = skip_existing

        os.makedirs(output_root, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  File discovery                                                     #
    # ------------------------------------------------------------------ #

    def _discover_files(self) -> List[str]:
        """
        Find all processable inputs: .h5 files and DICOM directories.

        Returns sorted list of file/directory paths.
        """
        if not os.path.isdir(self.raw_root):
            raise FileNotFoundError(f"fastMRI root not found: {self.raw_root}")

        inputs: List[str] = []

        # 1. HDF5 files (standard fastMRI format)
        for pattern in ("*.h5", "*.hdf5"):
            inputs.extend(glob.glob(os.path.join(self.raw_root, pattern)))
            inputs.extend(glob.glob(os.path.join(self.raw_root, "**", pattern),
                                     recursive=True))

        # 2. DICOM directories (alternative format per dataset path name)
        for entry in os.listdir(self.raw_root):
            if entry.startswith("._") or entry.startswith("."):
                continue
            full = os.path.join(self.raw_root, entry)
            if not os.path.isdir(full):
                continue
            # Check if directory contains DICOM files
            sub_files = [f for f in os.listdir(full)
                         if not f.startswith("._") and not f.startswith(".")]
            has_dicom = any(
                f.lower().endswith((".dcm", ".ima", ".dicom"))
                for f in sub_files
            )
            # Also check for files without extension (common DICOM pattern)
            if not has_dicom:
                has_dicom = any(
                    not os.path.splitext(f)[1] and os.path.isfile(os.path.join(full, f))
                    for f in sub_files[:10]  # check first 10 for speed
                )
            if has_dicom and full not in inputs:
                inputs.append(full)

        # Deduplicate and filter macOS resource forks
        inputs = sorted(set(
            p for p in inputs
            if not os.path.basename(p).startswith("._")
        ))
        logger.info(f"Discovered {len(inputs)} input files/dirs in {self.raw_root}")
        return inputs

    # ------------------------------------------------------------------ #
    #  Main run                                                           #
    # ------------------------------------------------------------------ #

    def run(self, file_list: Optional[List[str]] = None) -> None:
        """
        Process all (or selected) fastMRI files.

        Parameters
        ----------
        file_list : list of str, optional
            Specific file paths.  If None, discover all from raw_root.
        """
        if file_list is None:
            file_list = self._discover_files()

        if not file_list:
            logger.warning("No input files found. Check FASTMRI_ROOT path.")
            return

        logger.info(
            f"Processing {len(file_list)} files  |  "
            f"workers={self.n_workers}  |  skip_existing={self.skip_existing}"
        )

        results: List[Dict] = []
        t_start = time.time()

        if self.n_workers <= 1:
            # Sequential
            for fp in tqdm(file_list, desc="fastMRI preprocess"):
                r = _process_single_file(fp, self.output_root, self.skip_existing)
                results.append(r)
                if r["status"] == "failed":
                    logger.warning(f"  FAILED {r['file_id']}: {r['error'][:200]}")
        else:
            # Parallel
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {
                    pool.submit(
                        _process_single_file, fp, self.output_root, self.skip_existing
                    ): fp
                    for fp in file_list
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="fastMRI preprocess"):
                    r = future.result()
                    results.append(r)
                    if r["status"] == "failed":
                        logger.warning(f"  FAILED {r['file_id']}: {r['error'][:200]}")

        # Summary
        elapsed = time.time() - t_start
        ok      = sum(1 for r in results if r["status"] == "ok")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed  = sum(1 for r in results if r["status"] == "failed")

        logger.info(
            f"\nDone in {elapsed:.1f}s  |  OK={ok}  Skipped={skipped}  Failed={failed}"
        )
        if failed > 0:
            logger.warning("Failed files:")
            for r in results:
                if r["status"] == "failed":
                    logger.warning(f"  {r['file_id']}: {r['error'][:150]}")

        # Save report
        report_path = os.path.join(self.output_root, "preprocess_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "total": len(results),
                "ok": ok, "skipped": skipped, "failed": failed,
                "elapsed_sec": elapsed,
                "files": [
                    {k: v for k, v in r.items() if k != "error"}
                    for r in results
                ],
            }, f, indent=2)
        logger.info(f"Report saved to {report_path}")

    # ------------------------------------------------------------------ #
    #  Verification                                                       #
    # ------------------------------------------------------------------ #

    def verify(self, file_id: str) -> bool:
        """
        Verify a preprocessed fastMRI file: shapes, dtypes, value range.

        Parameters
        ----------
        file_id : str
            Stem of the original file (e.g. 'file_brain_AXT2_200_2000001').

        Returns
        -------
        bool : True if all checks pass.
        """
        ref_path  = os.path.join(self.output_root, f"{file_id}_reference.npy")
        ksp_path  = os.path.join(self.output_root, f"{file_id}_kspace.npy")
        meta_path = os.path.join(self.output_root, f"{file_id}_meta.json")

        ok = True

        # --- File existence -------------------------------------------------
        for path, name in [(ref_path, "reference"), (ksp_path, "kspace"),
                           (meta_path, "meta")]:
            if not os.path.exists(path):
                logger.error(f"  [{file_id}] {name} file missing: {path}")
                ok = False
        if not ok:
            return False

        # --- Load reference -------------------------------------------------
        ref = np.load(ref_path)

        # Shape: must be [1, 128, 128, 128]
        if ref.shape != (1, 128, 128, 128):
            logger.error(
                f"  [{file_id}] Reference shape {ref.shape} != (1,128,128,128)"
            )
            ok = False

        # Dtype
        if ref.dtype != np.float32:
            logger.error(f"  [{file_id}] Reference dtype {ref.dtype} != float32")
            ok = False

        # NaN / Inf
        if np.any(np.isnan(ref)):
            logger.error(f"  [{file_id}] Reference contains NaN")
            ok = False
        if np.any(np.isinf(ref)):
            logger.error(f"  [{file_id}] Reference contains Inf")
            ok = False

        # Value range [0, 1]
        vmin, vmax = float(ref.min()), float(ref.max())
        if vmin < -0.01 or vmax > 1.01:
            logger.warning(
                f"  [{file_id}] Reference range [{vmin:.4f}, {vmax:.4f}] "
                f"outside expected [0, 1]"
            )

        # Non-zero content
        nz_ratio = float(np.count_nonzero(ref)) / ref.size
        if nz_ratio < 0.01:
            logger.warning(f"  [{file_id}] Very low non-zero ratio: {nz_ratio:.4f}")

        # --- Load k-space ---------------------------------------------------
        ksp = np.load(ksp_path)
        if ksp.ndim != 4:
            logger.error(f"  [{file_id}] kspace ndim {ksp.ndim} != 4")
            ok = False
        if not np.iscomplexobj(ksp):
            logger.error(f"  [{file_id}] kspace dtype {ksp.dtype} is not complex")
            ok = False

        # --- Load meta ------------------------------------------------------
        with open(meta_path) as f:
            meta = json.load(f)

        # --- Report ---------------------------------------------------------
        if ok:
            logger.info(
                f"  [{file_id}] PASSED ✓  |  ref shape={ref.shape}  "
                f"range=[{vmin:.3f},{vmax:.3f}]  nz={nz_ratio:.2%}  "
                f"ksp={ksp.shape}  coils={meta.get('n_coils','?')}"
            )
        else:
            logger.error(f"  [{file_id}] FAILED ✗")

        return ok


# ======================================================================= #
#  __main__ — CLI + dry-run mode                                          #
# ======================================================================= #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "NYU fastMRI Brain Offline Preprocessing (Phase 2-B, "
            "Paper Section 4.1 — physics validation only)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only 2 files and print diagnostics.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Parallel CPU workers.",
    )
    parser.add_argument(
        "--verify", type=str, default=None, metavar="FILE_ID",
        help="Verify a single preprocessed file and exit.",
    )
    parser.add_argument(
        "--raw-root", type=str, default=FASTMRI_ROOT,
        help="Override raw data root path.",
    )
    parser.add_argument(
        "--output-root", type=str, default=PREPROCESSED_DIR,
        help="Override output directory.",
    )
    # --- Jupyter-safe argparse ------------------------------------------------
    _in_jupyter = any("ipykernel" in a or "kernel" in a for a in sys.argv)
    if _in_jupyter:
        args = argparse.Namespace(
            dry_run=True, n_workers=1, verify=None,
            raw_root=FASTMRI_ROOT, output_root=PREPROCESSED_DIR,
        )
        print("  ⚠  Jupyter detected — running in --dry-run mode.")
        print("     For full preprocessing, run from a terminal:")
        print("         python -m preprocessing.fastmri_preprocess\n")
    else:
        args = parser.parse_args()

    preprocessor = FastMRIPreprocessor(
        raw_root=args.raw_root,
        output_root=args.output_root,
        n_workers=1 if args.dry_run else args.n_workers,
        skip_existing=not args.dry_run,
    )

    # --- Verify mode -------------------------------------------------------
    if args.verify:
        ok = preprocessor.verify(args.verify)
        sys.exit(0 if ok else 1)

    # --- Dry run -----------------------------------------------------------
    if args.dry_run:
        logger.info("=== DRY RUN MODE (up to 2 files) ===")
        try:
            files = preprocessor._discover_files()
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info(
                "\nDry run with synthetic data (no real fastMRI files found) …"
            )
            _run_synthetic_dry_run(args.output_root)
            return

        if not files:
            logger.warning("No files found. Running synthetic dry run instead.")
            _run_synthetic_dry_run(args.output_root)
            return

        sample = files[:2]
        preprocessor.skip_existing = False
        preprocessor.run(file_list=sample)

        # Diagnostics
        print("\n" + "=" * 60)
        print("  DRY RUN DIAGNOSTICS")
        print("=" * 60)
        for fp in sample:
            fid = Path(fp).stem
            preprocessor.verify(fid)

        print("=" * 60)
        return

    # --- Full run ----------------------------------------------------------
    preprocessor.run()


def _run_synthetic_dry_run(output_root: str) -> None:
    """
    Generate a synthetic fastMRI-like volume for testing when real data
    is not available (e.g., on a development machine).
    """
    logger.info("Generating synthetic multi-coil k-space for shape testing …")

    n_slices, n_coils, H, W = 20, 8, 320, 320

    # Synthetic image: a 3D brain-like ellipsoid
    z, y, x = np.mgrid[:n_slices, :H, :W]
    zc, yc, xc = n_slices / 2, H / 2, W / 2
    ellipsoid = (
        ((z - zc) / (n_slices * 0.4)) ** 2
        + ((y - yc) / (H * 0.35)) ** 2
        + ((x - xc) / (W * 0.3)) ** 2
    )
    phantom = np.where(ellipsoid < 1.0, 1000.0, 0.0).astype(np.float32)

    # Generate multi-coil k-space with coil sensitivity variation
    kspace = np.zeros((n_slices, n_coils, H, W), dtype=np.complex64)
    for c in range(n_coils):
        # Simple coil sensitivity (spatial polynomial)
        sensitivity = 1.0 + 0.3 * np.sin(2 * np.pi * c / n_coils + y / H * np.pi)
        coil_img = phantom * sensitivity[np.newaxis, :, :]  # broadcast over slices
        for s in range(n_slices):
            kspace[s, c] = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(coil_img[s]), norm="ortho")
            )

    # Save as temp .h5
    os.makedirs(output_root, exist_ok=True)
    h5_path = os.path.join(output_root, "_synthetic_test.h5")

    import h5py
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("kspace", data=kspace)

    logger.info(f"Synthetic k-space: {kspace.shape}  saved to {h5_path}")

    # Process it
    result = _process_single_file(h5_path, output_root, skip_existing=False)
    print(f"\n  Processing result: {result['status']}  "
          f"({result['elapsed']:.2f}s)")
    if result["status"] == "failed":
        print(f"  Error: {result['error']}")
        return

    # Verify
    fid = "_synthetic_test"
    ref = np.load(os.path.join(output_root, f"{fid}_reference.npy"))
    ksp = np.load(os.path.join(output_root, f"{fid}_kspace.npy"))

    print(f"\n  Reference shape : {ref.shape}    dtype={ref.dtype}")
    print(f"  Reference range : [{ref.min():.4f}, {ref.max():.4f}]")
    print(f"  Non-zero ratio  : {np.count_nonzero(ref) / ref.size:.2%}")
    print(f"  NaN: {np.any(np.isnan(ref))}  Inf: {np.any(np.isinf(ref))}")
    print(f"  K-space shape   : {ksp.shape}    dtype={ksp.dtype}")
    print(f"  K-space complex : {np.iscomplexobj(ksp)}")

    # Shape assertions
    assert ref.shape == (1, 128, 128, 128), f"FAIL: ref shape {ref.shape}"
    assert ref.dtype == np.float32, f"FAIL: ref dtype {ref.dtype}"
    assert 0.0 <= ref.min() and ref.max() <= 1.0, f"FAIL: ref range"
    assert ksp.ndim == 4, f"FAIL: ksp ndim {ksp.ndim}"
    assert np.iscomplexobj(ksp), "FAIL: ksp not complex"

    print(f"\n  ✓ All shape/dtype assertions passed")
    print(f"  ✓ Output: [1, 128, 128, 128] float32 ∈ [0, 1]")
    print(f"  ✓ K-space preserved for PhysicsCorruptionOperator (Phase 3)")

    # Cleanup synthetic file
    os.remove(h5_path)
    logger.info("Synthetic test file cleaned up.")


if __name__ == "__main__":
    main()
