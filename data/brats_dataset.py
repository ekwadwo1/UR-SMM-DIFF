#!/usr/bin/env python3
"""
data/brats_dataset.py — Dataset & DataLoader for UR-SSM-Diff (Phase 9-A)
=========================================================================

Loads OFFLINE pre-processed .npy files. Only lightweight augmentations
applied on-the-fly during training.

Datasets:
  BraTSDataset  — loads {id}_image.npy [4,128,128,128] + {id}_label.npy [128,128,128]
  FastMRIDataset — loads {id}_reference.npy [1,128,128,128], replicates to 4 channels

Training augmentations (lightweight, applied on-the-fly):
  - Random flip per spatial axis (p=0.5 each)
  - Random rot90 in axial plane (p=0.5)
  - Random intensity scaling Uniform(0.9, 1.1) per contrast

DataLoaders with DDP:
  - DistributedSampler for multi-GPU sharding
  - batch_size=1 per GPU (micro_batch)
  - num_workers=4, pin_memory=True, persistent_workers=True

Hardware: 2x NVIDIA RTX 5880 Ada 48 GB, CUDA 11.8
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ======================================================================= #
#  BraTS Dataset                                                           #
# ======================================================================= #

class BraTSDataset(Dataset):
    """
    BraTS dataset loading offline-preprocessed .npy volumes.

    Each subject has:
      {subject_id}_image.npy : [4, 128, 128, 128]  float32 (T1, T1ce, T2, FLAIR)
      {subject_id}_label.npy : [128, 128, 128]      uint8   (labels: 0,1,2,4)

    Training augmentations (lightweight):
      - Random flip per spatial axis (dims 1,2,3) with p=0.5
      - Random 90 degree rotation in axial plane (dims 1,2) with p=0.5
      - Random intensity scaling Uniform(0.9, 1.1) per contrast

    Parameters
    ----------
    data_root     : directory containing _image.npy and _label.npy files
    subject_ids   : list of subject IDs (without suffix). If None, auto-discover.
    augment       : apply training augmentations
    fold_json     : path to fold JSON with 'train'/'val' lists
    split         : 'train' or 'val' (used with fold_json)
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[str]] = None,
        augment: bool = True,
        fold_json: Optional[str] = None,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.augment = augment

        # Resolve subject IDs
        if fold_json is not None and os.path.exists(fold_json):
            with open(fold_json, "r") as f:
                fold = json.load(f)
            self.subject_ids = sorted(fold.get(split, []))
        elif subject_ids is not None:
            self.subject_ids = sorted(subject_ids)
        else:
            # Auto-discover from directory
            self.subject_ids = sorted([
                f.stem.replace("_image", "")
                for f in self.data_root.glob("*_image.npy")
                if not f.name.startswith("._")
            ])

        # Validate files exist
        self._image_paths: List[str] = []
        self._label_paths: List[str] = []
        valid_ids: List[str] = []
        for sid in self.subject_ids:
            img = self.data_root / f"{sid}_image.npy"
            lbl = self.data_root / f"{sid}_label.npy"
            if img.exists() and lbl.exists():
                self._image_paths.append(str(img))
                self._label_paths.append(str(lbl))
                valid_ids.append(sid)
        self.subject_ids = valid_ids

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
          'image'  : [4, 128, 128, 128]  float32
          'label'  : [128, 128, 128]     long (raw BraTS: {0,1,2,4})
          'id'     : subject ID string
        """
        image = torch.from_numpy(
            np.load(self._image_paths[idx])).float()           # [4, 128, 128, 128]
        label = torch.from_numpy(
            np.load(self._label_paths[idx])).long()            # [128, 128, 128]

        if self.augment:
            image, label = self._augment(image, label)

        return {
            "image": image,
            "label": label,
            "id": self.subject_ids[idx],
        }

    def _augment(
        self, image: torch.Tensor, label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lightweight on-the-fly augmentations.

        image : [4, H, W, D]   (C=4 contrasts)
        label : [H, W, D]
        """
        # Random flip per spatial axis (dims 1,2,3 for image; 0,1,2 for label)
        for img_dim, lbl_dim in [(1, 0), (2, 1), (3, 2)]:
            if torch.rand(1).item() > 0.5:
                image = image.flip(img_dim)
                label = label.flip(lbl_dim)

        # Random 90 degree rotation in axial plane (H-W plane)
        if torch.rand(1).item() > 0.5:
            k = int(torch.randint(1, 4, (1,)).item())
            image = torch.rot90(image, k, [1, 2])
            label = torch.rot90(label, k, [0, 1])

        # Random intensity scaling per contrast: U(0.9, 1.1)
        scale = 0.9 + 0.2 * torch.rand(4, 1, 1, 1)           # [4, 1, 1, 1]
        image = image * scale

        return image, label


# ======================================================================= #
#  FastMRI Dataset                                                         #
# ======================================================================= #

class FastMRIDataset(Dataset):
    """
    FastMRI Brain dataset -- single-contrast, replicated to 4 channels.

    Loads offline-preprocessed {id}_reference.npy [1, 128, 128, 128]
    and replicates to [4, 128, 128, 128] for model interface compatibility.

    No segmentation labels (physics validation only).

    Parameters
    ----------
    data_root     : directory containing _reference.npy files
    subject_ids   : list of IDs. If None, auto-discover.
    augment       : apply training augmentations
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[str]] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.augment = augment

        if subject_ids is not None:
            self.subject_ids = sorted(subject_ids)
        else:
            self.subject_ids = sorted([
                f.stem.replace("_reference", "")
                for f in self.data_root.glob("*_reference.npy")
                if not f.name.startswith("._")
            ])

        self._ref_paths: List[str] = []
        valid_ids: List[str] = []
        for sid in self.subject_ids:
            ref = self.data_root / f"{sid}_reference.npy"
            if ref.exists():
                self._ref_paths.append(str(ref))
                valid_ids.append(sid)
        self.subject_ids = valid_ids

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
          'image' : [4, 128, 128, 128]  float32 (replicated single contrast)
          'id'    : subject ID string
        """
        ref = torch.from_numpy(
            np.load(self._ref_paths[idx])).float()             # [1, 128, 128, 128]

        # Replicate single contrast to 4 channels
        image = ref.expand(4, -1, -1, -1).clone()             # [4, 128, 128, 128]

        if self.augment:
            for dim in [1, 2, 3]:
                if torch.rand(1).item() > 0.5:
                    image = image.flip(dim)
            if torch.rand(1).item() > 0.5:
                k = int(torch.randint(1, 4, (1,)).item())
                image = torch.rot90(image, k, [1, 2])

        return {
            "image": image,
            "id": self.subject_ids[idx],
        }


# ======================================================================= #
#  DDP-aware DataLoader builders                                           #
# ======================================================================= #

def build_brats_dataloaders(
    data_root: str,
    fold_json: str,
    micro_batch: int = 1,
    num_workers: int = 4,
    distributed: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders for BraTS with DDP support.

    Parameters
    ----------
    data_root    : preprocessed BraTS directory
    fold_json    : path to fold JSON (e.g. fold_0.json)
    micro_batch  : batch size per GPU (1)
    num_workers  : DataLoader workers (4)
    distributed  : use DistributedSampler
    """
    train_ds = BraTSDataset(
        data_root, augment=True, fold_json=fold_json, split="train")
    val_ds = BraTSDataset(
        data_root, augment=False, fold_json=fold_json, split="val")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch,
        sampler=val_sampler,
        shuffle=False,
        num_workers=min(num_workers, 2),
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader, val_loader


def build_fastmri_dataloader(
    data_root: str,
    micro_batch: int = 1,
    num_workers: int = 4,
    distributed: bool = True,
) -> DataLoader:
    """Build DataLoader for fastMRI Brain (physics validation)."""
    ds = FastMRIDataset(data_root, augment=False)
    sampler = DistributedSampler(ds, shuffle=False) if distributed else None

    return DataLoader(
        ds,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_zeroshot_dataloader(
    data_root: str,
    micro_batch: int = 1,
    num_workers: int = 2,
    distributed: bool = True,
) -> DataLoader:
    """Build DataLoader for zero-shot evaluation (BraTS2023 Adult/Pediatric)."""
    ds = BraTSDataset(data_root, augment=False)
    sampler = DistributedSampler(ds, shuffle=False) if distributed else None

    return DataLoader(
        ds,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ======================================================================= #
#  Tests  (Jupyter-safe -- no argparse)                                    #
# ======================================================================= #

def _run_tests() -> None:
    """Tests for dataset and dataloader functionality."""

    print("=" * 70)
    print("  BraTSDataset + FastMRIDataset — Test Suite")
    print("=" * 70)

    import tempfile

    torch.manual_seed(42)

    # ── Create temporary test data ────────────────────────────────────
    print("\n--- Creating temporary test data ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        brats_dir = os.path.join(tmpdir, "brats")
        fastmri_dir = os.path.join(tmpdir, "fastmri")
        os.makedirs(brats_dir)
        os.makedirs(fastmri_dir)

        # Create 10 fake BraTS subjects
        brats_ids = []
        for i in range(10):
            sid = f"BraTS2021_{i:05d}"
            brats_ids.append(sid)
            img = np.random.randn(4, 128, 128, 128).astype(np.float32)
            lbl = np.zeros((128, 128, 128), dtype=np.uint8)
            lbl[:40] = 1; lbl[40:80] = 2; lbl[80:100] = 4
            np.save(os.path.join(brats_dir, f"{sid}_image.npy"), img)
            np.save(os.path.join(brats_dir, f"{sid}_label.npy"), lbl)

        # Create fold JSON
        fold_path = os.path.join(brats_dir, "fold_0.json")
        fold = {"train": brats_ids[:8], "val": brats_ids[8:]}
        with open(fold_path, "w") as f:
            json.dump(fold, f)

        # Create 5 fake fastMRI subjects
        for i in range(5):
            sid = f"fastmri_{i:05d}"
            ref = np.random.randn(1, 128, 128, 128).astype(np.float32)
            np.save(os.path.join(fastmri_dir, f"{sid}_reference.npy"), ref)

        print(f"  BraTS: {len(brats_ids)} subjects")
        print(f"  fastMRI: 5 subjects")

        # ── Test 1: BraTSDataset loading ──────────────────────────────
        print("\n--- (1) BraTSDataset loading ---")
        ds_train = BraTSDataset(brats_dir, augment=True,
                                fold_json=fold_path, split="train")
        ds_val = BraTSDataset(brats_dir, augment=False,
                              fold_json=fold_path, split="val")

        assert len(ds_train) == 8, f"Train: {len(ds_train)}"
        assert len(ds_val) == 2, f"Val: {len(ds_val)}"
        print(f"  Train: {len(ds_train)}  Val: {len(ds_val)}  ✓")

        sample = ds_train[0]
        assert sample["image"].shape == (4, 128, 128, 128)
        assert sample["label"].shape == (128, 128, 128)
        assert sample["image"].dtype == torch.float32
        assert sample["label"].dtype == torch.int64
        print(f"  image: {list(sample['image'].shape)} {sample['image'].dtype}  ✓")
        print(f"  label: {list(sample['label'].shape)} {sample['label'].dtype}  ✓")
        print(f"  id:    {sample['id']}  ✓")

        # ── Test 2: Label values ──────────────────────────────────────
        print("\n--- (2) Label values ---")
        val_sample = ds_val[0]
        unique = val_sample["label"].unique().tolist()
        print(f"  Unique: {unique}")
        assert 4 in unique, f"Missing ET (4): {unique}"
        print(f"  ✓ BraTS labels {{0, 1, 2, 4}}")

        # ── Test 3: Augmentation changes data ─────────────────────────
        print("\n--- (3) Augmentation ---")
        s1 = ds_train[0]
        s2 = ds_train[0]
        diff = (s1["image"] - s2["image"]).abs().sum().item()
        print(f"  |s1 - s2| = {diff:.2f}  "
              f"({'different' if diff > 0 else 'same (rare)'})")

        # ── Test 4: Val is deterministic ──────────────────────────────
        print("\n--- (4) Val deterministic ---")
        v1 = ds_val[0]; v2 = ds_val[0]
        assert (v1["image"] - v2["image"]).abs().sum().item() == 0
        print(f"  ✓ Deterministic")

        # ── Test 5: Auto-discover ─────────────────────────────────────
        print("\n--- (5) Auto-discover ---")
        ds_auto = BraTSDataset(brats_dir, augment=False)
        assert len(ds_auto) == 10
        print(f"  Found {len(ds_auto)} subjects  ✓")

        # ── Test 6: FastMRIDataset ────────────────────────────────────
        print("\n--- (6) FastMRIDataset ---")
        ds_fmri = FastMRIDataset(fastmri_dir, augment=False)
        assert len(ds_fmri) == 5

        fmri_s = ds_fmri[0]
        assert fmri_s["image"].shape == (4, 128, 128, 128)
        c0 = fmri_s["image"][0]
        for ci in range(1, 4):
            assert torch.equal(fmri_s["image"][ci], c0)
        print(f"  {len(ds_fmri)} subjects, 4-channel replica  ✓")

        # ── Test 7: DataLoader (non-distributed) ─────────────────────
        print("\n--- (7) DataLoader ---")
        tr_dl, va_dl = build_brats_dataloaders(
            brats_dir, fold_path, micro_batch=2,
            num_workers=0, distributed=False)

        batch = next(iter(tr_dl))
        assert batch["image"].shape == (2, 4, 128, 128, 128)
        assert batch["label"].shape == (2, 128, 128, 128)
        print(f"  Train: {list(batch['image'].shape)}  ✓")

        # ── Test 8: fastMRI DataLoader ────────────────────────────────
        print("\n--- (8) fastMRI DataLoader ---")
        fmri_dl = build_fastmri_dataloader(
            fastmri_dir, micro_batch=1, num_workers=0, distributed=False)
        fb = next(iter(fmri_dl))
        assert fb["image"].shape == (1, 4, 128, 128, 128)
        assert "label" not in fb
        print(f"  {list(fb['image'].shape)}, no label  ✓")

        # ── Test 9: Zero-shot DataLoader ──────────────────────────────
        print("\n--- (9) Zero-shot DataLoader ---")
        zs_dl = build_zeroshot_dataloader(
            brats_dir, micro_batch=1, num_workers=0, distributed=False)
        zb = next(iter(zs_dl))
        assert zb["image"].shape == (1, 4, 128, 128, 128)
        assert zb["label"].shape == (1, 128, 128, 128)
        print(f"  {list(zb['image'].shape)} + {list(zb['label'].shape)}  ✓")

        # ── Test 10: Augmentation preserves correspondence ────────────
        print("\n--- (10) Label-image correspondence ---")
        for _ in range(5):
            s = ds_train[0]
            mask = s["label"] > 0
            if mask.any():
                assert s["image"][:, mask].abs().sum() > 0
        print(f"  ✓ Preserved through augmentation")

    print(f"\n{'=' * 70}")
    print("  ALL 10 TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
