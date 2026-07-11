#!/usr/bin/env python3
"""
losses/segmentation_loss.py — Segmentation & Total Loss (Phase 7-B)
===================================================================

Paper Eqs. (20)–(22), Section 3.3:

  Segmentation loss (Eq. 20):
    L_seg = L_Dice(ŝ₀, s₀) + L_CE(ŝ₀, s₀)

  Total loss — Regime A (Eq. 21):
    L_total = L_diff + γ · L_seg
    γ ramps from 0 to γ_target over warmup steps (cosine schedule)

  Regime B (Eq. 22):
    L_total = L_seg( S_φ(stopgrad(ẑ₀)), s₀ )
    Denoiser θ frozen — receives NO gradient from seg loss.

BraTS label mapping:
  Raw labels: {0, 1, 2, 4} → Class indices: {0, 1, 2, 3}
  0 = Background, 1 = NCR/NET, 2 = ED, 4 = ET → 3

Hardware: 2× NVIDIA RTX 5880 Ada 48 GB · CUDA 11.8
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================= #
#  BraTS Label Mapping                                                     #
# ======================================================================= #

def map_brats_labels(raw: torch.Tensor) -> torch.Tensor:
    """
    Convert BraTS original labels {0, 1, 2, 4} to contiguous class
    indices {0, 1, 2, 3} for cross-entropy loss.

    Mapping:
      0 (Background) → 0
      1 (NCR/NET)     → 1
      2 (ED)          → 2
      4 (ET)          → 3

    Parameters
    ----------
    raw : Tensor of any shape with values in {0, 1, 2, 4}

    Returns
    -------
    mapped : same shape, values in {0, 1, 2, 3}  (dtype=long)
    """
    mapped = torch.zeros_like(raw, dtype=torch.long)
    mapped[raw == 1] = 1
    mapped[raw == 2] = 2
    mapped[raw == 4] = 3
    return mapped


# ======================================================================= #
#  Soft Dice Loss                                                          #
# ======================================================================= #

class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation (part of Eq. 20).

    Operates on softmax probabilities. Computes per-class Dice and
    averages (excluding background optionally).

    Parameters
    ----------
    n_classes         : number of classes (K=4 including background)
    smooth            : smoothing constant to avoid division by zero
    include_background: whether to include class 0 in the average
    """

    def __init__(
        self,
        n_classes: int = 4,
        smooth: float = 1.0,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, K, H, W, D]  raw logits (pre-softmax)
        target : [B, H, W, D]     class indices in {0, ..., K-1}

        Returns
        -------
        dice_loss : scalar   1 - mean(Dice_k)
        """
        B, K = logits.shape[:2]
        probs = F.softmax(logits, dim=1)                       # [B, K, H, W, D]

        # One-hot encode target: [B, H, W, D] → [B, K, H, W, D]
        target_oh = F.one_hot(target.long(), K)                # [B, H, W, D, K]
        target_oh = target_oh.permute(0, 4, 1, 2, 3).float()  # [B, K, H, W, D]

        # Flatten spatial dims
        probs_flat = probs.reshape(B, K, -1)                   # [B, K, N]
        target_flat = target_oh.reshape(B, K, -1)              # [B, K, N]

        # Per-class Dice
        intersection = (probs_flat * target_flat).sum(dim=2)   # [B, K]
        union = probs_flat.sum(dim=2) + target_flat.sum(dim=2) # [B, K]
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, K]

        # Average over classes
        start = 0 if self.include_background else 1
        dice_mean = dice[:, start:].mean()

        return 1.0 - dice_mean


# ======================================================================= #
#  SegmentationLoss (Eq. 20)                                               #
# ======================================================================= #

class SegmentationLoss(nn.Module):
    """
    Composite segmentation loss — Eq. (20):
      L_seg = L_Dice(ŝ₀, s₀) + L_CE(ŝ₀, s₀)

    Parameters
    ----------
    n_classes   : K=4 (BG, NCR/NET, ED, ET)
    ce_weights  : per-class weights for cross-entropy [0.1, 1.0, 1.0, 2.0]
    """

    def __init__(
        self,
        n_classes: int = 4,
        ce_weights: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        self.dice_loss = SoftDiceLoss(n_classes=n_classes, include_background=True)

        weights = ce_weights or (0.1, 1.0, 1.0, 2.0)
        self.register_buffer(
            "ce_weight", torch.tensor(weights, dtype=torch.float32))
        self.n_classes = n_classes

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        logits : [B, K, H, W, D]  raw segmentation logits
        target : [B, H, W, D]     class indices {0, ..., K-1}

        Returns
        -------
        l_seg : scalar   L_Dice + L_CE
        logs  : dict     component losses
        """
        l_dice = self.dice_loss(logits, target)

        l_ce = F.cross_entropy(
            logits, target.long(), weight=self.ce_weight)

        l_seg = l_dice + l_ce

        logs = {
            "l_dice": l_dice.item(),
            "l_ce": l_ce.item(),
            "l_seg": l_seg.item(),
        }

        return l_seg, logs


# ======================================================================= #
#  Gamma Scheduler (cosine ramp for Regime A)                              #
# ======================================================================= #

class GammaScheduler:
    """
    Cosine ramp for γ from 0 to γ_target over warmup_steps.

    Regime A (Eq. 21): γ is ramped gradually to preserve the generative
    prior during joint fine-tuning.

    Parameters
    ----------
    gamma_target  : final γ value (default 0.1)
    warmup_steps  : steps over which γ ramps from 0 (default 2000)
    """

    def __init__(
        self, gamma_target: float = 0.1, warmup_steps: int = 2000,
    ) -> None:
        self.gamma_target = gamma_target
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        """Return γ at the given training step."""
        if step >= self.warmup_steps:
            return self.gamma_target
        # Cosine ramp: 0 → γ_target
        progress = step / self.warmup_steps
        return self.gamma_target * 0.5 * (1.0 - math.cos(math.pi * progress))


# ======================================================================= #
#  TotalLoss (Eq. 21 / Eq. 22)                                            #
# ======================================================================= #

class TotalLoss(nn.Module):
    """
    Total training objective — Eqs. (21)–(22).

    Regime A (Eq. 21): L_total = L_diff + γ · L_seg
      γ cosine-ramped from 0 to γ_target.

    Regime B (Eq. 22): L_total = L_seg( S_φ(stopgrad(ẑ₀)), s₀ )
      Denoiser frozen — seg head trains alone.

    Parameters
    ----------
    gamma_target  : final γ for Regime A (default 0.1)
    warmup_steps  : γ ramp duration (default 2000)
    regime        : 'A' or 'B'
    """

    def __init__(
        self,
        n_classes: int = 4,
        ce_weights: Optional[Tuple[float, ...]] = None,
        gamma_target: float = 0.1,
        warmup_steps: int = 2000,
        regime: str = "A",
    ) -> None:
        super().__init__()
        self.seg_loss = SegmentationLoss(n_classes, ce_weights)
        self.gamma_sched = GammaScheduler(gamma_target, warmup_steps)
        self.regime = regime.upper()

    def forward(
        self,
        l_diff: Optional[torch.Tensor],
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        l_diff      : scalar diffusion loss (None for Regime B)
        seg_logits  : [B, K, H, W, D]  segmentation logits
        seg_target  : [B, H, W, D]     class indices
        global_step : current training step (for γ ramp)

        Returns
        -------
        l_total : scalar
        logs    : dict with all component losses + γ
        """
        l_seg, seg_logs = self.seg_loss(seg_logits, seg_target)

        if self.regime == "B":
            # Eq. 22: seg-only, denoiser frozen
            logs = {**seg_logs, "gamma": 0.0, "l_total": l_seg.item()}
            return l_seg, logs

        # Regime A: Eq. 21
        gamma = self.gamma_sched(global_step)
        l_total = l_diff + gamma * l_seg

        logs = {
            "l_diff": l_diff.item() if l_diff is not None else 0.0,
            **seg_logs,
            "gamma": gamma,
            "l_total": l_total.item(),
        }

        return l_total, logs


# ======================================================================= #
#  Tests  (Jupyter-safe — no argparse)                                     #
# ======================================================================= #

def _run_tests() -> None:
    """Comprehensive tests for segmentation and total loss."""

    print("=" * 70)
    print("  SegmentationLoss + TotalLoss — Test Suite (Eqs. 20–22)")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B, K, H, W, D = 2, 4, 16, 16, 16

    # ── Test 1: BraTS label mapping ───────────────────────────────────
    print("\n--- (1) BraTS label mapping {0,1,2,4} → {0,1,2,3} ---")
    raw = torch.tensor([0, 1, 2, 4, 0, 4, 2, 1])
    mapped = map_brats_labels(raw)
    expected = torch.tensor([0, 1, 2, 3, 0, 3, 2, 1])
    assert torch.equal(mapped, expected), f"Mapping failed: {mapped}"
    print(f"  {raw.tolist()} → {mapped.tolist()}  ✓")

    # Volume mapping
    raw_vol = torch.zeros(1, H, W, D, dtype=torch.long, device=device)
    raw_vol[0, :4, :, :] = 1
    raw_vol[0, 4:8, :, :] = 2
    raw_vol[0, 8:12, :, :] = 4
    mapped_vol = map_brats_labels(raw_vol)
    assert mapped_vol.max() == 3 and mapped_vol.min() == 0
    assert (mapped_vol[0, 8:12] == 3).all()  # ET → 3
    print(f"  Volume mapping: max={mapped_vol.max()}, unique={mapped_vol.unique().tolist()}  ✓")

    # ── Test 2: SegmentationLoss shape and output ─────────────────────
    print("\n--- (2) SegmentationLoss (Eq. 20) ---")
    seg_criterion = SegmentationLoss(n_classes=K).to(device)

    logits = torch.randn(B, K, H, W, D, device=device)
    target = torch.randint(0, K, (B, H, W, D), device=device)

    l_seg, logs = seg_criterion(logits, target)
    assert l_seg.shape == (), f"Not scalar: {l_seg.shape}"
    assert "l_dice" in logs and "l_ce" in logs
    print(f"  L_seg = {l_seg.item():.4f}  (dice={logs['l_dice']:.4f}, ce={logs['l_ce']:.4f})  ✓")

    # ── Test 3: Perfect prediction → low loss ─────────────────────────
    print("\n--- (3) Perfect prediction → low loss ---")
    target_perf = torch.randint(0, K, (B, H, W, D), device=device)
    # Create "perfect" logits: one-hot with large margin
    logits_perf = F.one_hot(target_perf, K).permute(0, 4, 1, 2, 3).float() * 10.0

    l_perf, _ = seg_criterion(logits_perf, target_perf)
    print(f"  L_seg(perfect) = {l_perf.item():.6f}  (should be near 0)")
    assert l_perf.item() < 0.1, f"Perfect prediction loss too high: {l_perf.item()}"
    print(f"  ✓ Near-zero loss for perfect prediction")

    # ── Test 4: Gradient flow through seg loss ────────────────────────
    print("\n--- (4) Gradient flow ---")
    logits_g = torch.randn(B, K, H, W, D, device=device, requires_grad=True)
    l_g, _ = seg_criterion(logits_g, target)
    l_g.backward()
    assert logits_g.grad is not None and logits_g.grad.abs().sum() > 0
    print(f"  Logits grad norm: {logits_g.grad.norm().item():.4f}  ✓")

    # ── Test 5: Gamma scheduler (cosine ramp) ─────────────────────────
    print("\n--- (5) Gamma scheduler (cosine ramp 0→0.1 over 2000 steps) ---")
    sched = GammaScheduler(gamma_target=0.1, warmup_steps=2000)

    g_0    = sched(0)
    g_500  = sched(500)
    g_1000 = sched(1000)
    g_2000 = sched(2000)
    g_5000 = sched(5000)

    print(f"  γ(0)    = {g_0:.6f}  (should be 0)")
    print(f"  γ(500)  = {g_500:.6f}  (should be ~0.015)")
    print(f"  γ(1000) = {g_1000:.6f}  (should be 0.05)")
    print(f"  γ(2000) = {g_2000:.6f}  (should be 0.1)")
    print(f"  γ(5000) = {g_5000:.6f}  (should be 0.1)")

    assert abs(g_0) < 1e-8, f"γ(0) should be 0: {g_0}"
    assert abs(g_1000 - 0.05) < 0.001, f"γ(1000) should be 0.05: {g_1000}"
    assert abs(g_2000 - 0.1) < 1e-6, f"γ(2000) should be 0.1: {g_2000}"
    assert abs(g_5000 - 0.1) < 1e-6, f"γ(5000) should be 0.1: {g_5000}"
    # Monotonically increasing
    gammas = [sched(s) for s in range(0, 2001, 100)]
    assert all(gammas[i] <= gammas[i+1] + 1e-8 for i in range(len(gammas)-1))
    print(f"  ✓ Cosine ramp correct and monotone")

    # ── Test 6: TotalLoss Regime A (Eq. 21) ───────────────────────────
    print("\n--- (6) TotalLoss Regime A (Eq. 21) ---")
    total_A = TotalLoss(n_classes=K, gamma_target=0.1,
                        warmup_steps=2000, regime="A").to(device)

    l_diff = torch.tensor(5.0, device=device, requires_grad=True)
    l_total_0, logs_0 = total_A(l_diff, logits, target, global_step=0)
    l_total_2k, logs_2k = total_A(l_diff, logits, target, global_step=2000)

    print(f"  Step 0:    γ={logs_0['gamma']:.4f}  L_total={logs_0['l_total']:.4f}")
    print(f"  Step 2000: γ={logs_2k['gamma']:.4f}  L_total={logs_2k['l_total']:.4f}")

    # At step 0, γ=0 → L_total = L_diff
    assert abs(logs_0['gamma']) < 1e-8, f"γ(0) not 0: {logs_0['gamma']}"
    assert abs(l_total_0.item() - l_diff.item()) < 0.01, \
        f"Step 0: L_total should ≈ L_diff"
    print(f"  ✓ Step 0: L_total ≈ L_diff (γ=0)")

    # At step 2000, γ=0.1 → L_total = L_diff + 0.1*L_seg
    assert abs(logs_2k['gamma'] - 0.1) < 1e-6
    expected_total = l_diff.item() + 0.1 * logs_2k['l_seg']
    assert abs(l_total_2k.item() - expected_total) < 0.01
    print(f"  ✓ Step 2000: L_total = L_diff + 0.1·L_seg")

    # ── Test 7: TotalLoss Regime B (Eq. 22) — stopgrad ────────────────
    print("\n--- (7) TotalLoss Regime B (Eq. 22) — stopgrad ---")
    total_B = TotalLoss(n_classes=K, regime="B").to(device)

    # Simulate: z_hat_0 comes from denoiser (has grad)
    z_hat_0 = torch.randn(B, 4, H, W, D, device=device, requires_grad=True)
    # Regime B: stopgrad before seg head
    z_detached = z_hat_0.detach()
    # Simple "seg head": just a 1x1 conv
    seg_head = nn.Conv3d(4, K, 1).to(device)
    seg_logits_B = seg_head(z_detached)

    l_total_B, logs_B = total_B(None, seg_logits_B, target, global_step=999)
    l_total_B.backward()

    # z_hat_0 should have NO gradient (detached)
    assert z_hat_0.grad is None, "FAIL: gradient leaked through stopgrad!"
    print(f"  z_hat_0.grad = None  ✓ (stopgrad working)")

    # seg_head should have gradient
    seg_head_grad = sum(p.grad.abs().sum().item() for p in seg_head.parameters()
                        if p.grad is not None)
    assert seg_head_grad > 0, "Seg head has no gradient!"
    print(f"  seg_head grad sum: {seg_head_grad:.4f}  ✓")
    print(f"  L_total_B = {l_total_B.item():.4f}  (seg-only, no L_diff)")

    # ── Test 8: Loss decreases when prediction improves ───────────────
    print("\n--- (8) Loss decreases with improving prediction ---")
    target_imp = torch.randint(0, K, (B, H, W, D), device=device)

    # Bad prediction: random logits
    logits_bad = torch.randn(B, K, H, W, D, device=device)
    l_bad, _ = seg_criterion(logits_bad, target_imp)

    # Good prediction: biased toward correct class
    logits_good = F.one_hot(target_imp, K).permute(0, 4, 1, 2, 3).float() * 3.0
    logits_good += 0.1 * torch.randn_like(logits_good)
    l_good, _ = seg_criterion(logits_good, target_imp)

    print(f"  L_seg(random):  {l_bad.item():.4f}")
    print(f"  L_seg(biased):  {l_good.item():.4f}")
    assert l_good.item() < l_bad.item(), "Good prediction should have lower loss!"
    print(f"  ✓ Better predictions → lower loss")

    # ── Test 9: CE weights applied correctly ──────────────────────────
    print("\n--- (9) CE class weights ---")
    print(f"  Weights: {seg_criterion.ce_weight.tolist()}")
    assert all(abs(a - b) < 1e-6 for a, b in
               zip(seg_criterion.ce_weight.tolist(), [0.1, 1.0, 1.0, 2.0]))

    # Demonstrate weight effect: misclassify ET vs misclassify BG
    # in a mixed-class volume where both classes are present
    target_mix = torch.zeros(1, H, W, D, dtype=torch.long, device=device)
    target_mix[0, :8, :, :] = 3                               # half ET
    # other half stays 0 (BG)

    # Logits that misclassify ET (predict class 0 everywhere)
    logits_miss_et = torch.zeros(1, K, H, W, D, device=device)
    logits_miss_et[:, 0, :, :, :] = 5.0                       # confident BG

    # Logits that misclassify BG (predict class 3 everywhere)
    logits_miss_bg = torch.zeros(1, K, H, W, D, device=device)
    logits_miss_bg[:, 3, :, :, :] = 5.0                       # confident ET

    # Both miss 50% of voxels, but missing ET (weight=2.0) should cost more
    # than missing BG (weight=0.1)
    l_miss_et, _ = seg_criterion(logits_miss_et, target_mix)
    l_miss_bg, _ = seg_criterion(logits_miss_bg, target_mix)
    print(f"  L_seg(miss ET, weight=2.0): {l_miss_et.item():.4f}")
    print(f"  L_seg(miss BG, weight=0.1): {l_miss_bg.item():.4f}")
    assert l_miss_et.item() > l_miss_bg.item(), \
        "Misclassifying ET (weight=2.0) should cost more than BG (weight=0.1)!"
    print(f"  ✓ ET errors (weight=2.0) penalised more than BG errors (weight=0.1)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ALL 9 TESTS PASSED ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    _run_tests()
