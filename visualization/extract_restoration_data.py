#!/usr/bin/env python3
"""
extract_restoration_data.py — Extract arrays for Fig. 3 (2×5 restoration vis)
===============================================================================

Selects 2 representative BraTS2021 subjects and saves:
  - Clean x₀, Corrupted y, Restored x̂₀, Variance σ²_θ
  (Error map |x₀ - x̂₀| is computed at render time)

Subject selection:
  Row 1: Subject with strong motion artifact AND visible tumor
  Row 2: Subject with ghosting artifact overlapping tumor boundary

Usage:
  python extract_restoration_data.py --fold 0
  python extract_restoration_data.py --fold 0 --subjects BraTS2021_00097 BraTS2021_00134
"""

from __future__ import annotations
import argparse, json, os, sys, logging
import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ur_ssm_diff import build_ur_ssm_diff
from losses.segmentation_loss import map_brats_labels

logger = logging.getLogger("extract_restore")

DRIVE_ROOT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS"
DEFAULTS = {
    "data_root":    os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs/preprocessed/BraTS2021"),
    "vqgan_ckpt":   os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"),
    "training_dir": os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs/training"),
    "output_dir":   os.path.join(DRIVE_ROOT, "UR_SSM_Diff_Outputs/figures/fig_restoration"),
}


def load_model(vqgan_ckpt, model_ckpt, device="cuda:0"):
    model = build_ur_ssm_diff(
        vqgan_ckpt=vqgan_ckpt, latent_dim=4, d_h=128,
        downsample_factor=4, n_classes=4, T=1000, device=device)
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=True)
    if "denoiser" in ckpt:
        model.denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    if "seg_head" in ckpt:
        model.seg_head.load_state_dict(ckpt["seg_head"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def run_restoration(model, x_0, device="cuda:0", seed=42,
                    timesteps=None, M=5):
    """Run Tweedie multi-timestep restoration. Returns restored volume + variance."""
    if timesteps is None:
        timesteps = [100, 200, 300, 400, 500]

    torch.manual_seed(seed)
    with autocast("cuda", dtype=torch.bfloat16):
        y = model.corruption_op(x_0)

    z_obs = model.vqgan(y, mode="encode")
    z_sum = torch.zeros_like(z_obs)
    v_sum = torch.zeros(1, 32*32*32, device=device)

    for t_val in timesteps:
        t = torch.tensor([t_val], device=device, dtype=torch.long)
        ab_t = model.diffusion.alpha_bars[t_val].to(device)
        eps = torch.randn_like(z_obs)
        z_t = torch.sqrt(ab_t) * z_obs + torch.sqrt(1 - ab_t) * eps

        with autocast("cuda", dtype=torch.bfloat16):
            eps_theta, v_theta = model.denoiser(z_t, t, z_obs)

        z_0_hat = (z_t - torch.sqrt(1 - ab_t) * eps_theta) / torch.sqrt(ab_t)
        v_tilde = v_theta.clamp(model.log_var_min, model.log_var_max)
        z_sum += z_0_hat
        v_sum += v_tilde

    z_0_avg = z_sum / M
    x_restored = model.vqgan(z_0_avg, mode="decode")
    variance = torch.exp(v_sum / M)[0].float().cpu().numpy().reshape(32, 32, 32)

    return (y[0].float().cpu().numpy(),
            x_restored[0].float().cpu().numpy(),
            variance)


def screen_subjects(data_root, fold, model, n_screen, device):
    """Screen candidates and rank by artifact intensity + tumor presence."""
    fold_json = os.path.join(data_root, f"fold_{fold}.json")
    with open(fold_json) as f:
        val_ids = sorted(json.load(f).get("val", []))

    results = []
    n = min(n_screen, len(val_ids))
    logger.info(f"Screening {n} candidates...")

    for i, sid in enumerate(val_ids[:n]):
        img_path = os.path.join(data_root, f"{sid}_image.npy")
        lbl_path = os.path.join(data_root, f"{sid}_label.npy")
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            continue

        x_0 = torch.from_numpy(np.load(img_path)).float().unsqueeze(0).to(device)
        seg = np.load(lbl_path)
        tumor_vol = (seg > 0).sum()
        if tumor_vol < 500:
            continue

        with autocast("cuda", dtype=torch.bfloat16):
            y = model.corruption_op(x_0)
        artifact = (y - x_0).abs().mean().item()

        # Best slice
        seg_mapped = seg.copy(); seg_mapped[seg == 4] = 3
        tumor_per_slice = (seg_mapped > 0).sum(axis=(0, 1))
        best_slice = int(np.argmax(tumor_per_slice))

        results.append({
            "sid": sid, "artifact": artifact,
            "tumor_vol": int(tumor_vol), "best_slice": best_slice,
            "score": artifact * np.log1p(tumor_per_slice[best_slice]),
        })

        if (i + 1) % 20 == 0:
            logger.info(f"  [{i+1}/{n}]")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser("Extract Fig. 3 restoration data")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--subjects", nargs=2, default=None)
    parser.add_argument("--slices", nargs=2, type=int, default=None)
    parser.add_argument("--n-screen", type=int, default=60)
    parser.add_argument("--data-root", default=DEFAULTS["data_root"])
    parser.add_argument("--vqgan-ckpt", default=DEFAULTS["vqgan_ckpt"])
    parser.add_argument("--training-dir", default=DEFAULTS["training_dir"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(args.output_dir, exist_ok=True)

    model_ckpt = os.path.join(
        args.training_dir, f"fold_{args.fold}", "phase3", "best_model.pt")
    model = load_model(args.vqgan_ckpt, model_ckpt, args.device)

    # Select subjects
    if args.subjects:
        selected = []
        for i, sid in enumerate(args.subjects):
            seg = np.load(os.path.join(args.data_root, f"{sid}_label.npy"))
            seg_m = seg.copy(); seg_m[seg == 4] = 3
            tps = (seg_m > 0).sum(axis=(0, 1))
            sl = args.slices[i] if args.slices else int(np.argmax(tps))
            selected.append({"sid": sid, "best_slice": sl})
    else:
        ranked = screen_subjects(args.data_root, args.fold, model,
                                 args.n_screen, args.device)
        selected = ranked[:2]
        logger.info(f"Selected: {selected[0]['sid']} and {selected[1]['sid']}")
        if args.slices:
            for i, s in enumerate(args.slices):
                selected[i]["best_slice"] = s

    # Process each subject
    metadata = {"subjects": [], "fold": args.fold, "seed": args.seed}

    for row_idx, subj in enumerate(selected):
        sid = subj["sid"]
        sl = subj["best_slice"]
        logger.info(f"\nRow {row_idx+1}: {sid}  slice={sl}")

        img = np.load(os.path.join(args.data_root, f"{sid}_image.npy"))
        x_0 = torch.from_numpy(img).float().unsqueeze(0).to(args.device)

        torch.manual_seed(args.seed + row_idx)
        corrupted, restored, variance = run_restoration(
            model, x_0, args.device, args.seed + row_idx)

        prefix = os.path.join(args.output_dir, f"row{row_idx+1}")
        np.save(f"{prefix}_clean.npy", img)             # [4,128,128,128]
        np.save(f"{prefix}_corrupted.npy", corrupted)    # [4,128,128,128]
        np.save(f"{prefix}_restored.npy", restored)      # [4,128,128,128]
        np.save(f"{prefix}_variance.npy", variance)      # [32,32,32]

        metadata["subjects"].append({
            "row": row_idx + 1, "sid": sid, "slice": sl,
        })
        logger.info(f"  Saved {prefix}_*.npy")

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    del model; torch.cuda.empty_cache()
    logger.info(f"\nDone. Render with:")
    logger.info(f"  python fig_restoration_2x5.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
