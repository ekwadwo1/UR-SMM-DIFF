#!/usr/bin/env python3
"""
fig_seg_comparison_3x7.py — Render Fig. 2 (3×7 segmentation comparison grid)
=============================================================================

Rows:  (i) Large WT, (ii) Thin ET rim, (iii) Heavy ghosting artifact
Cols:  FLAIR, T1ce, Ground truth, nnU-Net, Std. 3D DDPM,
       UR-SSM-Diff, Uncertainty σ²_θ

Usage:
  python fig_seg_comparison_3x7.py --data-dir /path/to/fig_seg_comparison
  python fig_seg_comparison_3x7.py --data-dir /path/to/fig_seg_comparison --output fig2.pdf
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

# ── Global rendering settings for crisp, bold text ──
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          14,
    'font.weight':        'bold',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'text.usetex':        False,
    'mathtext.fontset':   'dejavusans',
    'pdf.fonttype':       42,       # TrueType in PDF — no bitmap blur
    'ps.fonttype':        42,
    'savefig.dpi':        600,
    'figure.dpi':         150,
    'axes.linewidth':     0.8,
})


# ======================================================================= #
#  Helpers                                                                 #
# ======================================================================= #

# Segmentation RGBA colors (mapped labels: 0=BG, 1=ED, 2=NCR, 3=ET)
# Paper caption says: red=ET, green=ED, yellow=NCR
SEG_COLORS = np.array([
    [0.0,  0.0,  0.0,  0.0 ],    # 0: background (transparent)
    [0.15, 0.72, 0.20, 0.55],    # 1: edema (ED) — green
    [1.0,  0.85, 0.0,  0.60],    # 2: necrotic (NCR) — yellow
    [0.92, 0.10, 0.10, 0.60],    # 3: enhancing (ET) — red
])


def seg_to_rgba(seg_2d):
    """Convert 2D segmentation map to RGBA overlay."""
    out = np.zeros((*seg_2d.shape, 4))
    for k in range(len(SEG_COLORS)):
        out[seg_2d == k] = SEG_COLORS[k]
    return out


def crop_to_brain(img, pad=10):
    """Return crop box (y0, y1, x0, x1) for brain ROI."""
    mask = img > 0.01
    if mask.sum() == 0:
        return 0, img.shape[0], 0, img.shape[1]
    ys, xs = np.where(mask)
    y0 = max(0, ys.min() - pad)
    y1 = min(img.shape[0], ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(img.shape[1], xs.max() + pad)
    return y0, y1, x0, x1


def add_arrow(ax, y, x, length=14, color='yellow', lw=1.8):
    """Add a small annotation arrow pointing at (y, x) in image coords."""
    ax.annotate('', xy=(x, y), xytext=(x + length, y - length),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ======================================================================= #
#  Main rendering                                                          #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser("Render Fig. 2 — 3×7 Segmentation Comparison")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with extracted row*_*.npy arrays")
    parser.add_argument("--output", type=str, default="fig_seg_comparison_3x7.pdf")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--n-rows", type=int, default=3)
    args = parser.parse_args()

    d = args.data_dir
    n_rows = args.n_rows

    # ── Load metadata ──
    meta_path = os.path.join(d, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        subjects = meta["subjects"]
    else:
        # Fallback: infer from files
        subjects = [{"row": i+1, "sid": f"row{i+1}", "slice": 64,
                      "row_label": f"Row {i+1}"} for i in range(n_rows)]

    # ── Column definitions ──
    col_keys = ["flair", "t1ce", "gt", "nnunet", "ddpm3d", "ours", "variance"]
    col_titles = [
        "(a) FLAIR",
        "(b) T1ce",
        "(c) Ground truth",
        "(d) nnU-Net",
        "(e) Std. 3D DDPM",
        "(f) UR-SSM-Diff",
        "(g) Uncertainty $\\sigma^2_{\\theta}$",
    ]
    n_cols = len(col_titles)

    # ── Figure dimensions ──
    fig_w = 22.0
    fig_h = 3.8 * n_rows + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.08,
                           left=0.055, right=0.995,
                           top=1.0 - 0.55/fig_h,
                           bottom=0.55/fig_h)

    # Store variance imshow handle for colorbar
    im_var_handle = None

    for row_idx in range(n_rows):
        subj = subjects[row_idx]
        sl = subj["slice"]
        prefix = os.path.join(d, f"row{row_idx+1}")

        # ── Load arrays ──
        img = np.load(f"{prefix}_image.npy")                 # [4,128,128,128]
        seg_gt = np.load(f"{prefix}_seg_gt.npy")             # [128,128,128]
        seg_ours = np.load(f"{prefix}_seg_ours.npy")         # [128,128,128]
        seg_nnunet = np.load(f"{prefix}_seg_nnunet.npy")     # [128,128,128]
        seg_ddpm3d = np.load(f"{prefix}_seg_ddpm3d.npy")     # [128,128,128]
        var_3d = np.load(f"{prefix}_variance.npy")           # [32,32,32]

        # Auto-find best slice if current slice has no tumor
        gt_slice_check = seg_gt[:, :, sl]
        if gt_slice_check.sum() < 30:
            tumor_per_slice = (seg_gt > 0).sum(axis=(0, 1))
            sl = int(np.argmax(tumor_per_slice))

        # Extract axial slices
        flair_sl = img[3, :, :, sl]                           # FLAIR = channel 3
        t1ce_sl  = img[1, :, :, sl]                           # T1ce  = channel 1
        gt_sl     = seg_gt[:, :, sl]
        ours_sl   = seg_ours[:, :, sl]
        nnunet_sl = seg_nnunet[:, :, sl]
        ddpm3d_sl = seg_ddpm3d[:, :, sl]

        # Upsample variance to full resolution and extract slice
        r = img.shape[1] // var_3d.shape[0]
        var_up = zoom(var_3d, r, order=1)                     # [128,128,128]
        var_sl = var_up[:, :, sl]                              # [128,128]

        # Crop to brain ROI
        y0, y1, x0, x1 = crop_to_brain(flair_sl)
        cr = lambda a: a[y0:y1, x0:x1]

        # Intensity ranges
        brain_mask = flair_sl > 0.01
        flair_vmin = np.percentile(flair_sl[brain_mask], 1) if brain_mask.sum() else 0
        flair_vmax = np.percentile(flair_sl[brain_mask], 99) if brain_mask.sum() else 1

        brain_mask2 = t1ce_sl > 0.01
        t1ce_vmin = np.percentile(t1ce_sl[brain_mask2], 1) if brain_mask2.sum() else 0
        t1ce_vmax = np.percentile(t1ce_sl[brain_mask2], 99) if brain_mask2.sum() else 1

        # ── (a) FLAIR ──
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(cr(flair_sl), cmap='gray', vmin=flair_vmin, vmax=flair_vmax,
                  interpolation='bilinear')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=15, fontweight='bold', pad=10)

        # Row label
        row_label = subj.get("row_label", f"Row {row_idx+1}")
        ax.text(-0.12, 0.5, f"Row {row_idx+1}\n{row_label}",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='center', ha='right', rotation=90, color='0.15',
                linespacing=1.4)

        # ── (b) T1ce ──
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(cr(t1ce_sl), cmap='gray', vmin=t1ce_vmin, vmax=t1ce_vmax,
                  interpolation='bilinear')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=15, fontweight='bold', pad=10)

        # ── (c) Ground truth ──
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(cr(flair_sl), cmap='gray', vmin=flair_vmin, vmax=flair_vmax,
                  interpolation='bilinear')
        ax.imshow(seg_to_rgba(cr(gt_sl)), interpolation='nearest')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[2], fontsize=15, fontweight='bold', pad=10)

        # ── (d) nnU-Net ──
        ax = fig.add_subplot(gs[row_idx, 3])
        ax.imshow(cr(flair_sl), cmap='gray', vmin=flair_vmin, vmax=flair_vmax,
                  interpolation='bilinear')
        ax.imshow(seg_to_rgba(cr(nnunet_sl)), interpolation='nearest')

        # Row 1: yellow arrows for under-segmented edema
        if row_idx == 0:
            # Find edema voxels in GT but missing in nnU-Net
            missed_ed = cr((gt_sl == 1) & (nnunet_sl == 0))
            if missed_ed.sum() > 20:
                ys_m, xs_m = np.where(missed_ed)
                # Pick 2 representative arrow locations
                n_arrows = min(2, len(ys_m) // 10 + 1)
                indices = np.linspace(0, len(ys_m)-1, n_arrows+2, dtype=int)[1:-1]
                for idx in indices:
                    add_arrow(ax, ys_m[idx], xs_m[idx], length=12,
                              color='#FFD700', lw=2.0)

        # Row 2: highlight missed thin ET rim
        if row_idx == 1:
            missed_et = cr((gt_sl == 3) & (nnunet_sl != 3))
            if missed_et.sum() > 5:
                ys_m, xs_m = np.where(missed_et)
                cy, cx = ys_m.mean(), xs_m.mean()
                # Dashed circle around missed region
                rad = max(8, np.sqrt(missed_et.sum() / np.pi) + 4)
                circ = plt.Circle((cx, cy), rad, color='#FFD700',
                                  fill=False, lw=2.0, ls='--')
                ax.add_patch(circ)

        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[3], fontsize=15, fontweight='bold', pad=10)

        # ── (e) Std. 3D DDPM ──
        ax = fig.add_subplot(gs[row_idx, 4])
        ax.imshow(cr(flair_sl), cmap='gray', vmin=flair_vmin, vmax=flair_vmax,
                  interpolation='bilinear')
        ax.imshow(seg_to_rgba(cr(ddpm3d_sl)), interpolation='nearest')

        # Row 2: circle missed ET rim
        if row_idx == 1:
            missed_et = cr((gt_sl == 3) & (ddpm3d_sl != 3))
            if missed_et.sum() > 5:
                ys_m, xs_m = np.where(missed_et)
                cy, cx = ys_m.mean(), xs_m.mean()
                rad = max(8, np.sqrt(missed_et.sum() / np.pi) + 4)
                circ = plt.Circle((cx, cy), rad, color='#FFD700',
                                  fill=False, lw=2.0, ls='--')
                ax.add_patch(circ)

        # Row 3: highlight fragmented WT boundary
        if row_idx == 2:
            # Find disconnected WT regions in prediction
            from scipy.ndimage import label as nd_label
            wt_pred = (ddpm3d_sl > 0).astype(np.uint8)
            wt_cr = cr(wt_pred)
            labeled, n_components = nd_label(wt_cr)
            if n_components > 2:
                ax.text(0.03, 0.95, f'{n_components} fragments',
                        transform=ax.transAxes, fontsize=10,
                        fontweight='bold', color='#FFD700', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black',
                                  alpha=0.6, ec='none'))

        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[4], fontsize=15, fontweight='bold', pad=10)

        # ── (f) UR-SSM-Diff (ours) ──
        ax = fig.add_subplot(gs[row_idx, 5])
        ax.imshow(cr(flair_sl), cmap='gray', vmin=flair_vmin, vmax=flair_vmax,
                  interpolation='bilinear')
        ax.imshow(seg_to_rgba(cr(ours_sl)), interpolation='nearest')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[5], fontsize=15, fontweight='bold', pad=10)

        # ── (g) Uncertainty σ²_θ ──
        ax = fig.add_subplot(gs[row_idx, 6])
        v_cr = cr(var_sl)
        # Mask background for cleaner visualization
        brain_cr = cr(flair_sl) > 0.01
        v_display = np.where(brain_cr, v_cr, 0)
        vp = np.percentile(v_display[v_display > 0], 97) if (v_display > 0).sum() else 1
        im_var = ax.imshow(v_display, cmap='hot', vmin=0, vmax=vp,
                           interpolation='bilinear')
        if im_var_handle is None:
            im_var_handle = im_var
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[6], fontsize=15, fontweight='bold', pad=10)

    # ── Legend (bottom center) ──
    legend_patches = [
        mpatches.Patch(fc=SEG_COLORS[3][:3], alpha=0.75, label='ET'),
        mpatches.Patch(fc=SEG_COLORS[1][:3], alpha=0.75, label='ED'),
        mpatches.Patch(fc=SEG_COLORS[2][:3], alpha=0.75, label='NCR'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=14, prop={'size': 14, 'weight': 'bold'},
               frameon=True, fancybox=False, edgecolor='0.6',
               bbox_to_anchor=(0.42, 0.001),
               handlelength=1.6, handletextpad=0.5, columnspacing=2.5)

    # ── Variance colorbar (bottom right, aligned with column 7) ──
    if im_var_handle is not None:
        cax = fig.add_axes([0.88, 0.025, 0.08, 0.018])
        cb = fig.colorbar(im_var_handle, cax=cax, orientation='horizontal')
        cb.set_ticks([])
        cb.outline.set_linewidth(1.0)
        cb.ax.text(0.0, -2.5, 'Low', fontsize=11, fontweight='bold',
                   ha='left', va='top', transform=cb.ax.transAxes)
        cb.ax.text(1.0, -2.5, 'High', fontsize=11, fontweight='bold',
                   ha='right', va='top', transform=cb.ax.transAxes)

    # ── Save ──
    ext = args.output.split('.')[-1]
    fig.savefig(args.output, format=ext, dpi=args.dpi,
                bbox_inches='tight', pad_inches=0.05,
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nFigure saved: {args.output}  ({n_rows}×{n_cols} panels)")


if __name__ == "__main__":
    main()
