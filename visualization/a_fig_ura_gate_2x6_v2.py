#!/usr/bin/env python3
"""
fig_ura_gate_2x6.py — Render Fig. 6 (2×6 panel) from two subjects
===================================================================

Takes .npy arrays produced by extract_ura_figure_data.py for 2 subjects
and generates a publication-quality 2×6 panel figure.

Rows: 2 subjects (distinct clinical scenarios)
Cols: (a) Corrupted input, (b) Variance σ²_θ, (c) Gate g,
      (d) Seg w/o URA, (e) Seg w/ URA, (f) Ground truth

Usage:
  python fig_ura_gate_2x6.py \
      --dirs /path/to/sub1 /path/to/sub2 \
      --slices 64 64 \
      --contrast 3 \
      --row-labels "S00150" "S00294" \
      --output fig_ura_gate_2x6.pdf

# Pick 2 subjects with distinct clinical scenarios
# (e.g., one with ghosting artifacts, one with motion artifacts)

BASE=/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs

python fig_ura_gate_2x6.py \
    --dirs \
        ${BASE}/figures/ura_data/S00150 \
        ${BASE}/figures/ura_data/S00294 \
    --slices 64 64 \
    --contrast 3 \
    --row-labels "S00150" "S00294" \
    --output fig_ura_gate_2x6.pdf \
    --dpi 600
      
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

# ── Global rendering settings ──
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          12,
    'font.weight':        'bold',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'text.usetex':        False,
    'mathtext.fontset':   'dejavusans',
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'savefig.dpi':        600,
    'figure.dpi':         150,
    'axes.linewidth':     0.8,
})


def load_subject(data_dir, slice_idx, contrast):
    """Load all arrays for one subject and extract the axial slice."""
    corrupted = np.load(os.path.join(data_dir, "corrupted_input.npy"))
    clean     = np.load(os.path.join(data_dir, "clean_input.npy"))
    var_map   = np.load(os.path.join(data_dir, "variance_map.npy"))
    gate_map  = np.load(os.path.join(data_dir, "gate_map.npy"))
    seg_ura   = np.load(os.path.join(data_dir, "seg_with_ura.npy"))
    seg_no    = np.load(os.path.join(data_dir, "seg_without_ura.npy"))
    seg_gt    = np.load(os.path.join(data_dir, "seg_gt.npy"))

    s = slice_idx
    c = contrast

    # Upsample variance and gate from latent to full resolution
    r = corrupted.shape[1] // var_map.shape[0]
    var_up  = zoom(var_map, r, order=1)
    gate_up = zoom(gate_map, r, order=1)

    # Load metadata for subject ID
    meta_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        subject_id = meta.get("subject_id", os.path.basename(data_dir))
    else:
        subject_id = os.path.basename(data_dir)

    # Auto-find slice with most tumor if requested slice has little tumor
    gt_slice = seg_gt[:, :, s]
    if gt_slice.sum() < 50:
        tumor_per_slice = (seg_gt > 0).sum(axis=(0, 1))
        s = int(np.argmax(tumor_per_slice))
        gt_slice = seg_gt[:, :, s]

    return {
        "corr":     corrupted[c, :, :, s],
        "clean":    clean[c, :, :, s],
        "var":      var_up[:, :, s],
        "gate":     gate_up[:, :, s],
        "seg_ura":  seg_ura[:, :, s],
        "seg_no":   seg_no[:, :, s],
        "seg_gt":   gt_slice,
        "sid":      subject_id,
        "slice":    s,
    }


def seg_to_rgba(seg_2d, colors):
    """Convert segmentation map to RGBA overlay."""
    out = np.zeros((*seg_2d.shape, 4))
    for k in range(len(colors)):
        out[seg_2d == k] = colors[k]
    return out


def crop_to_brain(img, pad=8):
    """Return crop indices for brain ROI."""
    mask = img > 0.01
    if mask.sum() == 0:
        return 0, img.shape[0], 0, img.shape[1]
    ys, xs = np.where(mask)
    y0 = max(0, ys.min() - pad)
    y1 = min(img.shape[0], ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(img.shape[1], xs.max() + pad)
    return y0, y1, x0, x1


def main():
    parser = argparse.ArgumentParser("Render 2×6 URA Gate Figure")
    parser.add_argument("--dirs", nargs="+", required=True,
                        help="2 directories with extracted .npy arrays")
    parser.add_argument("--slices", nargs="+", type=int, default=None,
                        help="Axial slice per subject (default: auto-find best)")
    parser.add_argument("--contrast", type=int, default=3,
                        help="MRI contrast: 0=T1, 1=T1ce, 2=T2, 3=FLAIR")
    parser.add_argument("--output", type=str, default="fig_ura_gate_2x6.pdf")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--row-labels", nargs="+", default=None,
                        help="Row labels (e.g., 'S00150' 'S00294')")
    args = parser.parse_args()

    n_rows = len(args.dirs)
    assert n_rows == 2, f"Expected exactly 2 subject dirs, got {n_rows}"

    if args.slices is None:
        args.slices = [64] * n_rows
    assert len(args.slices) == n_rows

    # ── Segmentation overlay colors ──
    seg_colors = np.array([
        [0.0, 0.0, 0.0, 0.0],       # 0: background
        [0.18, 0.75, 0.22, 0.50],    # 1: edema (green)
        [1.0, 0.82, 0.0, 0.62],      # 2: ET (yellow)
        [0.88, 0.12, 0.12, 0.58],    # 3: necrotic (red)
    ])

    # ── Gate colormap ──
    gate_cmap = LinearSegmentedColormap.from_list('gate', [
        (0.03, 0.03, 0.18), (0.10, 0.18, 0.42), (0.25, 0.40, 0.65),
        (0.50, 0.68, 0.85), (0.80, 0.90, 0.97), (1.0, 1.0, 1.0)], N=256)

    # ── Load subjects ──
    subjects = []
    for i, d in enumerate(args.dirs):
        subj = load_subject(d, args.slices[i], args.contrast)
        subjects.append(subj)
        print(f"  Row {i}: {subj['sid']}  slice={subj['slice']}  "
              f"tumor_vox={subj['seg_gt'].sum()}")

    # ── Column titles ──
    col_titles = [
        '(a) Corrupted',
        r'(b) Variance $\sigma^2_{\theta}$',
        '(c) Gate $g$',
        '(d) w/o URA',
        '(e) w/ URA',
        '(f) Ground truth',
    ]

    # ── Figure layout (2 rows) ──
    fig_w = 19.0
    fig_h = 3.6 * n_rows + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    gs = gridspec.GridSpec(n_rows, 6, wspace=0.04, hspace=0.08,
                           left=0.06, right=0.995,
                           top=1.0 - 0.60 / fig_h,
                           bottom=0.65 / fig_h)

    im_var_ref = None

    for row, subj in enumerate(subjects):
        y0, y1, x0, x1 = crop_to_brain(subj["corr"])
        cr = lambda a: a[y0:y1, x0:x1]

        brain = subj["corr"] > 0.01
        vmin = np.percentile(subj["corr"][brain], 1) if brain.sum() > 0 else 0
        vmax = np.percentile(subj["corr"][brain], 99) if brain.sum() > 0 else 1

        # ── (a) Corrupted input ──
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(cr(subj["corr"]), cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[0], fontsize=15, fontweight="bold", pad=10)

        # Row label
        if args.row_labels and row < len(args.row_labels):
            label = args.row_labels[row]
        else:
            sid = subj["sid"]
            label = sid.replace("BraTS2021_", "").replace("BraTS", "")
        ax.text(-0.12, 0.5, label, transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='center', ha='right', rotation=90,
                color='0.15')

        # ── (b) Variance ──
        ax = fig.add_subplot(gs[row, 1])
        v_cr = cr(subj["var"])
        vp = np.percentile(v_cr[v_cr > 0], 96) if (v_cr > 0).sum() > 0 else 1
        im_var = ax.imshow(v_cr, cmap='hot', vmin=0, vmax=vp,
                           interpolation='bilinear')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[1], fontsize=15, fontweight="bold", pad=10)
            im_var_ref = im_var

        # ── (c) Gate ──
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(cr(subj["gate"]), cmap=gate_cmap, vmin=0, vmax=1,
                  interpolation='bilinear')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[2], fontsize=15, fontweight="bold", pad=10)

        # ── (d) Seg without URA ──
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(cr(subj["corr"]), cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.imshow(seg_to_rgba(cr(subj["seg_no"]), seg_colors),
                  interpolation='nearest')
        # Circle false-positive ET regions
        fp_et = cr((subj["seg_no"] == 2) & (subj["seg_gt"] != 2))
        if fp_et.sum() > 10:
            fp_ys, fp_xs = np.where(fp_et)
            cy, cx = fp_ys.mean(), fp_xs.mean()
            rad = max(10, np.sqrt(fp_et.sum() / np.pi) + 5)
            circ = plt.Circle((cx, cy), rad, color='white', fill=False,
                              lw=2.5, ls='--')
            ax.add_patch(circ)
            ax.text(cx + rad + 3, cy, 'FP', color='white', fontsize=13,
                    fontweight='bold', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black',
                              alpha=0.7, ec='none'))
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[3], fontsize=15, fontweight="bold", pad=10)

        # ── (e) Seg with URA ──
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(cr(subj["corr"]), cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.imshow(seg_to_rgba(cr(subj["seg_ura"]), seg_colors),
                  interpolation='nearest')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[4], fontsize=15, fontweight="bold", pad=10)

        # ── (f) Ground truth ──
        ax = fig.add_subplot(gs[row, 5])
        ax.imshow(cr(subj["clean"]), cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.imshow(seg_to_rgba(cr(subj["seg_gt"]), seg_colors),
                  interpolation='nearest')
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[5], fontsize=15, fontweight="bold", pad=10)

    # ── Legend ──
    legend_patches = [
        mpatches.Patch(fc=seg_colors[1][:3], alpha=0.7, label='Edema (ED)'),
        mpatches.Patch(fc=seg_colors[2][:3], alpha=0.7, label='Enhancing (ET)'),
        mpatches.Patch(fc=seg_colors[3][:3], alpha=0.7, label='Necrotic (NCR)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=13, frameon=True, fancybox=False, edgecolor='0.65',
               bbox_to_anchor=(0.5, 0.002), handlelength=1.5,
               handletextpad=0.5, columnspacing=2.0)

    # ── Colorbars (below row 0) ──
    cb_y = 1.0 - 0.55 / fig_h - 3.6 / fig_h + 0.015

    # Variance colorbar
    if im_var_ref is not None:
        cax1 = fig.add_axes([0.21, cb_y, 0.11, 0.025])
        cb1 = fig.colorbar(im_var_ref, cax=cax1, orientation='horizontal')
        cb1.set_ticks([])
        cb1.outline.set_linewidth(1.0)
        cb1.ax.text(0.0, -2.2, 'Low', fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=cb1.ax.transAxes)
        cb1.ax.text(1.0, -2.2, 'High', fontsize=11, fontweight='bold',
                    ha='right', va='top', transform=cb1.ax.transAxes)

    # Gate colorbar
    sm = plt.cm.ScalarMappable(cmap=gate_cmap, norm=plt.Normalize(0, 1))
    cax2 = fig.add_axes([0.375, cb_y, 0.11, 0.025])
    cb2 = fig.colorbar(sm, cax=cax2, orientation='horizontal')
    cb2.set_ticks([0, 1])
    cb2.set_ticklabels(['0 (suppress)', '1 (pass)'], fontsize=10,
                        fontweight='bold')
    cb2.ax.tick_params(length=3, pad=2, width=1.0)
    cb2.outline.set_linewidth(1.0)

    # ── Save ──
    ext = args.output.split('.')[-1]
    fig.savefig(args.output, format=ext, dpi=args.dpi,
                bbox_inches='tight', pad_inches=0.04,
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nFigure saved: {args.output}  ({n_rows}×6 panels)")


if __name__ == "__main__":
    main()
