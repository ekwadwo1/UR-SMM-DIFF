#!/usr/bin/env python3
"""
fig_restoration_2x5.py — Render Fig. 3 (2×5 restoration visualization)
========================================================================

Columns: Clean x₀, Corrupted y, Restored x̂₀, |x₀−x̂₀| error, Uncertainty σ²_θ
Rows: 2 representative BraTS 2021 axial slices

Yellow arrows link high-error regions to corresponding high-uncertainty
regions, demonstrating that the variance head identifies voxels with
limited restoration fidelity.

Usage:
  python fig_restoration_2x5.py --data-dir /path/to/fig_restoration
  python fig_restoration_2x5.py --data-dir /path/to/fig_restoration --output fig3.pdf
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
from scipy.ndimage import zoom, gaussian_filter, label as nd_label

# ── Crisp, bold text for TMI publication ──
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          16,
    'font.weight':        'bold',
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'text.usetex':        False,
    'mathtext.fontset':   'dejavusans',
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'savefig.dpi':        600,
    'figure.dpi':         150,
    'axes.linewidth':     1.0,
})

# ── Arrow styling (bright cyan — visible on hot, error, and gray cmaps) ──
ARROW_COLOR  = '#00FFFF'          # bright cyan
ARROW_EDGE   = '#005F5F'          # dark teal edge for contrast
ARROW_LW     = 3.0                # thick line
ARROW_SCALE  = 16                 # arrowhead size
ARROW_OFFSET = 20                 # px from tip to tail


# ======================================================================= #
#  Helpers                                                                 #
# ======================================================================= #

def crop_to_brain(img, pad=10):
    """Return crop box (y0, y1, x0, x1) for brain ROI."""
    mask = img > 0.01
    if mask.sum() == 0:
        return 0, img.shape[0], 0, img.shape[1]
    ys, xs = np.where(mask)
    return (max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad),
            max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad))


def find_high_regions(map_2d, brain_mask, n_peaks=3, min_sep=15,
                      sigma=3.0, percentile=92):
    """
    Find local peaks in a heatmap for arrow placement.
    Returns list of (y, x) coordinates of top-N peaks.
    """
    # Smooth for stable peak detection
    smoothed = gaussian_filter(map_2d * brain_mask, sigma=sigma)
    threshold = np.percentile(smoothed[brain_mask], percentile)
    peaks_mask = smoothed > threshold

    # Label connected components
    labeled, n_comp = nd_label(peaks_mask)
    if n_comp == 0:
        return []

    # Compute centroid of each component
    centroids = []
    for c in range(1, n_comp + 1):
        ys, xs = np.where(labeled == c)
        area = len(ys)
        if area < 5:
            continue
        cy, cx = ys.mean(), xs.mean()
        intensity = smoothed[labeled == c].mean()
        centroids.append((cy, cx, intensity, area))

    # Sort by intensity (descending)
    centroids.sort(key=lambda c: c[2], reverse=True)

    # Select peaks with minimum separation
    selected = []
    for cy, cx, _, _ in centroids:
        too_close = False
        for sy, sx in selected:
            if np.sqrt((cy - sy)**2 + (cx - sx)**2) < min_sep:
                too_close = True
                break
        if not too_close:
            selected.append((cy, cx))
        if len(selected) >= n_peaks:
            break

    return selected


# ======================================================================= #
#  Main rendering                                                          #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser("Render Fig. 3 — 2×5 Restoration Visualization")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="fig_restoration_2x5.pdf")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--contrast", type=int, default=3,
                        help="MRI contrast: 0=T1, 1=T1ce, 2=T2, 3=FLAIR")
    parser.add_argument("--n-arrows", type=int, default=3,
                        help="Number of correspondence arrows per row")
    args = parser.parse_args()

    d = args.data_dir
    c = args.contrast
    contrast_names = {0: "T1", 1: "T1ce", 2: "T2", 3: "FLAIR"}
    cname = contrast_names.get(c, f"ch{c}")

    # ── Load metadata ──
    meta_path = os.path.join(d, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        subjects = meta["subjects"]
    else:
        subjects = [{"row": 1, "sid": "row1", "slice": 64},
                    {"row": 2, "sid": "row2", "slice": 64}]

    n_rows = len(subjects)

    # ── Column definitions ──
    col_titles = [
        f'(a) Clean $x_0$ ({cname})',
        '(b) Corrupted $y$',
        '(c) Restored $\\hat{{x}}_0$',
        '(d) $|x_0 - \\hat{x}_0|$',
        '(e) Uncertainty $\\sigma^2_{\\theta}$',
    ]
    n_cols = 5

    # ── Error-map colormap (black → blue → cyan → yellow → white) ──
    error_cmap = LinearSegmentedColormap.from_list('error', [
        (0.0,  0.0,  0.05),   # near-black
        (0.05, 0.15, 0.45),   # dark blue
        (0.10, 0.50, 0.80),   # blue
        (0.30, 0.85, 0.95),   # cyan
        (0.95, 0.90, 0.20),   # yellow
        (1.0,  1.0,  1.0),    # white
    ], N=256)

    # ── Figure layout ──
    fig_w = 22.0
    fig_h = 4.2 * n_rows + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.03, hspace=0.10,
                           left=0.045, right=0.995,
                           top=1.0 - 0.50/fig_h,
                           bottom=0.50/fig_h)

    # Handles for colorbars
    im_error_handle = None
    im_var_handle = None

    for row_idx in range(n_rows):
        subj = subjects[row_idx]
        sl = subj["slice"]
        prefix = os.path.join(d, f"row{row_idx+1}")

        # ── Load arrays ──
        clean     = np.load(f"{prefix}_clean.npy")        # [4,128,128,128]
        corrupted = np.load(f"{prefix}_corrupted.npy")     # [4,128,128,128]
        restored  = np.load(f"{prefix}_restored.npy")      # [4,128,128,128]
        var_3d    = np.load(f"{prefix}_variance.npy")       # [32,32,32]

        # Extract axial slices for chosen contrast
        clean_sl     = clean[c, :, :, sl]
        corrupted_sl = corrupted[c, :, :, sl]
        restored_sl  = restored[c, :, :, sl]

        # Compute error map
        error_sl = np.abs(clean_sl - restored_sl)

        # Upsample variance to full resolution
        r = clean.shape[1] // var_3d.shape[0]
        var_up = zoom(var_3d, r, order=1)
        var_sl = var_up[:, :, sl]

        # Crop to brain ROI
        y0, y1, x0, x1 = crop_to_brain(clean_sl)
        cr = lambda a: a[y0:y1, x0:x1]

        # Brain mask for display
        brain_mask = clean_sl > 0.01
        brain_cr = cr(brain_mask)

        # Intensity range from clean image
        vmin = np.percentile(clean_sl[brain_mask], 1) if brain_mask.sum() else 0
        vmax = np.percentile(clean_sl[brain_mask], 99) if brain_mask.sum() else 1

        # Error and variance ranges (brain-masked)
        err_cr = cr(error_sl) * brain_cr
        var_cr = cr(var_sl) * brain_cr
        err_vmax = np.percentile(err_cr[brain_cr], 97) if brain_cr.sum() else 1
        var_vmax = np.percentile(var_cr[brain_cr], 97) if brain_cr.sum() else 1

        # ── Find arrow correspondence points ──
        # Peaks in error map that also have high uncertainty
        error_peaks = find_high_regions(err_cr, brain_cr,
                                        n_peaks=args.n_arrows, min_sep=18)
        var_peaks = find_high_regions(var_cr, brain_cr,
                                      n_peaks=args.n_arrows, min_sep=18)

        # ── (a) Clean x₀ ──
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(cr(clean_sl), cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=18, fontweight='bold', pad=12)

        # Row label
        sid_short = subj.get("sid", "").replace("BraTS2021_", "S")
        ax.text(-0.10, 0.5, f"Row {row_idx+1}\n({sid_short})",
                transform=ax.transAxes, fontsize=15, fontweight='bold',
                va='center', ha='right', rotation=90, color='0.15',
                linespacing=1.4)

        # ── (b) Corrupted y ──
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(cr(corrupted_sl), cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=18, fontweight='bold', pad=12)

        # ── (c) Restored x̂₀ ──
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(cr(restored_sl), cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(col_titles[2], fontsize=18, fontweight='bold', pad=12)

        # ── (d) Error map |x₀ − x̂₀| ──
        ax_err = fig.add_subplot(gs[row_idx, 3])
        err_display = np.where(brain_cr, err_cr, 0)
        im_err = ax_err.imshow(err_display, cmap=error_cmap, vmin=0,
                               vmax=err_vmax, interpolation='bilinear')
        if im_error_handle is None:
            im_error_handle = im_err

        # Draw arrows at high-error peaks (bright cyan with dark edge)
        for py, px in error_peaks:
            # Dark outline arrow (drawn first, slightly thicker)
            ax_err.annotate('',
                xy=(px, py),
                xytext=(px + ARROW_OFFSET, py - ARROW_OFFSET),
                arrowprops=dict(arrowstyle='->', color=ARROW_EDGE,
                                lw=ARROW_LW + 2, mutation_scale=ARROW_SCALE + 4))
            # Bright arrow on top
            ax_err.annotate('',
                xy=(px, py),
                xytext=(px + ARROW_OFFSET, py - ARROW_OFFSET),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=ARROW_LW, mutation_scale=ARROW_SCALE))

        ax_err.axis('off')
        if row_idx == 0:
            ax_err.set_title(col_titles[3], fontsize=18, fontweight='bold', pad=12)

        # ── (e) Uncertainty σ²_θ ──
        ax_var = fig.add_subplot(gs[row_idx, 4])
        var_display = np.where(brain_cr, var_cr, 0)
        im_var = ax_var.imshow(var_display, cmap='hot', vmin=0,
                               vmax=var_vmax, interpolation='bilinear')
        if im_var_handle is None:
            im_var_handle = im_var

        # Draw arrows at high-uncertainty peaks (same style for correspondence)
        for py, px in var_peaks:
            # Dark outline arrow
            ax_var.annotate('',
                xy=(px, py),
                xytext=(px + ARROW_OFFSET, py - ARROW_OFFSET),
                arrowprops=dict(arrowstyle='->', color=ARROW_EDGE,
                                lw=ARROW_LW + 2, mutation_scale=ARROW_SCALE + 4))
            # Bright arrow on top
            ax_var.annotate('',
                xy=(px, py),
                xytext=(px + ARROW_OFFSET, py - ARROW_OFFSET),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=ARROW_LW, mutation_scale=ARROW_SCALE))

        ax_var.axis('off')
        if row_idx == 0:
            ax_var.set_title(col_titles[4], fontsize=18, fontweight='bold', pad=12)

    # ── Colorbars ──
    # Error map colorbar (below column d)
    if im_error_handle is not None:
        cax1 = fig.add_axes([0.60, 0.020, 0.12, 0.020])
        cb1 = fig.colorbar(im_error_handle, cax=cax1, orientation='horizontal')
        cb1.set_ticks([])
        cb1.outline.set_linewidth(1.0)
        cb1.ax.text(0.0, -2.5, '0', fontsize=13, fontweight='bold',
                    ha='left', va='top', transform=cb1.ax.transAxes)
        cb1.ax.text(1.0, -2.5, 'High', fontsize=13, fontweight='bold',
                    ha='right', va='top', transform=cb1.ax.transAxes)
        cb1.ax.text(0.5, -2.5, '$|x_0 - \\hat{x}_0|$', fontsize=13,
                    fontweight='bold', ha='center', va='top',
                    transform=cb1.ax.transAxes)

    # Uncertainty colorbar (below column e)
    if im_var_handle is not None:
        cax2 = fig.add_axes([0.83, 0.020, 0.12, 0.020])
        cb2 = fig.colorbar(im_var_handle, cax=cax2, orientation='horizontal')
        cb2.set_ticks([])
        cb2.outline.set_linewidth(1.0)
        cb2.ax.text(0.0, -2.5, 'Low', fontsize=13, fontweight='bold',
                    ha='left', va='top', transform=cb2.ax.transAxes)
        cb2.ax.text(1.0, -2.5, 'High', fontsize=13, fontweight='bold',
                    ha='right', va='top', transform=cb2.ax.transAxes)
        cb2.ax.text(0.5, -2.5, '$\\sigma^2_{\\theta}$', fontsize=13,
                    fontweight='bold', ha='center', va='top',
                    transform=cb2.ax.transAxes)

    # ── Correspondence annotation ──
    fig.text(0.78, 1.0 - 0.50/fig_h - 0.02,
             '$\\longleftrightarrow$ spatial correspondence',
             fontsize=14, fontweight='bold', ha='center', va='top',
             color='#008B8B',
             bbox=dict(boxstyle='round,pad=0.3', fc='white',
                       alpha=0.90, ec='#008B8B', lw=1.8))

    # ── Save ──
    ext = args.output.split('.')[-1]
    fig.savefig(args.output, format=ext, dpi=args.dpi,
                bbox_inches='tight', pad_inches=0.05,
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nFigure saved: {args.output}  ({n_rows}×{n_cols} panels)")


if __name__ == "__main__":
    main()
