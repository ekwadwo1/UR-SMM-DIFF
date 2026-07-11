#!/usr/bin/env python3
"""
fig_failure_case.py — Render Fig. 7 (Failure case: thin ET rim)
================================================================

Layout: 1×4 panels with zoomed insets
  (a) T1ce input — shows enhancing rim
  (b) Ground truth overlay — thin ET rim highlighted
  (c) UR-SSM-Diff prediction — under-segmented ET + DSC_ET annotation
  (d) Uncertainty σ²_θ — arrows at missed ET boundary

Usage:
  python fig_failure_case.py --data-dir /path/to/fig_failure_case
  python fig_failure_case.py --data-dir /path/to/fig_failure_case --output fig7.pdf
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
from matplotlib.patches import FancyArrowPatch, Rectangle, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom, binary_dilation, binary_erosion, label as nd_label
from scipy.ndimage import gaussian_filter

# ── Crisp, bold text for TMI ──
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          14,
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


# ======================================================================= #
#  Segmentation colors                                                     #
# ======================================================================= #

SEG_COLORS = np.array([
    [0.0,  0.0,  0.0,  0.0 ],    # 0: BG
    [0.15, 0.72, 0.20, 0.55],    # 1: ED — green
    [1.0,  0.85, 0.0,  0.60],    # 2: NCR — yellow
    [0.92, 0.10, 0.10, 0.65],    # 3: ET — red
])

# Brighter version for zoomed insets
SEG_COLORS_BRIGHT = np.array([
    [0.0,  0.0,  0.0,  0.0 ],
    [0.15, 0.80, 0.20, 0.70],
    [1.0,  0.85, 0.0,  0.75],
    [0.95, 0.08, 0.08, 0.80],
])


def seg_to_rgba(seg_2d, colors=None):
    if colors is None:
        colors = SEG_COLORS
    out = np.zeros((*seg_2d.shape, 4))
    for k in range(len(colors)):
        out[seg_2d == k] = colors[k]
    return out


def crop_to_brain(img, pad=10):
    mask = img > 0.01
    if mask.sum() == 0:
        return 0, img.shape[0], 0, img.shape[1]
    ys, xs = np.where(mask)
    return (max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad),
            max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad))


def find_et_roi(seg_2d, pad=12):
    """Find bounding box around ET (class 3) region for zoom inset."""
    et = (seg_2d == 3)
    if et.sum() < 3:
        # Fallback: use any tumor
        et = (seg_2d > 0)
    if et.sum() < 3:
        h, w = seg_2d.shape
        return h//4, 3*h//4, w//4, 3*w//4

    ys, xs = np.where(et)
    cy, cx = int(ys.mean()), int(xs.mean())

    # Fixed-size zoom box centered on ET centroid
    box_half = 22
    y0 = max(0, cy - box_half - pad)
    y1 = min(seg_2d.shape[0], cy + box_half + pad)
    x0 = max(0, cx - box_half - pad)
    x1 = min(seg_2d.shape[1], cx + box_half + pad)
    return y0, y1, x0, x1


def find_missed_et_peaks(seg_gt_2d, seg_pred_2d, var_2d, brain_mask,
                         n_peaks=3, min_sep=10):
    """Find locations where ET is in GT but missed in prediction,
    AND uncertainty is high."""
    missed_et = (seg_gt_2d == 3) & (seg_pred_2d != 3) & brain_mask
    if missed_et.sum() < 3:
        return []

    # Weight by variance at missed locations
    weighted = gaussian_filter(missed_et.astype(float) * var_2d, sigma=2.5)
    threshold = np.percentile(weighted[weighted > 0], 75) if (weighted > 0).sum() else 0

    peaks_mask = weighted > threshold
    labeled, n_comp = nd_label(peaks_mask)
    if n_comp == 0:
        return []

    centroids = []
    for c in range(1, n_comp + 1):
        ys, xs = np.where(labeled == c)
        if len(ys) < 2:
            continue
        centroids.append((ys.mean(), xs.mean(), weighted[labeled == c].mean()))

    centroids.sort(key=lambda c: c[2], reverse=True)

    selected = []
    for cy, cx, _ in centroids:
        too_close = any(np.sqrt((cy-sy)**2 + (cx-sx)**2) < min_sep
                        for sy, sx in selected)
        if not too_close:
            selected.append((cy, cx))
        if len(selected) >= n_peaks:
            break
    return selected


# ======================================================================= #
#  Main rendering                                                          #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser("Render Fig. 7 — Failure Case")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="fig_failure_case.pdf")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--contrast", type=int, default=1,
                        help="0=T1, 1=T1ce (default, best for ET), 2=T2, 3=FLAIR")
    parser.add_argument("--n-arrows", type=int, default=3)
    args = parser.parse_args()

    d = args.data_dir
    c = args.contrast
    cnames = {0: "T1", 1: "T1ce", 2: "T2", 3: "FLAIR"}
    cname = cnames.get(c, f"ch{c}")

    # ── Load data ──
    img       = np.load(os.path.join(d, "image.npy"))       # [4,128,128,128]
    seg_gt    = np.load(os.path.join(d, "seg_gt.npy"))      # [128,128,128]
    seg_pred  = np.load(os.path.join(d, "seg_pred.npy"))    # [128,128,128]
    var_3d    = np.load(os.path.join(d, "variance.npy"))     # [32,32,32]

    meta_path = os.path.join(d, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        sl = meta["slice"]
        dsc_et = meta["dsc_et"]
        sid = meta["sid"]
        et_thick = meta.get("et_thickness", 0)
    else:
        et_per_slice = (seg_gt == 3).sum(axis=(0, 1))
        sl = int(np.argmax(et_per_slice))
        p, g = (seg_pred == 3).astype(float), (seg_gt == 3).astype(float)
        inter = (p * g).sum()
        dsc_et = 2*inter / max(p.sum() + g.sum(), 1)
        sid = "unknown"
        et_thick = 0

    # ── Extract slices ──
    input_sl   = img[c, :, :, sl]
    flair_sl   = img[3, :, :, sl]    # FLAIR for background
    gt_sl      = seg_gt[:, :, sl]
    pred_sl    = seg_pred[:, :, sl]

    r = img.shape[1] // var_3d.shape[0]
    var_up = zoom(var_3d, r, order=1)
    var_sl = var_up[:, :, sl]

    # ── Crop to brain ──
    y0, y1, x0, x1 = crop_to_brain(input_sl, pad=8)
    cr = lambda a: a[y0:y1, x0:x1]

    input_cr   = cr(input_sl)
    gt_cr      = cr(gt_sl)
    pred_cr    = cr(pred_sl)
    var_cr     = cr(var_sl)
    brain_cr   = input_cr > 0.01

    vmin = np.percentile(input_cr[brain_cr], 1) if brain_cr.sum() else 0
    vmax = np.percentile(input_cr[brain_cr], 99) if brain_cr.sum() else 1

    # ── ET zoom ROI (relative to cropped image) ──
    zy0, zy1, zx0, zx1 = find_et_roi(gt_cr, pad=10)

    # ── Find missed-ET + high-uncertainty peaks ──
    var_display = np.where(brain_cr, var_cr, 0)
    var_pmax = np.percentile(var_display[var_display > 0], 97) if (var_display > 0).sum() else 1
    peaks = find_missed_et_peaks(gt_cr, pred_cr, var_cr, brain_cr,
                                 n_peaks=args.n_arrows)

    # ── Figure: 1×4 main panels + insets ──
    fig = plt.figure(figsize=(22.0, 6.2), dpi=args.dpi)

    # Main grid: 1×4
    gs_main = gridspec.GridSpec(1, 4, wspace=0.04,
                                left=0.035, right=0.995,
                                top=0.82, bottom=0.12)

    # Inset grid: 1×2 below panels (b) and (c)
    gs_inset = gridspec.GridSpec(1, 4, wspace=0.04,
                                 left=0.035, right=0.995,
                                 top=0.10, bottom=-0.20)

    col_titles = [
        f'(a) {cname} Input',
        '(b) Ground Truth',
        '(c) UR-SSM-Diff',
        '(d) Uncertainty $\\sigma^2_{\\theta}$',
    ]

    # ────────────────────────────────────────────────────────
    #  (a) T1ce Input
    # ────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs_main[0, 0])
    ax_a.imshow(input_cr, cmap='gray', vmin=vmin, vmax=vmax,
                interpolation='bilinear')
    # Draw zoom box
    rect = Rectangle((zx0, zy0), zx1-zx0, zy1-zy0,
                      linewidth=2.5, edgecolor='#00BFFF', facecolor='none',
                      linestyle='-')
    ax_a.add_patch(rect)
    ax_a.set_title(col_titles[0], fontsize=16, fontweight='bold', pad=10)
    ax_a.axis('off')

    # Subject ID label
    ax_a.text(0.03, 0.03, sid.replace("BraTS2021_", "S"),
              transform=ax_a.transAxes, fontsize=11, fontweight='bold',
              color='white', va='bottom',
              bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5,
                        ec='none'))

    # ────────────────────────────────────────────────────────
    #  (b) Ground Truth
    # ────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs_main[0, 1])
    ax_b.imshow(input_cr, cmap='gray', vmin=vmin, vmax=vmax,
                interpolation='bilinear')
    ax_b.imshow(seg_to_rgba(gt_cr), interpolation='nearest')
    # Zoom box
    rect_b = Rectangle((zx0, zy0), zx1-zx0, zy1-zy0,
                        linewidth=2.5, edgecolor='#00BFFF', facecolor='none')
    ax_b.add_patch(rect_b)
    ax_b.set_title(col_titles[1], fontsize=16, fontweight='bold', pad=10)
    ax_b.axis('off')

    # ET volume annotation
    et_at_slice = (gt_cr == 3).sum()
    ax_b.text(0.97, 0.03, f'ET: {et_at_slice} vox\n~{et_thick:.1f} vox thick',
              transform=ax_b.transAxes, fontsize=11, fontweight='bold',
              color='white', va='bottom', ha='right',
              bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.6,
                        ec='none'))

    # ────────────────────────────────────────────────────────
    #  (c) UR-SSM-Diff Prediction
    # ────────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs_main[0, 2])
    ax_c.imshow(input_cr, cmap='gray', vmin=vmin, vmax=vmax,
                interpolation='bilinear')
    ax_c.imshow(seg_to_rgba(pred_cr), interpolation='nearest')
    # Zoom box
    rect_c = Rectangle((zx0, zy0), zx1-zx0, zy1-zy0,
                        linewidth=2.5, edgecolor='#00BFFF', facecolor='none')
    ax_c.add_patch(rect_c)
    ax_c.set_title(col_titles[2], fontsize=16, fontweight='bold', pad=10)
    ax_c.axis('off')

    # DSC_ET annotation (prominent)
    ax_c.text(0.97, 0.03, f'DSC$_{{ET}}$ = {dsc_et:.2f}',
              transform=ax_c.transAxes, fontsize=14, fontweight='bold',
              color='#FF4444', va='bottom', ha='right',
              bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7,
                        ec='#FF4444', lw=1.5))

    # Circle missed ET regions
    missed_et = (gt_cr == 3) & (pred_cr != 3)
    if missed_et.sum() > 3:
        ys_m, xs_m = np.where(missed_et)
        cy, cx = ys_m.mean(), xs_m.mean()
        rad = max(10, np.sqrt(missed_et.sum() / np.pi) + 6)
        circ = plt.Circle((cx, cy), rad, color='#FFD700',
                          fill=False, lw=2.5, ls='--')
        ax_c.add_patch(circ)
        ax_c.text(cx + rad + 3, cy - 4, 'Missed\nET', color='#FFD700',
                  fontsize=10, fontweight='bold', va='center',
                  bbox=dict(boxstyle='round,pad=0.2', fc='black',
                            alpha=0.6, ec='none'))

    # ────────────────────────────────────────────────────────
    #  (d) Uncertainty σ²_θ
    # ────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs_main[0, 3])
    im_var = ax_d.imshow(var_display, cmap='hot', vmin=0, vmax=var_pmax,
                          interpolation='bilinear')

    # Arrows at missed-ET + high-uncertainty locations
    for py, px in peaks:
        ax_d.annotate('',
            xy=(px, py),
            xytext=(px + 18, py - 18),
            arrowprops=dict(arrowstyle='->', color='#FFD700',
                            lw=2.8, mutation_scale=14))

    # "Safe failure" label
    ax_d.text(0.97, 0.03, 'High $\\sigma^2$ at\nmissed ET',
              transform=ax_d.transAxes, fontsize=11, fontweight='bold',
              color='#FFD700', va='bottom', ha='right',
              bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.65,
                        ec='#FFD700', lw=1.2))

    ax_d.set_title(col_titles[3], fontsize=16, fontweight='bold', pad=10)
    ax_d.axis('off')

    # ────────────────────────────────────────────────────────
    #  Zoomed insets (below panels b and c)
    # ────────────────────────────────────────────────────────
    zoom_gt   = gt_cr[zy0:zy1, zx0:zx1]
    zoom_pred = pred_cr[zy0:zy1, zx0:zx1]
    zoom_img  = input_cr[zy0:zy1, zx0:zx1]

    # Inset (b) — GT zoomed
    ax_zb = fig.add_axes([0.29, 0.01, 0.18, 0.18])
    ax_zb.imshow(zoom_img, cmap='gray', vmin=vmin, vmax=vmax,
                 interpolation='bilinear')
    ax_zb.imshow(seg_to_rgba(zoom_gt, SEG_COLORS_BRIGHT),
                 interpolation='nearest')

    # Highlight ET contour in zoomed view
    et_zoom = (zoom_gt == 3).astype(np.uint8)
    if et_zoom.sum() > 0:
        contour = binary_dilation(et_zoom, iterations=1).astype(int) - et_zoom
        contour_rgba = np.zeros((*contour.shape, 4))
        contour_rgba[contour > 0] = [1, 1, 1, 0.9]
        ax_zb.imshow(contour_rgba, interpolation='nearest')

    for spine in ax_zb.spines.values():
        spine.set_edgecolor('#00BFFF')
        spine.set_linewidth(2.5)
    ax_zb.set_xticks([]); ax_zb.set_yticks([])
    ax_zb.set_title('GT (zoomed)', fontsize=12, fontweight='bold',
                    pad=4, color='#00BFFF')

    # Inset (c) — Pred zoomed
    ax_zc = fig.add_axes([0.52, 0.01, 0.18, 0.18])
    ax_zc.imshow(zoom_img, cmap='gray', vmin=vmin, vmax=vmax,
                 interpolation='bilinear')
    ax_zc.imshow(seg_to_rgba(zoom_pred, SEG_COLORS_BRIGHT),
                 interpolation='nearest')

    # Show missed ET as dashed red outline
    missed_zoom = (zoom_gt == 3) & (zoom_pred != 3)
    if missed_zoom.sum() > 0:
        missed_rgba = np.zeros((*missed_zoom.shape, 4))
        missed_outline = binary_dilation(missed_zoom, iterations=1).astype(int) \
                         - missed_zoom.astype(int)
        missed_outline = np.clip(missed_outline, 0, 1)
        missed_rgba[missed_outline > 0] = [1, 0.3, 0.3, 0.85]
        ax_zc.imshow(missed_rgba, interpolation='nearest')

    for spine in ax_zc.spines.values():
        spine.set_edgecolor('#00BFFF')
        spine.set_linewidth(2.5)
    ax_zc.set_xticks([]); ax_zc.set_yticks([])
    ax_zc.set_title('Pred (zoomed) — ET missed', fontsize=12,
                    fontweight='bold', pad=4, color='#FF6666')

    # ────────────────────────────────────────────────────────
    #  Legend + colorbar
    # ────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(fc=SEG_COLORS[3][:3], alpha=0.8, label='ET (enhancing)'),
        mpatches.Patch(fc=SEG_COLORS[1][:3], alpha=0.8, label='ED (edema)'),
        mpatches.Patch(fc=SEG_COLORS[2][:3], alpha=0.8, label='NCR (necrotic)'),
    ]
    fig.legend(handles=legend_patches, loc='lower left',
               bbox_to_anchor=(0.02, 0.005), ncol=3, fontsize=12,
               prop={'size': 14, 'weight': 'bold'}, frameon=True, fancybox=False,
               edgecolor='0.6', handlelength=1.5, handletextpad=0.4,
               columnspacing=1.8)

    # Variance colorbar
    cax = fig.add_axes([0.82, 0.06, 0.12, 0.022])
    cb = fig.colorbar(im_var, cax=cax, orientation='horizontal')
    cb.set_ticks([])
    cb.outline.set_linewidth(1.0)
    cb.ax.text(0.0, -2.5, 'Low', fontsize=11, fontweight='bold',
               ha='left', va='top', transform=cb.ax.transAxes)
    cb.ax.text(1.0, -2.5, 'High', fontsize=11, fontweight='bold',
               ha='right', va='top', transform=cb.ax.transAxes)
    cb.ax.text(0.5, -2.5, '$\\sigma^2_{\\theta}$', fontsize=12,
               fontweight='bold', ha='center', va='top',
               transform=cb.ax.transAxes)

    # ── Save ──
    ext = args.output.split('.')[-1]
    fig.savefig(args.output, format=ext, dpi=args.dpi,
                bbox_inches='tight', pad_inches=0.06,
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\nFigure saved: {args.output}")
    print(f"  Subject: {sid}")
    print(f"  DSC_ET = {dsc_et:.3f}")
    print(f"  ET thickness ~ {et_thick:.1f} voxels")


if __name__ == "__main__":
    main()
