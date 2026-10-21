"""Study-area figure: elevation, ensemble probability, observed extent.

Panel (a) is the 90 m elevation. S, M and L are crops of the 30 m grid drawn on that extent.
Panel (b) is the 500-member probability at 0.10 m, summed from the stored count chunks.
Panel (c) is the observed flood extent for event 2734 with permanent water removed.

Reads the rasters that prep/ writes under data/processed/udupi_coast, plus the
count chunks in results/e12/main. Those downloads are not in this repository.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prep"))

os.environ.setdefault("MPLBACKEND", "Agg")
_mpl = os.environ.get("MPLCONFIGDIR") or os.path.join(
    "/tmp", os.environ.get("USER", "floodca") + "_mplconfig"
)
os.environ["MPLCONFIGDIR"] = _mpl
Path(_mpl).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from floodca_prep.io_bin import read_bin

CHUNKS = ((0, 8), (8, 8), (16, 16), (32, 32), (64, 64), (128, 122), (250, 250))


def _extent(hdr):
    # Row 0 is the north edge.
    x0 = float(hdr["x0"])
    y0 = float(hdr["y0"])
    dx = float(hdr["dx"])
    nrows, ncols = int(hdr["nrows"]), int(hdr["ncols"])
    return [x0, x0 + ncols * dx, y0 - nrows * dx, y0]


def main() -> int:
    proc = ROOT / "data" / "processed" / "udupi_coast"
    dem, dh = read_bin(proc / "90" / "dem.bin")
    mask, _ = read_bin(proc / "90" / "mask.bin")
    flood, _ = read_bin(proc / "validation" / "gfd_2734_flooded_90.bin")
    perm, _ = read_bin(proc / "validation" / "gfd_2734_jrc_perm_water_90.bin")
    ext = _extent(dh)
    counts = np.zeros(dem.size, dtype=np.int64)
    for first, n in CHUNKS:
        path = ROOT / "results" / "e12" / "main" / f"chunk_{first:03d}_{n:03d}" / "count_010.bin"
        counts += np.fromfile(path, dtype=np.uint16).astype(np.int64)
    prob = (counts / 500.0).reshape(dem.shape)
    land = mask == 0

    fig = plt.figure(figsize=(7.7, 3.85))
    grid = fig.add_gridspec(1, 3, wspace=0.78, left=0.07, right=0.985, bottom=0.26, top=0.97)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    km = [ext[0] / 1000.0, ext[1] / 1000.0, ext[2] / 1000.0, ext[3] / 1000.0]
    z = np.ma.masked_invalid(dem)
    im0 = axes[0].imshow(z, cmap="terrain", extent=km, origin="upper", interpolation="nearest")
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    c0.set_label("elevation (m)", fontsize=7)
    c0.ax.tick_params(labelsize=6)
    crops = (("S", "#000000"), ("M", "#0072B2"), ("L", "#D55E00"))
    for name, color in crops:
        hdr = __import__("json").loads((proc / name / "dem.json").read_text())
        x0, y0 = float(hdr["x0"]), float(hdr["y0"])
        dx = float(hdr["dx"])
        width = int(hdr["ncols"]) * dx
        height = int(hdr["nrows"]) * dx
        axes[0].add_patch(Rectangle((x0 / 1000.0, (y0 - height) / 1000.0), width / 1000.0, height / 1000.0,
                                    fill=False, edgecolor=color, linewidth=1.2, label=name))
    axes[0].plot([], [], color="none", label="frame: 90 m")
    axes[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28),
                   ncol=2, frameon=False, columnspacing=0.6, handlelength=1.1)
    axes[0].set_xlabel("easting (km)")
    axes[0].set_ylabel("northing (km)")

    shown = np.ma.masked_where(~land, prob)
    im1 = axes[1].imshow(shown, cmap="viridis", vmin=0, vmax=1, extent=km,
                         origin="upper", interpolation="nearest")
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    c1.set_label("probability at 0.10 m", fontsize=7)
    c1.ax.tick_params(labelsize=6)
    axes[1].set_xlabel("easting (km)")
    axes[1].set_yticks([])

    obs = (flood == 1) & (perm != 1) & land
    rgb = np.zeros(dem.shape + (3,), dtype=float)
    rgb[..., 0] = 0.93
    rgb[..., 1] = 0.93
    rgb[..., 2] = 0.90
    rgb[mask == 1] = (0.75, 0.86, 0.95)
    rgb[obs] = (0.80, 0.15, 0.15)
    rows, cols = np.where(obs)
    dx = float(dh["dx"])
    pad = 15
    r0 = max(int(rows.min()) - pad, 0)
    r1 = min(int(rows.max()) + pad + 1, dem.shape[0])
    c0 = max(int(cols.min()) - pad, 0)
    c1 = min(int(cols.max()) + pad + 1, dem.shape[1])
    x0 = float(dh["x0"])
    y_north = float(dh["y0"])
    ext_c = [
        (x0 + c0 * dx) / 1000.0,
        (x0 + c1 * dx) / 1000.0,
        (y_north - r1 * dx) / 1000.0,
        (y_north - r0 * dx) / 1000.0,
    ]
    axes[2].imshow(rgb, extent=km, origin="upper", interpolation="nearest")
    # Match the aspect of (a) and (b) so (c) is the same height and not a strip.
    full_w = km[1] - km[0]
    full_h = km[3] - km[2]
    target = full_w / full_h
    cx0, cx1, cy0, cy1 = ext_c
    fw, fh = cx1 - cx0, cy1 - cy0
    if fw / fh < target:
        need = fh * target
        mid = 0.5 * (cx0 + cx1)
        cx0, cx1 = mid - 0.5 * need, mid + 0.5 * need
    else:
        need = fw / target
        mid = 0.5 * (cy0 + cy1)
        cy0, cy1 = mid - 0.5 * need, mid + 0.5 * need

    def _shift_inside(lo, hi, bound_lo, bound_hi):
        span = hi - lo
        room = bound_hi - bound_lo
        if span >= room:
            return bound_lo, bound_hi
        if lo < bound_lo:
            hi += bound_lo - lo
            lo = bound_lo
        if hi > bound_hi:
            lo -= hi - bound_hi
            hi = bound_hi
        return lo, hi

    cx0, cx1 = _shift_inside(cx0, cx1, km[0], km[1])
    cy0, cy1 = _shift_inside(cy0, cy1, km[2], km[3])
    axes[2].set_xlim(cx0, cx1)
    axes[2].set_ylim(cy0, cy1)
    axes[2].set_xlabel("easting (km)")
    axes[2].set_ylabel("northing (km)")
    axes[2].plot([], [], color=(0.80, 0.15, 0.15), lw=4, label="observed flood")
    axes[2].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)

    def _scalebar(ax, x_right, y, length, label):
        ax.plot([x_right - length, x_right], [y, y], color="k", lw=1.6,
                solid_capstyle="butt", clip_on=True, zorder=5)
        ax.text(x_right - 0.5 * length, y + 0.018 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                label, ha="center", va="bottom", fontsize=6, color="k", clip_on=True, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.15))

    # Lower right, inset from the colourbar and clear of the panel letter.
    _scalebar(axes[0], km[1] - 12, km[2] + 8, 20, "20 km")
    _scalebar(axes[1], km[1] - 12, km[2] + 8, 20, "20 km")
    cspan = cx1 - cx0
    bar = 5 if cspan > 12 else 2
    _scalebar(axes[2], cx1 - 0.12 * cspan, cy0 + 0.10 * (cy1 - cy0), bar, f"{bar} km")

    for ax, letter in zip(axes, "abc"):
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, va="top", ha="left",
                fontsize=8, zorder=6)
    out = ROOT / "figures" / "fig1.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("F1 written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
