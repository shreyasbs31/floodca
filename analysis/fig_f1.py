"""Draft F1: study area DEM hillshade, land cover, GFD 2734 flood (SPEC 14.5)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

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

from floodca_prep.io_bin import read_bin


def _hillshade(z: np.ndarray, azimuth=315.0, altitude=45.0) -> np.ndarray:
    x, y = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.hypot(x, y))
    aspect = np.arctan2(-x, y)
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return (hs - hs.min()) / (hs.max() - hs.min() + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--out", default="paper/shared/figures/F1_study_area.png")
    args = ap.parse_args()
    proc = Path(args.proc)
    dem, _ = read_bin(proc / "90" / "dem.bin")
    mask, _ = read_bin(proc / "90" / "mask.bin")
    lc, _ = read_bin(proc / "90" / "lc.bin")
    flood, _ = read_bin(proc / "validation" / "gfd_2734_flooded_90.bin")
    perm, _ = read_bin(proc / "validation" / "gfd_2734_jrc_perm_water_90.bin")
    step = 2
    z = dem[::step, ::step]
    hs = _hillshade(z)
    lc_s = lc[::step, ::step]
    m_s = mask[::step, ::step]
    fl = (flood[::step, ::step] == 1) & (perm[::step, ::step] != 1) & (m_s == 0)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    axes[0].imshow(hs, cmap="gray", origin="upper")
    axes[0].set_title("DEM hillshade (90 m)")
    axes[0].axis("off")

    lc_plot = lc_s.astype(float)
    lc_plot[m_s == 1] = np.nan
    axes[1].imshow(lc_plot, cmap="tab20", origin="upper")
    axes[1].set_title("WorldCover (mode to 90 m)")
    axes[1].axis("off")

    rgb = np.stack([hs, hs, hs], axis=-1)
    overlay = rgb.copy()
    overlay[fl] = (0.85, 0.1, 0.1)
    axes[2].imshow(overlay, origin="upper")
    axes[2].set_title("GFD 2734 flood, perm water excluded")
    axes[2].axis("off")
    fig.suptitle("F1 draft: udupi_coast, GFD event 2734 (modified Tier A)", fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
