"""Diagnostic for the V0 smoke run: where the model floods at z ~ 0, and why some
250 m comparison cells are excluded. Prints facts only; changes nothing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from floodca_prep.io_bin import read_bin

PROC = Path("data/processed/udupi_coast")
RUN = Path("results/v0_smoke")


def main() -> int:
    z, _ = read_bin(PROC / "90" / "dem.bin")
    mask, _ = read_bin(PROC / "90" / "mask.bin")
    hmax, _ = read_bin(RUN / "hmax.bin")
    land = mask == 0
    sea = mask == 1
    wet10 = land & (hmax >= 0.10)
    out: dict = {"n_land": int(land.sum()), "n_sea": int(sea.sum()), "n_wet10": int(wet10.sum())}

    lab, n = ndimage.label(wet10, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:5]
    comps = []
    for k in order:
        sel = lab == (k + 1)
        rows, cols = np.nonzero(sel)
        zz = z[sel]
        hh = hmax[sel]
        ring = ndimage.binary_dilation(sel, iterations=1) & ~sel
        comps.append({
            "cells": int(sel.sum()),
            "rows": [int(rows.min()), int(rows.max())],
            "cols": [int(cols.min()), int(cols.max())],
            "z_min": float(zz.min()), "z_median": float(np.median(zz)), "z_max": float(zz.max()),
            "frac_z_eq_0": float(np.mean(zz == 0.0)),
            "frac_z_lt_0p5": float(np.mean(zz < 0.5)),
            "hmax_median": float(np.median(hh)), "hmax_max": float(hh.max()),
            "ring_sea_cells": int((ring & sea).sum()),
            "ring_land_z_median": float(np.median(z[ring & land])) if (ring & land).any() else None,
        })
    out["largest_wet10_components"] = comps

    # Land cells at z < 0.5 m (they would be SEA if 4-connected to the west edge).
    low = land & (z < 0.5)
    out["land_z_lt_0p5"] = int(low.sum())
    out["land_z_eq_0"] = int((land & (z == 0.0)).sum())
    lab4, n4 = ndimage.label(low)
    s4 = ndimage.sum(np.ones_like(lab4), lab4, index=np.arange(1, n4 + 1)) if n4 else np.array([])
    out["land_low_components_top5"] = sorted([int(s) for s in s4], reverse=True)[:5]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
