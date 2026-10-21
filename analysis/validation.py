"""CSI / POD / FAR / Bias / F1 for V0 (SPEC Sec 10.1, 10.2).

Uses GFD jrc_perm_water as the permanent-water exclusion. Does not read
JRC GSW max_extent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floodca_prep.io_bin import read_bin
from floodca_prep.validation_metrics import (
    confusion_counts,
    exclusion_mask,
    metrics_from_counts,
    model_flood_on_obs_grid,
    resample_nearest_u8,
)


def compare_v0(
    proc_root: Path,
    model_hmax_90: Path | None,
    h_flood: float = 0.25,
    frac_threshold: float = 0.5,
) -> dict:
    obs, obs_hdr = read_bin(proc_root / "validation" / "gfd_2734_flooded_250.bin")
    perm, _ = read_bin(proc_root / "validation" / "gfd_2734_jrc_perm_water_250.bin")
    mask90, mask_hdr = read_bin(proc_root / "90" / "mask.bin")
    sea_wall_90 = (mask90 == 1) | (mask90 == 2)
    sea_wall_250 = resample_nearest_u8(
        sea_wall_90.astype(np.uint8),
        float(mask_hdr["dx"]),
        float(mask_hdr["x0"]),
        float(mask_hdr["y0"]),
        obs.shape,
        float(obs_hdr["dx"]),
        float(obs_hdr["x0"]),
        float(obs_hdr["y0"]),
    ).astype(bool)
    valid = exclusion_mask(perm, sea_wall_250, clear_perc=None)
    if model_hmax_90 is None or not Path(model_hmax_90).exists():
        return {
            "status": "obs_ready",
            "perm_source": "GFD jrc_perm_water",
            "n_valid": int(valid.sum()),
            "n_obs_flood_valid": int(((obs == 1) & valid).sum()),
            "h_flood": h_flood,
            "frac_threshold": frac_threshold,
            "note": "model hmax not supplied; exclusion mask is wired and ready for V0",
        }
    hmax, hhdr = read_bin(model_hmax_90)
    model90 = ((hmax >= h_flood) & (mask90 == 0)).astype(np.uint8)
    model_on_obs = model_flood_on_obs_grid(
        model90,
        hhdr,
        obs.shape,
        float(obs_hdr["dx"]),
        float(obs_hdr["x0"]),
        float(obs_hdr["y0"]),
        frac_threshold,
    )
    counts = confusion_counts(obs, model_on_obs, valid)
    mets = metrics_from_counts(counts)
    return {
        "status": "compared",
        "perm_source": "GFD jrc_perm_water",
        "counts": counts,
        "metrics": mets,
        "n_valid": int(valid.sum()),
        "h_flood": h_flood,
        "frac_threshold": frac_threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--hmax", default=None, help="optional 90 m hmax.bin from a V0 run")
    ap.add_argument("--h-flood", type=float, default=0.25)
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args()
    out = compare_v0(Path(args.proc), Path(args.hmax) if args.hmax else None, args.h_flood, args.frac)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
