"""E8 accuracy cost of the time-step update interval U (SPEC 11, RQ3).

B0 (90 m, 24 h): each U against U=1: max and mean |hmax diff| over LAND, flood-extent
agreement (CSI of the U map against the U=1 map on 90 m LAND) at 0.10/0.25/0.50 m,
mass error and step count.
V0 (90 m, 23 days): each U against GFD 2734 exactly as in E3 (250 m, fraction 0.5)
at 0.10/0.25/0.50 m, the CSI difference to U=1, and max |hmax diff| to U=1.
Output: results/e8_accuracy.csv.
"""
from __future__ import annotations

import argparse
import csv
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

US = (1, 5, 10, 25, 50)
H_FLOOD = (0.10, 0.25, 0.50)


def _load(run: Path):
    if not (run / "hmax.bin").is_file():
        return None, None
    hmax, _ = read_bin(run / "hmax.bin")
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    return hmax.astype(np.float64), man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--runs", default="results/e8")
    ap.add_argument("--v0-u1", default="results/v0_smoke")
    ap.add_argument("--out", default="results/e8_accuracy.csv")
    args = ap.parse_args()
    proc, runs = Path(args.proc), Path(args.runs)
    mask90, mhdr = read_bin(proc / "90" / "mask.bin")
    land = mask90 == 0
    rows = []

    ref, _ = _load(runs / "B0_U1")
    for U in US:
        h, man = _load(runs / f"B0_U{U}")
        if h is None or ref is None:
            print(f"skip B0 U={U}")
            continue
        d = np.abs(h - ref)[land]
        for thr in H_FLOOD:
            a, b = (h >= thr) & land, (ref >= thr) & land
            inter, union = int((a & b).sum()), int((a | b).sum())
            rows.append({"case": "B0_24h", "U": U, "h_flood": thr, "steps": man["steps"],
                         "E_mass": man["E_mass"], "max_abs_hmax_diff_m": float(d.max()),
                         "mean_abs_hmax_diff_m": float(d.mean()), "flooded_cells_90": int(a.sum()),
                         "flooded_cells_90_U1": int(b.sum()),
                         "CSI_vs_U1_90m": inter / union if union else 1.0,
                         "CSI_vs_GFD": "", "dCSI_vs_GFD": "", "run": str(runs / f"B0_U{U}")})

    obs, obs_hdr = read_bin(proc / "validation" / "gfd_2734_flooded_250.bin")
    perm, _ = read_bin(proc / "validation" / "gfd_2734_jrc_perm_water_250.bin")
    shape, dx_o = obs.shape, float(obs_hdr["dx"])
    x0_o, y0_o = float(obs_hdr["x0"]), float(obs_hdr["y0"])
    sea_wall = resample_nearest_u8(((mask90 == 1) | (mask90 == 2)).astype(np.uint8), 90.0,
                                   float(mhdr["x0"]), float(mhdr["y0"]), shape, dx_o, x0_o, y0_o)
    valid = exclusion_mask(perm, sea_wall.astype(bool), clear_perc=None)
    v0 = {1: Path(args.v0_u1), **{U: runs / f"V0_U{U}" for U in US if U != 1}}
    ref, _ = _load(v0[1])
    csi1 = {}
    for U in US:
        h, man = _load(v0[U])
        if h is None or ref is None:
            print(f"skip V0 U={U}")
            continue
        d = np.abs(h - ref)[land]
        for thr in H_FLOOD:
            m90 = ((h >= thr) & land).astype(np.uint8)
            m250 = model_flood_on_obs_grid(m90, dict(mhdr), shape, dx_o, x0_o, y0_o, 0.5)
            csi = metrics_from_counts(confusion_counts(obs, m250, valid))["CSI"]
            if U == 1:
                csi1[thr] = csi
            b = (ref >= thr) & land
            a = m90.astype(bool)
            union = int((a | b).sum())
            rows.append({"case": "V0_23d", "U": U, "h_flood": thr, "steps": man["steps"],
                         "E_mass": man["E_mass"], "max_abs_hmax_diff_m": float(d.max()),
                         "mean_abs_hmax_diff_m": float(d.mean()), "flooded_cells_90": int(a.sum()),
                         "flooded_cells_90_U1": int(b.sum()),
                         "CSI_vs_U1_90m": int((a & b).sum()) / union if union else 1.0,
                         "CSI_vs_GFD": csi, "dCSI_vs_GFD": csi - csi1[thr] if thr in csi1 else "",
                         "run": str(v0[U])})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["h_flood"] == 0.25:
            print({k: r[k] for k in ("case", "U", "steps", "E_mass", "max_abs_hmax_diff_m",
                                     "CSI_vs_U1_90m", "CSI_vs_GFD", "dCSI_vs_GFD")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
