"""V0 smoke test: CSI/POD/FAR/Bias at two depth thresholds and where flooding lies.

Not the final E3 (no calibration, K=4 only). Uses the SPEC 10.1 comparison grid
and exclusion mask (GFD jrc_perm_water, SEA, WALL).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from floodca_prep.io_bin import read_bin
from floodca_prep.validation_metrics import (
    confusion_counts,
    exclusion_mask,
    metrics_from_counts,
    model_flood_on_obs_grid,
    resample_binary_average,
    resample_nearest_u8,
)

CHANNEL_KM = 0.5
LOW_Z_M = 10.0


def _to_obs(arr90, hdr90, obs_shape, obs_hdr):
    return resample_binary_average(
        arr90.astype(np.float32), float(hdr90["dx"]), float(hdr90["x0"]), float(hdr90["y0"]),
        obs_shape, float(obs_hdr["dx"]), float(obs_hdr["x0"]), float(obs_hdr["y0"]),
    )


def _classes(z250, dperm_km):
    cls = np.full(z250.shape, "higher_ground", dtype=object)
    cls[z250 < LOW_Z_M] = "low_coastal_plain"
    cls[dperm_km <= CHANNEL_KM] = "channel_estuary_adjacent"
    return cls


def _describe(name, thr, sel, cls, z250, dsea_km, dperm_km, rows):
    n = int(sel.sum())
    for c in ("channel_estuary_adjacent", "low_coastal_plain", "higher_ground"):
        k = int((sel & (cls == c)).sum())
        rows.append({"set": name, "h_flood": thr, "class": c, "count": k,
                     "frac": (k / n) if n else 0.0,
                     "median_z_m": float(np.median(z250[sel])) if n else float("nan"),
                     "median_dist_sea_km": float(np.median(dsea_km[sel])) if n else float("nan"),
                     "median_dist_permwater_km": float(np.median(dperm_km[sel])) if n else float("nan"),
                     "n_total": n})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--run", default="results/v0_smoke")
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args()
    proc, run = Path(args.proc), Path(args.run)

    obs, obs_hdr = read_bin(proc / "validation" / "gfd_2734_flooded_250.bin")
    perm, _ = read_bin(proc / "validation" / "gfd_2734_jrc_perm_water_250.bin")
    mask90, mhdr = read_bin(proc / "90" / "mask.bin")
    dem90, _ = read_bin(proc / "90" / "dem.bin")
    hmax, hhdr = read_bin(run / "hmax.bin")
    hdr90 = dict(mhdr)
    if abs(float(hhdr.get("x0", 0)) - float(mhdr["x0"])) > 1e-3:
        raise RuntimeError("hmax header is not on the 90 m model grid")

    sea_wall_250 = resample_nearest_u8(
        ((mask90 == 1) | (mask90 == 2)).astype(np.uint8), 90.0, float(mhdr["x0"]), float(mhdr["y0"]),
        obs.shape, float(obs_hdr["dx"]), float(obs_hdr["x0"]), float(obs_hdr["y0"]),
    ).astype(bool)
    valid = exclusion_mask(perm, sea_wall_250, clear_perc=None)

    z250 = _to_obs(np.where(mask90 == 0, dem90, 0.0), hdr90, obs.shape, obs_hdr)
    dsea90 = distance_transform_edt(mask90 != 1) * 90.0 / 1000.0
    dsea_km = _to_obs(dsea90, hdr90, obs.shape, obs_hdr)
    dperm_km = distance_transform_edt(perm != 1) * float(obs_hdr["dx"]) / 1000.0
    cls = _classes(z250, dperm_km)

    obs_sel = (obs == 1) & valid
    metric_rows, loc_rows = [], []
    _describe("valid_all", "", valid, cls, z250, dsea_km, dperm_km, loc_rows)
    _describe("observed", "", obs_sel, cls, z250, dsea_km, dperm_km, loc_rows)
    model_maps = {}
    for thr in (0.10, 0.25):
        m90 = ((hmax >= thr) & (mask90 == 0)).astype(np.uint8)
        m250 = model_flood_on_obs_grid(m90, hdr90, obs.shape, float(obs_hdr["dx"]),
                                       float(obs_hdr["x0"]), float(obs_hdr["y0"]), args.frac)
        model_maps[thr] = m250
        cnt = confusion_counts(obs, m250, valid)
        met = metrics_from_counts(cnt)
        msel = (m250 == 1) & valid
        metric_rows.append({"h_flood": thr, "frac_threshold": args.frac, **cnt, **met,
                            "n_obs_flood_valid_250": int(obs_sel.sum()),
                            "n_model_flood_valid_250": int(msel.sum()),
                            "n_model_flood_land_90": int(m90.sum()),
                            "n_land_90": int((mask90 == 0).sum())})
        _describe("model", thr, msel, cls, z250, dsea_km, dperm_km, loc_rows)
        _describe("hit", thr, msel & obs_sel, cls, z250, dsea_km, dperm_km, loc_rows)

    land = mask90 == 0
    hl = hmax[land]
    depth_stats = {
        "hmax_land_max_m": float(hl.max()),
        "hmax_land_p99_m": float(np.percentile(hl, 99)),
        "hmax_land_p999_m": float(np.percentile(hl, 99.9)),
        "n_land_hmax_ge_0.05": int((hl >= 0.05).sum()),
        "n_land_hmax_ge_0.10": int((hl >= 0.10).sum()),
        "n_land_hmax_ge_0.25": int((hl >= 0.25).sum()),
        "n_land_hmax_ge_0.50": int((hl >= 0.50).sum()),
        "n_land_hmax_ge_1.00": int((hl >= 1.00).sum()),
    }

    run.mkdir(parents=True, exist_ok=True)
    with (run / "v0_smoke_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)
    with (run / "v0_smoke_location.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loc_rows[0].keys()))
        w.writeheader()
        w.writerows(loc_rows)
    (run / "v0_smoke_depth.json").write_text(json.dumps(depth_stats, indent=2) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["#ffffff", "#e8e8e8", "#1f77b4", "#ff7f0e", "#d62728"])
        rows_any = np.where((obs_sel | (model_maps[0.10] == 1)).any(axis=1))[0]
        cols_any = np.where((obs_sel | (model_maps[0.10] == 1)).any(axis=0))[0]
        pad = 10
        r0, r1 = max(rows_any.min() - pad, 0), min(rows_any.max() + pad, obs.shape[0])
        c0, c1 = max(cols_any.min() - pad, 0), min(cols_any.max() + pad, obs.shape[1])
        fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
        for ax, thr in zip(axes, (0.10, 0.25)):
            m = model_maps[thr] == 1
            img = np.zeros(obs.shape, dtype=np.uint8)
            img[valid] = 1
            img[valid & m & obs_sel] = 2
            img[valid & ~m & obs_sel] = 3
            img[valid & m & ~obs_sel] = 4
            ax.imshow(img[r0:r1, c0:c1], cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            met = next(x for x in metric_rows if x["h_flood"] == thr)
            ax.set_title(f"V0 smoke, h_flood={thr:.2f} m: CSI={met['CSI']:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in (1, 2, 3, 4)]
        fig.legend(handles, ["correct negative", "hit", "miss", "false alarm"], loc="lower center", ncol=4)
        fig.savefig(run / "v0_smoke_confusion.png", dpi=110)
    except Exception as exc:  # figure is optional for the smoke test
        print(f"figure skipped: {exc!r}")

    print(json.dumps({"metrics": metric_rows, "depth": depth_stats}, indent=2))
    for r in loc_rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
