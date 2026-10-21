"""E3 validation accuracy (SPEC 10.1 to 10.3, 11.3): Table T5 and Figure F15.

Every V0 run (K in {4, 8} x k_C in {0.7, 1.0, 1.3}) is scored on the 250 m GFD 2734
grid at h_flood in {0.10, 0.25, 0.50} and model-flood fraction threshold in
{0.3, 0.5, 0.7}. Calibration (SPEC 10.3) picks, per K, the (h_flood, k_C) with the
highest CSI at fraction 0.5. It is calibrated on the validation event itself.
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

H_FLOOD = (0.10, 0.25, 0.50)
FRACS = (0.3, 0.5, 0.7)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--runs", default="K4_kC1.0=results/v0_smoke,K8_kC1.0=results/e3/K8_kC1.0,"
                    "K4_kC0.7=results/e3/K4_kC0.7,K8_kC0.7=results/e3/K8_kC0.7,"
                    "K4_kC1.3=results/e3/K4_kC1.3,K8_kC1.3=results/e3/K8_kC1.3")
    ap.add_argument("--out-csv", default="results/e3_metrics.csv")
    ap.add_argument("--out-json", default="results/e3_calibration.json")
    ap.add_argument("--fig", default="paper/shared/figures/F15_validation_map.png")
    args = ap.parse_args()
    proc = Path(args.proc)

    obs, obs_hdr = read_bin(proc / "validation" / "gfd_2734_flooded_250.bin")
    perm, _ = read_bin(proc / "validation" / "gfd_2734_jrc_perm_water_250.bin")
    mask90, mhdr = read_bin(proc / "90" / "mask.bin")
    shape, dx_o = obs.shape, float(obs_hdr["dx"])
    x0_o, y0_o = float(obs_hdr["x0"]), float(obs_hdr["y0"])
    sea_wall = resample_nearest_u8(((mask90 == 1) | (mask90 == 2)).astype(np.uint8), 90.0,
                                   float(mhdr["x0"]), float(mhdr["y0"]), shape, dx_o, x0_o, y0_o)
    valid = exclusion_mask(perm, sea_wall.astype(bool), clear_perc=None)

    rows, maps = [], {}
    for item in args.runs.split(","):
        tag, path = item.split("=")
        hmax_path = Path(path) / "hmax.bin"
        if not hmax_path.is_file():
            print(f"skip {tag}: {hmax_path} missing")
            continue
        hmax, _ = read_bin(hmax_path)
        man = json.loads((Path(path) / "run_manifest.json").read_text(encoding="utf-8"))
        K = int(tag.split("_")[0][1:])
        k_c = float(tag.split("kC")[1])
        for thr in H_FLOOD:
            m90 = ((hmax >= thr) & (mask90 == 0)).astype(np.uint8)
            for frac in FRACS:
                m250 = model_flood_on_obs_grid(m90, dict(mhdr), shape, dx_o, x0_o, y0_o, frac)
                cnt = confusion_counts(obs, m250, valid)
                met = metrics_from_counts(cnt)
                rows.append({"run": tag, "K": K, "k_C": k_c, "h_flood": thr, "frac_threshold": frac,
                             **cnt, **met, "n_model_flood_land_90": int(m90.sum()),
                             "steps": man["steps"], "E_mass": man["E_mass"], "run_dir": path})
                maps[(tag, thr, frac)] = m250

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    calib = {"note": "calibrated on the validation event (GFD 2734); SPEC 10.3 limitation",
             "criterion": "max CSI at frac_threshold 0.5 over h_flood x k_C", "by_K": {}}
    for K in sorted({r["K"] for r in rows}):
        cand = [r for r in rows if r["K"] == K and r["frac_threshold"] == 0.5]
        best = max(cand, key=lambda r: r["CSI"])
        default = next((r for r in cand if r["k_C"] == 1.0 and r["h_flood"] == 0.25), None)
        calib["by_K"][str(K)] = {"best": {k: best[k] for k in ("run", "k_C", "h_flood", "CSI", "POD",
                                                                "FAR", "Bias", "F1", "H", "F", "M")},
                                 "default_kC1_h0.25": None if default is None else
                                 {k: default[k] for k in ("CSI", "POD", "FAR", "Bias", "F1")},
                                 "n_candidates": len(cand)}
    Path(args.out_json).write_text(json.dumps(calib, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(calib, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(["#ffffff", "#e8e8e8", "#1f77b4", "#ff7f0e", "#d62728"])
        obs_sel = (obs == 1) & valid
        panels = []
        for K, c in calib["by_K"].items():
            b = c["best"]
            panels.append((f"K={K}, calibrated: k_C={b['k_C']:g}, h_flood={b['h_flood']:.2f} m",
                           maps[(b["run"], b["h_flood"], 0.5)], b["CSI"]))
        if ("K4_kC1.0", 0.25, 0.5) in maps:
            d = next(r for r in rows if r["run"] == "K4_kC1.0" and r["h_flood"] == 0.25
                     and r["frac_threshold"] == 0.5)
            panels.insert(0, ("K=4, default: k_C=1, h_flood=0.25 m", maps[("K4_kC1.0", 0.25, 0.5)],
                              d["CSI"]))
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 6.5), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (title, m250, csi) in zip(axes, panels):
            m = m250 == 1
            img = np.zeros(shape, dtype=np.uint8)
            img[valid] = 1
            img[valid & m & obs_sel] = 2
            img[valid & ~m & obs_sel] = 3
            img[valid & m & ~obs_sel] = 4
            ax.imshow(img, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            ax.set_title(f"{title}\nCSI={csi:.3f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in (1, 2, 3, 4)]
        fig.legend(handles, ["correct negative", "hit", "miss", "false alarm"], loc="lower center", ncol=4)
        fig.suptitle("F15: V0 (GFD 2734, 2005-09-19 to 2005-10-11) on the 250 m comparison grid")
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=120)
        print("wrote", args.fig)
    except Exception as exc:
        print(f"figure skipped: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
