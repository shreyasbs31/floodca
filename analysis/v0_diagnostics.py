"""Honest V0 diagnostics on the fixed 90 m mask. No parameters are changed.

Continuous score: area-weighted mean of hmax over the 90 m LAND cells inside each
250 m cell. ROC AUC is Mann-Whitney on the valid cells (SPEC 10.1 exclusions).
Zones match analysis/v0_smoke.py: within 0.5 km of GFD permanent water, else
z < 10 m, else higher ground. Baselines are scored on the same valid cells.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from analysis.e12 import auc_mann_whitney
from floodca_prep.io_bin import read_bin
from floodca_prep.validation_metrics import (
    confusion_counts,
    exclusion_mask,
    metrics_from_counts,
    model_flood_on_obs_grid,
    resample_binary_average,
    resample_nearest_u8,
)


def _avg(arr, hdr90, shape, obs_hdr):
    return resample_binary_average(
        arr.astype(np.float32), float(hdr90["dx"]), float(hdr90["x0"]), float(hdr90["y0"]),
        shape, float(obs_hdr["dx"]), float(obs_hdr["x0"]), float(obs_hdr["y0"]),
    )


def _row(name, zone, obs, pred, valid):
    c = confusion_counts(obs, pred.astype(np.uint8), valid)
    return {"score": name, "zone": zone, **c, **metrics_from_counts(c), "n_valid": int(valid.sum())}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/v0_smoke", help="directory containing hmax.bin")
    ap.add_argument("--event", type=int, default=2734)
    args = ap.parse_args(argv)
    proc = Path("data/processed/udupi_coast")
    flooded = proc / "validation" / f"gfd_{args.event}_flooded_250.bin"
    perm_path = proc / "validation" / f"gfd_{args.event}_jrc_perm_water_250.bin"
    hmax_path = Path(args.run) / "hmax.bin"
    if not flooded.is_file() or not perm_path.is_file() or not hmax_path.is_file():
        print(f"missing inputs for event {args.event}: {flooded} {perm_path} {hmax_path}")
        return 2
    obs, obs_hdr = read_bin(flooded)
    perm, _ = read_bin(perm_path)
    mask90, mhdr = read_bin(proc / "90" / "mask.bin")
    dem90, _ = read_bin(proc / "90" / "dem.bin")
    hmax, _ = read_bin(hmax_path)
    label = "V0" if args.event == 2734 else f"GFD{args.event}"
    stem = Path("results/v0_diagnostics") if args.event == 2734 else Path(f"results/val_{args.event}_diagnostics")
    hdr90 = dict(mhdr)
    shape = obs.shape
    sea_wall = resample_nearest_u8(
        ((mask90 == 1) | (mask90 == 2)).astype(np.uint8), 90.0, float(mhdr["x0"]), float(mhdr["y0"]),
        shape, float(obs_hdr["dx"]), float(obs_hdr["x0"]), float(obs_hdr["y0"]),
    ).astype(bool)
    valid = exclusion_mask(perm, sea_wall, clear_perc=None)
    land90 = mask90 == 0
    land_frac = _avg(land90.astype(np.float32), hdr90, shape, obs_hdr)
    h_avg = _avg(np.where(land90, hmax, 0.0).astype(np.float32), hdr90, shape, obs_hdr)
    score = np.zeros(shape, dtype=np.float64)
    np.divide(h_avg, land_frac, out=score, where=land_frac > 1e-6)
    o = (obs == 1) & valid
    auc = auc_mann_whitney(score[valid], o[valid].astype(np.float64))

    z250 = _avg(np.where(land90, dem90, 0.0).astype(np.float32), hdr90, shape, obs_hdr)
    dperm_km = distance_transform_edt(perm != 1) * float(obs_hdr["dx"]) / 1000.0
    zone = np.full(shape, "higher_ground", dtype=object)
    zone[z250 < 10.0] = "low_coastal_plain"
    zone[dperm_km <= 0.5] = "within_0.5km_perm_water"

    rows = []
    maps = {}
    for thr in (0.10, 0.25):
        m90 = ((hmax >= thr) & land90).astype(np.uint8)
        m250 = model_flood_on_obs_grid(
            m90, hdr90, shape, float(obs_hdr["dx"]), float(obs_hdr["x0"]), float(obs_hdr["y0"]), 0.5)
        pred = (m250 == 1) & valid
        maps[thr] = m250
        rows.append(_row(f"model_h{thr:.2f}", "all_valid", obs, pred, valid))
        for zname in ("within_0.5km_perm_water", "low_coastal_plain", "higher_ground"):
            rows.append(_row(f"model_h{thr:.2f}", zname, obs, pred, valid & (zone == zname)))
        n_model = int(pred.sum())
        base_a = valid & (dperm_km <= 0.5)
        rows.append(_row(f"baseline_near_perm_vs_model_h{thr:.2f}", "all_valid", obs, base_a, valid))
        order = np.argsort(z250, axis=None)
        lowest = np.zeros(shape, dtype=bool).ravel()
        picked = 0
        for idx in order:
            if not valid.ravel()[idx]:
                continue
            lowest[idx] = True
            picked += 1
            if picked >= n_model:
                break
        rows.append(_row(f"baseline_lowest_{n_model}_vs_model_h{thr:.2f}", "all_valid", obs,
                         lowest.reshape(shape), valid))

    out = {
        "event_id": args.event,
        "label": label,
        "hmax_run": str(hmax_path),
        "score": "area-weighted mean of hmax over 90 m LAND cells in each 250 m cell",
        "ROC_AUC": auc,
        "n_valid": int(valid.sum()),
        "n_observed_flood_valid": int(o.sum()),
        "zone_rule": "within 0.5 km of permanent water, else z < 10 m, else higher ground",
    }
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", *rows[0].keys()])
        w.writeheader()
        for r in rows:
            w.writerow({"event_id": args.event, **r})
    print(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['score']:44} {r['zone']:28} CSI={r['CSI']:.4f} POD={r['POD']:.4f} "
              f"FAR={r['FAR']:.4f} Bias={r['Bias']:.4f} H={r['H']} F={r['F']} M={r['M']}")
    _figures(score, o, valid, obs, maps, auc, label, args.event)
    _write_compare()
    return 0


def _roc_curve(p: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p, kind="mergesort")
    y = o.astype(bool)[order]
    n1 = max(int(y.sum()), 1)
    n0 = max(int((~y).sum()), 1)
    tpr = np.concatenate([[0.0], np.cumsum(y) / n1])
    fpr = np.concatenate([[0.0], np.cumsum(~y) / n0])
    return fpr, tpr


def _write_compare() -> None:
    """One table of every scored event. Missing events are not filled in."""
    paths = [Path("results/v0_diagnostics.csv"), Path("results/val_4198_diagnostics.csv")]
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    out = Path("results/validation_compare.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, "rows", len(rows))


def _figures(score, observed, valid, obs, maps, auc, label: str, event_id: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    figdir = Path("paper/shared/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    step = max(len(score[valid]) // 2000, 1)
    fpr, tpr = _roc_curve(score[valid], observed[valid].astype(np.float64))
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    ax.plot(fpr[::step], tpr[::step], color="#0072B2", label=f"{label} mean hmax, AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#4D4D4D", ls="--", lw=0.8, label="no skill")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.legend(fontsize=11)
    roc_name = "F_v0_roc.png" if event_id == 2734 else f"F_{event_id}_roc.png"
    fig.savefig(figdir / roc_name, dpi=140)
    fig.savefig(Path("results/figures") / roc_name, dpi=140)
    plt.close(fig)

    cmap = ListedColormap(["#ffffff", "#e8e8e8", "#1f77b4", "#ff7f0e", "#d62728"])
    obs_sel = (obs == 1) & valid
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), constrained_layout=True)
    for ax, thr in zip(axes, (0.10, 0.25)):
        m = maps[thr] == 1
        img = np.zeros(obs.shape, dtype=np.uint8)
        img[valid] = 1
        img[valid & m & obs_sel] = 2
        img[valid & ~m & obs_sel] = 3
        img[valid & m & ~obs_sel] = 4
        ax.imshow(img, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.set_xlabel(f"threshold {thr:.2f} m")
        ax.xaxis.label.set_size(12)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in (1, 2, 3, 4)]
    fig.legend(handles, ["correct negative", "hit", "miss", "false alarm"], loc="lower center", ncol=4, fontsize=11)
    conf_name = "F15_v0_confusion.png" if event_id == 2734 else f"F15_{event_id}_confusion.png"
    fig.savefig(figdir / conf_name, dpi=120)
    fig.savefig(Path("results/figures") / conf_name, dpi=120)
    plt.close(fig)
    print("wrote", roc_name, "and", conf_name)


if __name__ == "__main__":
    raise SystemExit(main())
