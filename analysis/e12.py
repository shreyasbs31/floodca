"""E12 probabilistic products (SPEC 6.4, 6.5, 11): F16, F17, F20.

Main ensemble (N=500, 90 m, B6 counts in nested chunks): P_N[a] = count_N[a] / N on
LAND for N in {8, 16, 32, 64, 128, 250, 500}, MAD(N) = mean over LAND |P_N - P_500|,
and the smallest N with MAD < 0.01. Because most LAND cells never flood, MAD is also
reported over the LAND cells with P_500 > 0 (MAD_wet, a supplementary diagnostic).

Validation ensemble (N=64, per-member hmax): each member's flood flag is taken to the
250 m GFD 2734 grid exactly as in Sec 10.1 (area-weighted average, fraction >= 0.5),
P is the member mean, and it is scored against the observed map over the valid cells
with the Brier score, ROC AUC (Mann-Whitney, ties counted half) and a 10-bin
reliability table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from floodca_prep.io_bin import read_bin
from floodca_prep.validation_metrics import exclusion_mask, model_flood_on_obs_grid, resample_nearest_u8

THR = (("010", 0.10), ("025", 0.25), ("050", 0.50))
NS = (8, 16, 32, 64, 128, 250, 500)
CHUNKS = ((0, 8), (8, 8), (16, 16), (32, 32), (64, 64), (128, 122), (250, 250))
H_DEFAULT = "025"


def _chunk_counts(main: Path, tag: str, n_cells: int) -> list[np.ndarray]:
    out = []
    for first, n in CHUNKS:
        f = main / f"chunk_{first:03d}_{n:03d}" / f"count_{tag}.bin"
        out.append(np.fromfile(f, dtype=np.uint16).astype(np.int64) if f.is_file() else None)
        if out[-1] is not None and out[-1].size != n_cells:
            raise ValueError(f"{f}: {out[-1].size} cells, expected {n_cells}")
    return out


def auc_mann_whitney(p: np.ndarray, o: np.ndarray) -> float:
    from scipy.stats import rankdata

    pos = o.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_points(p: np.ndarray, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = o.astype(bool)
    n1, n0 = max(int(pos.sum()), 1), max(int((~pos).sum()), 1)
    levels = np.unique(p)[::-1]
    tpr = [0.0] + [float((pos & (p >= t)).sum()) / n1 for t in levels]
    fpr = [0.0] + [float((~pos & (p >= t)).sum()) / n0 for t in levels]
    return np.array(fpr), np.array(tpr)


def reliability(p: np.ndarray, o: np.ndarray, nbins: int = 10) -> list[dict]:
    idx = np.minimum((p * nbins).astype(int), nbins - 1)
    rows = []
    for b in range(nbins):
        sel = idx == b
        n = int(sel.sum())
        rows.append({"bin": b, "p_lo": b / nbins, "p_hi": (b + 1) / nbins, "n": n,
                     "mean_p": float(p[sel].mean()) if n else float("nan"),
                     "obs_freq": float(o[sel].mean()) if n else float("nan")})
    return rows


def score_members(proc: Path, mhdr, mask90, members: list[Path]) -> tuple[dict, list, dict]:
    """Score one validation ensemble. Returns (summary by threshold, reliability rows, ROC)."""
    if not members:
        return {}, [], {}
    obs, obs_hdr = read_bin(proc / "validation" / "gfd_2734_flooded_250.bin")
    perm, _ = read_bin(proc / "validation" / "gfd_2734_jrc_perm_water_250.bin")
    shape, dx_o = obs.shape, float(obs_hdr["dx"])
    x0_o, y0_o = float(obs_hdr["x0"]), float(obs_hdr["y0"])
    sea_wall = resample_nearest_u8(((mask90 == 1) | (mask90 == 2)).astype(np.uint8), 90.0,
                                   float(mhdr["x0"]), float(mhdr["y0"]), shape, dx_o, x0_o, y0_o)
    valid = exclusion_mask(perm, sea_wall.astype(bool), clear_perc=None)
    o = (obs[valid] == 1).astype(np.float64)
    acc = {tag: np.zeros(shape, dtype=np.float64) for tag, _ in THR}
    for mpath in members:
        hmax, _ = read_bin(mpath)
        for tag, thr in THR:
            m90 = ((hmax >= thr) & (mask90 == 0)).astype(np.uint8)
            acc[tag] += model_flood_on_obs_grid(m90, dict(mhdr), shape, dx_o, x0_o, y0_o, 0.5)
    obar = float(o.mean())
    summary, rel_rows, roc = {}, [], {}
    for tag, thr in THR:
        p = (acc[tag] / len(members))[valid]
        bs = float(np.mean((p - o) ** 2))
        bs_ref = obar * (1.0 - obar)
        summary[tag] = {"threshold_m": thr, "n_members": len(members), "valid_cells": int(valid.sum()),
                        "observed_flooded": int(o.sum()), "base_rate": obar, "Brier": bs,
                        "Brier_climatology": bs_ref, "BSS": 1.0 - bs / bs_ref if bs_ref > 0 else None,
                        "ROC_AUC": auc_mann_whitney(p, o), "cells_P_gt_0": int((p > 0).sum()),
                        "observed_flooded_with_P_gt_0": int(((p > 0) & (o == 1)).sum())}
        for r in reliability(p, o):
            rel_rows.append({"threshold_m": thr, **r})
        roc[tag] = roc_points(p, o)
    return summary, rel_rows, roc


def _f20(figdir: Path, summary: dict, rel_rows: list, roc: dict, title: str, filename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    for tag, thr in THR:
        if tag not in summary:
            continue
        rr = [r for r in rel_rows if r["threshold_m"] == thr and r["n"] > 0]
        color = {0.10: "#0072B2", 0.25: "#D55E00", 0.50: "#009E73"}[thr]
        a1.plot([r["mean_p"] for r in rr], [r["obs_freq"] for r in rr], marker="o", color=color, label=f"{thr:.2f} m")
        fpr, tpr = roc[tag]
        a2.plot(fpr, tpr, marker=".", color=color, label=f"{thr:.2f} m, AUC={summary[tag]['ROC_AUC']:.3f}")
    a1.plot([0, 1], [0, 1], color="#4D4D4D", ls="--", lw=0.8)
    a1.set_xlabel("forecast probability (bin mean)")
    a1.set_ylabel("observed frequency")
    a1.tick_params(labelsize=11)
    a1.xaxis.label.set_size(12)
    a1.yaxis.label.set_size(12)
    a1.legend(fontsize=11)
    a2.plot([0, 1], [0, 1], color="#4D4D4D", ls="--", lw=0.8)
    a2.set_xlabel("false positive rate")
    a2.set_ylabel("true positive rate")
    a2.tick_params(labelsize=11)
    a2.xaxis.label.set_size(12)
    a2.yaxis.label.set_size(12)
    a2.legend(fontsize=11)
    fig.savefig(figdir / filename, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--runs", default="results/e12")
    ap.add_argument("--figdir", default="paper/shared/figures")
    args = ap.parse_args()
    proc, runs, figdir = Path(args.proc), Path(args.runs), Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    mask90, mhdr = read_bin(proc / "90" / "mask.bin")
    land = (mask90 == 0).ravel()
    shape90 = mask90.shape

    # ---- main ensemble: convergence and probability map ----
    conv_rows, p500, main_summary = [], {}, {}
    for tag, thr in THR:
        chunks = _chunk_counts(runs / "main", tag, mask90.size)
        cum, avail, ncum = np.zeros(mask90.size, dtype=np.int64), {}, 0
        for (first, n), c in zip(CHUNKS, chunks):
            if c is None:
                break
            cum += c
            ncum = first + n
            avail[ncum] = cum.copy()
        if 500 not in avail:
            print(f"main ensemble incomplete for {tag}: have N={sorted(avail)}")
            continue
        p500[tag] = avail[500] / 500.0
        wet = land & (p500[tag] > 0)
        for N in NS:
            err = np.abs(avail[N] / N - p500[tag])
            conv_rows.append({"threshold_m": thr, "N": N, "MAD": float(err[land].mean()),
                              "MAD_wet": float(err[wet].mean()) if wet.any() else 0.0})
        mine = [r for r in conv_rows if r["threshold_m"] == thr and r["N"] < 500]
        below = [r["N"] for r in mine if r["MAD"] < 0.01]
        below_wet = [r["N"] for r in mine if r["MAD_wet"] < 0.01]
        pl = p500[tag][land]
        main_summary[tag] = {"threshold_m": thr, "N_first_MAD_below_0.01": min(below) if below else None,
                             "N_first_MAD_wet_below_0.01": min(below_wet) if below_wet else None,
                             "land_cells": int(land.sum()), "cells_P_gt_0": int((pl > 0).sum()),
                             "cells_P_ge_0.5": int((pl >= 0.5).sum()), "cells_P_eq_1": int((pl >= 1).sum()),
                             "mean_P_land": float(pl.mean())}
    if conv_rows:
        with open("results/e12_convergence.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
            w.writeheader()
            w.writerows(conv_rows)

    # ---- validation ensembles (24 h reading, and the full-event reading when present) ----
    def _members(name: str) -> list[Path]:
        d = runs / name
        return sorted(d.glob("scen_*/hmax.bin")) if d.is_dir() else []

    val_summary, rel_rows, roc = score_members(proc, mhdr, mask90, _members("val"))
    full_summary, full_rel, full_roc = score_members(proc, mhdr, mask90, _members("val_full"))
    all_rel = [{"ensemble": "validation_24h", **r} for r in rel_rows]
    all_rel += [{"ensemble": "validation_full_event", **r} for r in full_rel]
    if all_rel:
        with open("results/e12_reliability.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rel[0].keys()))
            w.writeheader()
            w.writerows(all_rel)
    elif not val_summary:
        print(f"validation ensemble missing under {runs / 'val'}")

    summary = {"main_N500": main_summary, "validation_N64": val_summary,
               "validation_full_event": full_summary,
               "note": "validation_N64 is the 24 h single-day reading. validation_full_event is the "
                       "23-day V0 forcing with each day scaled by the member lambda. n_members is the "
                       "number of hmax files present, which is 64 only when that ensemble has finished."}
    Path("results/e12_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if p500:
        fig, axes = plt.subplots(1, len(p500), figsize=(5.2 * len(p500), 6.0), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (tag, thr) in zip(axes, [t for t in THR if t[0] in p500]):
            img = np.ma.masked_where(mask90 != 0, p500[tag].reshape(shape90))
            ax.set_facecolor("#cfe3f3")
            im = ax.imshow(img, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xlabel(f"threshold {thr:.2f} m")
            ax.xaxis.label.set_size(12)
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="probability of flooding")
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.yaxis.label.set_size(12)
        fig.savefig(figdir / "F16_probability_map.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
        for tag, thr in THR:
            mine = [r for r in conv_rows if r["threshold_m"] == thr and r["N"] < 500]
            if mine:
                color = {0.10: "#0072B2", 0.25: "#D55E00", 0.50: "#009E73"}[thr]
                ax.plot([r["N"] for r in mine], [r["MAD"] for r in mine], marker="o", color=color,
                        label=f"{thr:.2f} m, all LAND")
                ax.plot([r["N"] for r in mine], [r["MAD_wet"] for r in mine], marker="s", ls=":",
                        color=color, label=f"{thr:.2f} m, LAND with P_500 > 0")
        ax.axhline(0.01, color="#4D4D4D", ls="--", lw=0.8, label="MAD = 0.01")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([n for n in NS if n < 500])
        ax.set_xticklabels([str(n) for n in NS if n < 500])
        ax.set_xlabel("ensemble size N (first N of 500)")
        ax.set_ylabel("MAD(N) over LAND vs N=500")
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.legend(fontsize=11)
        fig.savefig(figdir / "F17_convergence.png", dpi=120)
        plt.close(fig)

    if val_summary:
        _f20(figdir, val_summary, rel_rows, roc,
             "F20: 24 h validation ensemble vs GFD 2734 (250 m comparison grid)",
             "F20_validation_reliability_roc.png")
    if full_summary:
        n = next(iter(full_summary.values()))["n_members"]
        _f20(figdir, full_summary, full_rel, full_roc,
             f"F20 full-event validation ({n} members) vs GFD 2734 (250 m comparison grid)",
             "F20_validation_full_event.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
