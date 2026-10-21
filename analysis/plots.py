"""Paper figures from results/bench.csv (SPEC Sec 11, 14.5). F7, F8, F9, F10, F11, F12, F13.

Rules (SPEC 11.1): warm-up rows and contaminated rows are dropped; each configuration
is summarised by the median and IQR of its repeats; a configuration whose IQR/median
exceeds 10% is flagged for a re-run. Summary tables go to analysis/out/summaries/.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/{os.environ.get('USER', 'floodca')}_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SUMMARY_DIR = Path("analysis/out/summaries")


def _summary(name: str) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    return SUMMARY_DIR / name


KEYS = ["exp_id", "backend", "grid", "precision", "neigh", "threads", "block", "reduce_impl"]
GRID_ORDER = ["C32", "C64", "C128", "C256", "S", "C768", "M", "C1536", "L", "C3072", "30"]
# One palette for F7, F9 and F10. Colour marks the grid or the series, not the figure.
GRID_COLOR = {
    "C32": "#999999", "C64": "#666666", "C128": "#56B4E9", "C256": "#CC79A7",
    "S": "#0072B2", "C768": "#D55E00", "M": "#E69F00", "C1536": "#F0E442", "L": "#009E73",
}
SERIES_COLOR = {
    "serial": "#4D4D4D", "omp": "#0072B2", "cuda_e2e": "#D55E00", "cuda_kernel": "#E69F00",
}
PHASE_COLOR = ["#4D4D4D", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


# Figures are included at one IEEE column (about 3.5 in). Fonts are set so
# that size survives the inclusion. Two-panel plots are stacked, not stretched
# across a page and then shrunk.
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
})


def _style_ax(ax):
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.grid(alpha=0.3)


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype={"block": str, "reduce_impl": str, "notes": str})
    df = df[(df["warmup"] == 0) & (df["contaminated"] == 0)].copy()
    df["grid"] = df["grid_name"].astype(str).str.replace("grid_", "", regex=False)
    df["reduce_impl"] = df["reduce_impl"].fillna("")
    df["block"] = df["block"].fillna("")
    for c in ("t_h2d_s", "t_d2h_s"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["t_e2e_s"] = df["wall_total_s"] + df["t_h2d_s"].fillna(0) + df["t_d2h_s"].fillna(0)
    return df


def summarise(df: pd.DataFrame, col: str = "wall_total_s") -> pd.DataFrame:
    g = df.groupby(KEYS, dropna=False)[col]
    out = pd.DataFrame({
        "n": g.size(),
        "median": g.median(),
        "q25": g.quantile(0.25),
        "q75": g.quantile(0.75),
    }).reset_index()
    out["iqr_over_median"] = (out["q75"] - out["q25"]) / out["median"]
    out["rerun_flag"] = out["iqr_over_median"] > 0.10
    cells = df.groupby(KEYS, dropna=False)["active_cells"].first().reset_index(drop=True)
    out["active_cells"] = cells.values
    return out


def _order(grids):
    return sorted(grids, key=lambda g: GRID_ORDER.index(g) if g in GRID_ORDER else 99)


def fig_f7(df: pd.DataFrame, figdir: Path, resdir: Path, title_note: str = "") -> None:
    """Speedup against the serial binary (SPEC) and against OpenMP at 1 thread."""
    e4 = df[df["exp_id"] == "E4"]
    if e4.empty:
        print("F7: no E4 rows")
        return
    s = summarise(e4)
    rows = []
    fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.6))
    for grid in _order(s["grid"].unique()):
        ser = s[(s["grid"] == grid) & (s["backend"] == "serial")]
        omp = s[(s["grid"] == grid) & (s["backend"] == "omp")].sort_values("threads")
        one = omp[omp["threads"] == 1]
        if ser.empty or omp.empty or one.empty:
            continue
        ts = float(ser["median"].iloc[0])
        t1 = float(one["median"].iloc[0])
        p = omp["threads"].to_numpy(dtype=float)
        med = omp["median"].to_numpy()
        sp_s = ts / med
        sp_1 = t1 / med
        lo_s, hi_s = ts / omp["q75"].to_numpy(), ts / omp["q25"].to_numpy()
        lo_1, hi_1 = t1 / omp["q75"].to_numpy(), t1 / omp["q25"].to_numpy()
        eff_s, eff_1 = sp_s / p, sp_1 / p
        kf_s = np.where(p > 1, (1.0 / sp_s - 1.0 / p) / (1.0 - 1.0 / p), np.nan)
        kf_1 = np.where(p > 1, (1.0 / sp_1 - 1.0 / p) / (1.0 - 1.0 / p), np.nan)
        c = GRID_COLOR.get(grid, "#333333")
        kw = dict(marker="o", capsize=2, color=c, label=grid)
        ax[0, 0].errorbar(p, sp_s, yerr=[sp_s - lo_s, hi_s - sp_s], **kw)
        ax[0, 1].errorbar(p, eff_s, yerr=[(sp_s - lo_s) / p, (hi_s - sp_s) / p], **kw)
        ax[1, 0].errorbar(p, sp_1, yerr=[sp_1 - lo_1, hi_1 - sp_1], **kw)
        ax[1, 1].errorbar(p, eff_1, yerr=[(sp_1 - lo_1) / p, (hi_1 - sp_1) / p], **kw)
        for i in range(len(p)):
            rows.append({
                "grid": grid, "threads": int(p[i]),
                "T_serial_s": ts, "T_omp1_s": t1, "T_median_s": float(med[i]),
                "speedup_vs_serial": float(sp_s[i]), "efficiency_vs_serial": float(eff_s[i]),
                "karp_flatt_vs_serial": float(kf_s[i]) if p[i] > 1 else None,
                "speedup_vs_omp1": float(sp_1[i]), "efficiency_vs_omp1": float(eff_1[i]),
                "karp_flatt_vs_omp1": float(kf_1[i]) if p[i] > 1 else None,
                "rerun_flag": bool(omp["rerun_flag"].iloc[i] or ser["rerun_flag"].iloc[0]
                                   or one["rerun_flag"].iloc[0]),
            })
    pmax = float(e4["threads"].max()) if not e4.empty else 1.0
    for a in (ax[0, 0], ax[1, 0]):
        a.plot([1, pmax], [1, pmax], "k:", lw=0.8, label="ideal")
        a.set_xlabel("OpenMP threads")
    ax[0, 0].set_ylabel("speedup vs serial binary")
    ax[1, 0].set_ylabel("speedup vs OpenMP 1 thread")
    ax[0, 1].set_ylabel("efficiency vs serial binary")
    ax[1, 1].set_ylabel("efficiency vs OpenMP 1 thread")
    for a in ax.ravel():
        _style_ax(a)
        a.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(figdir / "fig5.pdf", dpi=150)
    pd.DataFrame(rows).to_csv(_summary("p7_e4_summary.csv"), index=False)
    print("F7 written;", len(rows), "rows")


def fig_f9(df: pd.DataFrame, figdir: Path, resdir: Path, title_note: str = "") -> None:
    e6 = df[df["exp_id"] == "E6"]
    e6t = df[df["exp_id"] == "E6t"]
    if e6.empty:
        print("F9: no E6 rows")
        return
    s = summarise(e6)
    steps = e6.groupby(KEYS, dropna=False)["steps"].first().reset_index(drop=True)
    s["mcups"] = s["active_cells"] * steps.values / s["median"] / 1e6
    s.to_csv(_summary("p7_e6_summary.csv"), index=False)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.6))
    phases = ["t_h2d_s", "t_flux_s", "t_limiter_s", "t_update_s", "t_dt_s", "t_d2h_s"]
    sel = e6t[(e6t["block"] == "16x16") & (e6t["reduce_impl"] == "cub") & (e6t["precision"] == "float")]
    if not sel.empty:
        med = sel.groupby("grid")[phases].median()
        med = med.loc[_order(med.index)]
        bottom = np.zeros(len(med))
        phase_label = {
            "t_h2d_s": "H2D", "t_flux_s": "flux", "t_limiter_s": "limiter",
            "t_update_s": "update", "t_dt_s": "reduce", "t_d2h_s": "D2H",
        }
        for i, ph in enumerate(phases):
            vals = med[ph].fillna(0).to_numpy()
            ax[0].bar(med.index, vals, bottom=bottom, label=phase_label[ph], color=PHASE_COLOR[i])
            bottom += vals
        ax[0].set_ylabel("time (s), 2000 steps")
        ax[0].legend(fontsize=6, ncol=3, loc="upper right")
    cub = s[s["reduce_impl"] == "cub"]
    for prec, sub in cub.groupby("precision"):
        for grid in ("S", "M", "L"):
            gg = sub[sub["grid"] == grid].sort_values(
                "block", key=lambda b: b.str.split("x").str[0].astype(int))
            if gg.empty:
                continue
            ax[1].plot(
                gg["block"], gg["mcups"],
                marker="o",
                ls="-" if prec == "float" else "--",
                color=GRID_COLOR[grid],
                label=f"{grid}, {'single' if prec == 'float' else 'double'}",
            )
    ax[1].set_xlabel("block")
    ax[1].set_ylabel("MCUPS")
    ax[1].legend(fontsize=6, ncol=2, loc="best", frameon=False)
    for a in ax:
        _style_ax(a)
    fig.tight_layout()
    fig.savefig(figdir / "fig7.pdf", dpi=150, bbox_inches="tight")
    print("F9 written")


def _crossover(x, y_cpu, y_gpu):
    """Interpolated cell count where log(y_cpu) - log(y_gpu) changes sign (CPU slower after)."""
    d = np.log(y_cpu) - np.log(y_gpu)
    for i in range(len(d) - 1):
        if d[i] <= 0 < d[i + 1]:
            lx0, lx1 = np.log(x[i]), np.log(x[i + 1])
            return float(np.exp(lx0 + (0 - d[i]) * (lx1 - lx0) / (d[i + 1] - d[i])))
    return float("nan")


def fig_f10(df: pd.DataFrame, figdir: Path, resdir: Path, title_note: str = "") -> None:
    cpu = df[df["exp_id"].isin(["E4", "E7"]) & df["backend"].isin(["serial", "omp"])]
    gpu = df[(df["exp_id"] == "E7") & (df["backend"] == "cuda")
             & (df["block"] == "16x16") & (df["reduce_impl"] == "cub")
             & (df["precision"] == "float")]
    if gpu.empty or cpu.empty:
        print("F10: needs E7 CUDA rows and E4/E7 CPU rows")
        return
    s_cpu = summarise(cpu)
    s_gk = summarise(gpu, "wall_total_s")
    s_ge = summarise(gpu, "t_e2e_s")
    rows = []
    for grid in _order(s_ge["grid"].unique()):
        ser = s_cpu[(s_cpu["grid"] == grid) & (s_cpu["backend"] == "serial")]
        omp = s_cpu[(s_cpu["grid"] == grid) & (s_cpu["backend"] == "omp")]
        if ser.empty or omp.empty:
            continue
        best = omp.loc[omp["median"].idxmin()]
        rows.append({
            "grid": grid,
            "active_cells": int(s_ge[s_ge["grid"] == grid]["active_cells"].iloc[0]),
            "serial_s": float(ser["median"].min()),
            "best_omp_s": float(best["median"]),
            "best_omp_threads": int(best["threads"]),
            "cuda_e2e_s": float(s_ge[s_ge["grid"] == grid]["median"].iloc[0]),
            "cuda_kernel_s": float(s_gk[s_gk["grid"] == grid]["median"].iloc[0]),
        })
    t = pd.DataFrame(rows).sort_values("active_cells")
    if t.empty:
        print("F10: no grid has all of serial, OpenMP and CUDA")
        return
    x = t["active_cells"].to_numpy(dtype=float)
    cross = {
        "omp_vs_cuda_e2e": _crossover(x, t["best_omp_s"].to_numpy(), t["cuda_e2e_s"].to_numpy()),
        "serial_vs_cuda_e2e": _crossover(x, t["serial_s"].to_numpy(), t["cuda_e2e_s"].to_numpy()),
        "omp_vs_cuda_kernel": _crossover(x, t["best_omp_s"].to_numpy(), t["cuda_kernel_s"].to_numpy()),
    }
    t.to_csv(_summary("p7_e7_summary.csv"), index=False)
    pd.DataFrame([cross]).to_csv(_summary("p7_e7_crossover.csv"), index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    series = (
        ("serial_s", "serial", SERIES_COLOR["serial"]),
        ("best_omp_s", "best OpenMP", SERIES_COLOR["omp"]),
        ("cuda_e2e_s", "CUDA end-to-end", SERIES_COLOR["cuda_e2e"]),
        ("cuda_kernel_s", "CUDA kernel only", SERIES_COLOR["cuda_kernel"]),
    )
    styles = {
        "cuda_kernel_s": dict(marker="s", ls="--", markersize=5),
    }
    for col, lab, color in series:
        kw = styles.get(col, dict(marker="o", ls="-"))
        ax.loglog(x, t[col], label=lab, color=color, **kw)
    if np.isfinite(cross["omp_vs_cuda_e2e"]):
        ax.axvline(cross["omp_vs_cuda_e2e"], color="k", ls=":", lw=0.8,
                   label=f"crossover {cross['omp_vs_cuda_e2e']:.3g} cells")
    ax.set_xlabel("active cells")
    ax.set_ylabel("time for 2000 steps (s)")
    _style_ax(ax)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(figdir / "fig8.pdf", dpi=150)
    print("F10 written; crossover", cross)


def fig_f8(df: pd.DataFrame, figdir: Path, resdir: Path) -> None:
    e5 = df[df["exp_id"] == "E5"]
    if e5.empty:
        print("F8: no E5 rows")
        return
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.2))
    strong = e5[e5["grid"] == "L"].copy()
    if not strong.empty:
        strong["ranks"] = strong["ranks"].astype(int)
    rows = []
    if not strong.empty:
        g = strong.groupby("ranks")["wall_total_s"]
        med = g.median().sort_index()
        t1 = float(med.loc[1]) if 1 in med.index else float("nan")
        sp = t1 / med
        ax[0].plot(sp.index, sp.values, marker="o", color=SERIES_COLOR["omp"], label="L strong")
        ax[0].plot(sp.index, sp.index, color=SERIES_COLOR["serial"], ls=":", lw=0.8, label="ideal")
        for p, t in med.items():
            rows.append({"scaling": "strong", "grid": "L", "ranks": int(p), "T_median_s": float(t),
                         "speedup": float(t1 / t) if t else float("nan")})
    weak = e5[e5["grid"].astype(str).str.startswith("W")].copy()
    if not weak.empty:
        wmed = weak.groupby("grid")["wall_total_s"].median()
        order = [g for g in ("W1", "W2", "W4", "W8", "W16") if g in wmed.index]
        if "W1" in wmed.index:
            ew = float(wmed["W1"]) / wmed.loc[order]
            ax[1].plot([int(g[1:]) for g in order], ew.values, marker="o", color=SERIES_COLOR["omp"])
            for gname in order:
                rows.append({"scaling": "weak", "grid": gname, "ranks": int(gname[1:]),
                             "T_median_s": float(wmed[gname]),
                             "efficiency": float(wmed["W1"] / wmed[gname])})
    ax[0].set_xlabel("MPI ranks")
    ax[0].set_ylabel("speedup vs P=1")
    ax[0].legend(fontsize=6.5)
    ax[1].set_xlabel("MPI ranks")
    ax[1].set_ylabel("weak efficiency")
    ax[1].set_ylim(0, 1.2)
    for a in ax:
        _style_ax(a)
    fig.tight_layout()
    fig.savefig(figdir / "fig6.pdf", dpi=150)
    pd.DataFrame(rows).to_csv(_summary("p10_e5_summary.csv"), index=False)
    print("F8 written;", len(rows), "rows")


def fig_f11(df: pd.DataFrame, figdir: Path, resdir: Path) -> None:
    e8 = df[df["exp_id"] == "E8"].copy()
    if e8.empty:
        print("F11: no E8 rows")
        return
    e8["ranks"] = e8["ranks"].astype(int)
    e8["U"] = e8["U"].astype(int)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.2))
    rows = []
    line_colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#4D4D4D"]
    for i, ((backend, ranks), sub) in enumerate(e8.groupby(["backend", "ranks"])):
        g = sub.groupby("U")[["wall_total_s", "t_dt_s"]].median().sort_index()
        label = f"{backend} P={int(ranks)}" if backend == "mpi" else backend
        color = line_colors[i % len(line_colors)]
        ax[0].plot(g.index, g["wall_total_s"], marker="o", color=color, label=label)
        for u, r in g.iterrows():
            rows.append({"backend": backend, "ranks": int(ranks), "U": int(u),
                         "T_median_s": float(r["wall_total_s"]),
                         "dt_fraction": float(r["t_dt_s"] / r["wall_total_s"]) if r["wall_total_s"] else float("nan")})
    # Right panel: measured step share at U=1 against the measured U=50 wall saving.
    # CUDA share is the isolated reduction (0.481 s of 2.056 s), not the blocking timer.
    labels = ["OpenMP\n40 thr", "MPI\n4 ranks", "MPI\n8 ranks", "CUDA"]
    share = [32.1, 33.2, 33.6, 23.4]
    saving = [31.5, 32.5, 32.8, 23.2]
    xpos = np.arange(len(labels))
    ax[1].bar(xpos - 0.18, share, width=0.36, color=SERIES_COLOR["omp"], label="step share at U=1")
    ax[1].bar(xpos + 0.18, saving, width=0.36, color=SERIES_COLOR["cuda_e2e"], label="U=50 run-time saving")
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(labels)
    ax[0].set_xlabel("reuse interval U", labelpad=6)
    ax[0].set_ylabel("median run time (s)")
    ax[0].set_yscale("log")
    ax[1].set_ylabel("percent")
    _style_ax(ax[0])
    _style_ax(ax[1])
    ax[0].legend(fontsize=6.5, loc="center", frameon=True)
    ax[1].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                 ncol=2, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, hspace=0.45)
    fig.savefig(figdir / "fig9.pdf", dpi=150, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(_summary("p10_e8_summary.csv"), index=False)
    print("F11 written;", len(rows), "rows")


def fig_f12(df: pd.DataFrame, figdir: Path, resdir: Path) -> None:
    e9 = df[df["exp_id"] == "E9"].copy()
    if e9.empty:
        print("F12: no E9 rows")
        return
    e9["ranks"] = e9["ranks"].astype(int)
    e9["threads"] = e9["threads"].astype(int)
    e9["neigh"] = e9["neigh"].astype(int)
    if "halo_mode" not in e9.columns:
        e9["halo_mode"] = ""
    e9["halo_mode"] = e9["halo_mode"].fillna("")
    g = e9.groupby(["backend", "threads", "ranks", "neigh", "halo_mode"])["wall_total_s"].median().reset_index()
    g.to_csv(_summary("p10_e9_summary.csv"), index=False)
    specs = (
        ("omp", 40, 1, "", "OpenMP\n40 threads"),
        ("mpi", 1, 4, "sendrecv2", "MPI\n4 ranks"),
        ("cuda", 1, 1, "", "CUDA"),
    )
    labels, ratios = [], []
    for backend, threads, ranks, halo, label in specs:
        a = g[(g["backend"] == backend) & (g["threads"] == threads) & (g["ranks"] == ranks)
              & (g["halo_mode"] == halo) & (g["neigh"] == 4)]
        b = g[(g["backend"] == backend) & (g["threads"] == threads) & (g["ranks"] == ranks)
              & (g["halo_mode"] == halo) & (g["neigh"] == 8)]
        if a.empty or b.empty:
            continue
        labels.append(label)
        ratios.append(float(b["wall_total_s"].iloc[0]) / float(a["wall_total_s"].iloc[0]))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(labels))
    ax.bar(x, ratios, color="#0072B2", width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("K=8 / K=4 run time")
    ax.set_ylim(0, max(ratios) * 1.25 if ratios else 1)
    for i, val in enumerate(ratios):
        ax.text(i, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(figdir / "fig10.pdf", dpi=150)
    print("F12 written;", list(zip(labels, ratios)))


def fig_f13(df: pd.DataFrame, figdir: Path, resdir: Path) -> None:
    """Scenarios per hour from the finished E10 CPU and GPU tables.

    Energy rows are not the throughput figure. A crashed energy wrapper stays
    out of results/e11_table.csv. The bars are the CPU configurations in
    results/e10_cpu.csv plus the 1-GPU and 2-GPU rows in results/e10_gpu100.csv.
    """
    energy = resdir / "e11_energy.csv"
    if energy.is_file():
        e = pd.read_csv(energy)
        exit_code = pd.to_numeric(e["exit_code"], errors="coerce") if "exit_code" in e.columns else 0
        scenarios = pd.to_numeric(e["scenarios"], errors="coerce")
        valid = e[(exit_code.fillna(1) == 0) & (scenarios.fillna(0) > 0)].copy()
        valid.to_csv(_summary("e11_table.csv"), index=False)
    out_pdf = figdir / "fig11.pdf"
    labels, rates, colors = [], [], []
    cpu_path = resdir / "e10_cpu.csv"
    cpu_order = [
        ("serial", "serial", SERIES_COLOR["serial"]),
        ("b5_1x40", "1 x 40", SERIES_COLOR["omp"]),
        ("b5_2x20", "2 x 20", SERIES_COLOR["omp"]),
        ("b5_4x10", "4 x 10", SERIES_COLOR["omp"]),
        ("b5_8x5", "8 x 5", SERIES_COLOR["omp"]),
    ]
    if cpu_path.is_file():
        cpu = pd.read_csv(cpu_path)
        for cfg, lab, color in cpu_order:
            sub = cpu[cpu["config"].astype(str) == cfg]
            if sub.empty:
                continue
            r = sub.iloc[0]
            n = float(r["scenarios"])
            wall = float(r["wall_s"])
            rate = float(r["scen_per_hour"])
            if n <= 0 or wall <= 0 or rate <= 0:
                continue
            labels.append(lab)
            rates.append(rate)
            colors.append(color)
    gpu_path = resdir / "e10_gpu100.csv"
    if gpu_path.is_file():
        gpu = pd.read_csv(gpu_path)
        for cfg, lab in (("B4", "1 GPU"), ("B6a", "2 GPU")):
            sub = gpu[gpu["config"].astype(str) == cfg]
            if sub.empty:
                continue
            r = sub.iloc[0]
            n = float(r["scenarios"])
            wall = float(r["wall_s"])
            rate = float(r["scen_per_hour"])
            if n <= 0 or wall <= 0 or rate <= 0:
                continue
            labels.append(lab)
            rates.append(rate)
            colors.append(SERIES_COLOR["cuda_e2e"])
    if not labels:
        if out_pdf.is_file():
            out_pdf.unlink()
        print("F13: no valid E10 throughput rows; figure left out")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(labels, rates, color=colors)
    ax.set_yscale("log")
    ax.set_ylim(1, max(rates) * 1.8)
    ax.set_ylabel("scenarios per hour")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150)
    print("F13 written;", len(labels), "rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/bench.csv")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--resdir", default="results")
    ap.add_argument("--only", default="F7,F8,F9,F10,F11,F12,F13")
    ap.add_argument("--jobs", default="", help="comma-separated Slurm job ids to keep")
    args = ap.parse_args()
    df = load(Path(args.csv))
    if args.jobs.strip():
        keep = {int(x) for x in args.jobs.split(",") if x.strip()}
        ids = pd.to_numeric(df["slurm_job_id"], errors="coerce")
        df = df[ids.isin(keep)].copy()
    figdir, resdir = Path(args.figdir), Path(args.resdir)
    figdir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("F7", fig_f7), ("F8", fig_f8), ("F9", fig_f9), ("F10", fig_f10),
                     ("F11", fig_f11), ("F12", fig_f12), ("F13", fig_f13)):
        if name in args.only.split(","):
            fn(df, figdir, resdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
