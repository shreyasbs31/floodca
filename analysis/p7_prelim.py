"""Preliminary F9 and F10 from the finished P-7 jobs only.

Jobs: 242994 (E4 S,M), 242995 (E4 serial L), 242998 (E6), 242999 (E7).
Rows from any other job are ignored. Missing grids are left missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.plots import KEYS, fig_f9, fig_f10, load, summarise

JOBS = {242994, 242995, 242998, 242999}
NOTE = ", preliminary"


def main() -> int:
    raw = load(Path("results/bench.csv"))
    df = raw[raw["slurm_job_id"].astype(int).isin(JOBS)].copy()
    resdir = Path("results")
    figdir = Path("paper/shared/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(resdir / "p7_prelim_rows.csv", index=False)

    var = summarise(df)
    jobmap = (
        df.groupby(KEYS, dropna=False)["slurm_job_id"]
        .agg(lambda s: ",".join(str(int(x)) for x in sorted(set(s))))
        .reset_index(name="slurm_jobs")
    )
    var = var.merge(jobmap, on=KEYS, how="left")
    var.to_csv(resdir / "p7_prelim_variability.csv", index=False)
    flagged = var[var["rerun_flag"] & (var["n"] >= 2)]
    flagged.to_csv(resdir / "p7_prelim_variability_over_10pct.csv", index=False)

    e4 = df[df["exp_id"] == "E4"]
    s4 = summarise(e4)
    rows = []
    for grid in ["S", "M", "L"]:
        ser = s4[(s4["grid"] == grid) & (s4["backend"] == "serial")]
        omp = s4[(s4["grid"] == grid) & (s4["backend"] == "omp")].sort_values("threads")
        if ser.empty:
            continue
        ts = float(ser["median"].iloc[0])
        if omp.empty:
            rows.append({
                "grid": grid, "threads": 1, "backend": "serial", "T_median_s": ts,
                "speedup": None, "efficiency": None,
                "iqr_over_median": float(ser["iqr_over_median"].iloc[0]),
                "n": int(ser["n"].iloc[0]), "note": "OpenMP for this grid has not finished",
            })
            continue
        for _, r in omp.iterrows():
            sp = ts / float(r["median"])
            rows.append({
                "grid": grid, "threads": int(r["threads"]), "backend": "omp",
                "T_serial_s": ts, "T_median_s": float(r["median"]),
                "speedup": sp, "efficiency": sp / float(r["threads"]),
                "iqr_over_median": float(r["iqr_over_median"]), "n": int(r["n"]),
                "serial_iqr_over_median": float(ser["iqr_over_median"].iloc[0]),
            })
    pd.DataFrame(rows).to_csv(resdir / "p7_prelim_e4.csv", index=False)

    fig_f9(df, figdir, resdir, NOTE)
    fig_f10(df, figdir, resdir, NOTE)
    digest = {
        "n_rows": int(len(df)),
        "jobs": sorted(int(x) for x in df["slurm_job_id"].unique()),
        "exp_counts": {k: int(v) for k, v in df.groupby("exp_id").size().items()},
        "flagged_n": int(len(flagged)),
        "e4_grids": sorted(e4["grid"].unique()),
        "e6_grids": sorted(df.loc[df["exp_id"] == "E6", "grid"].unique()),
        "e7_grids": sorted(df.loc[df["exp_id"] == "E7", "grid"].unique()),
    }
    if (resdir / "p7_e7_crossover.csv").is_file():
        digest["crossover"] = pd.read_csv(resdir / "p7_e7_crossover.csv").to_dict(orient="records")
    if (resdir / "p7_e7_summary.csv").is_file():
        digest["e7_grids_in_figure"] = pd.read_csv(resdir / "p7_e7_summary.csv")["grid"].tolist()
    Path(resdir / "p7_prelim_digest.json").write_text(json.dumps(digest, indent=2) + "\n")
    print(json.dumps(digest, indent=2))
    print("FLAGGED")
    if flagged.empty:
        print("none")
    else:
        cols = ["exp_id", "backend", "grid", "precision", "threads", "block", "reduce_impl",
                "n", "median", "iqr_over_median"]
        print(flagged[cols].to_string(index=False))
    print("P7_PRELIM_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
