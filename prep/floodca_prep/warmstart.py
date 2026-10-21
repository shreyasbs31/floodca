"""B0 warm-start states h0_B0_<grid>.bin (SPEC Sec 6.3).

For each benchmark grid: run B0 (base_date = largest p_mean_mm pool date, lambda 1,
D 24 h, phi 0.5, triangle hyetograph) from a dry start for 6 simulated hours, then
store the final depth as float32. Each grid runs in its own directory of symlinks
to dem/mask/n/C, so no stray initial-depth file (h.bin) is picked up.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

GRIDS = ["S", "M", "L", "30", "90"]
STATIC = ["dem", "mask", "n", "C"]
T_WARM = 21600


def dry_grid_dir(proc: Path, grid: str) -> Path:
    d = proc / "warmstart" / f"grid_{grid}"
    d.mkdir(parents=True, exist_ok=True)
    for name in STATIC:
        for ext in (".bin", ".json"):
            src = (proc / grid / f"{name}{ext}").resolve()
            dst = d / f"{name}{ext}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)
    return d


def run_grid(proc: Path, grid: str, exe: str, threads: int, b0: dict, neigh: int, csv: str) -> dict:
    import numpy as np
    from floodca_prep.io_bin import read_bin, write_bin

    gdir = dry_grid_dir(proc, grid)
    rain = proc / "rain" / grid / f"rain_{b0['base_date']}.bin"
    if not rain.is_file():
        raise FileNotFoundError(f"{rain} missing; run floodca_prep.scenarios first")
    out = proc / "warmstart" / f"run_{grid}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, "--grid", str(gdir), "--rain", str(rain),
        "--lambda", str(b0["lambda"]), "--D_h", str(b0["D_h"]), "--phi", str(b0["phi"]),
        "--hyeto", "triangle", "--tsim", str(T_WARM), "--neigh", str(neigh), "--U", "1",
        "--threads", str(threads), "--diag_every", "1000", "--progress", "20000",
        "--out", str(out), "--exp", f"B0warm_{grid}", "--csv", csv,
    ]
    print("RUN", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    wall = time.time() - t0

    h, hh = read_bin(out / "h_final.bin")
    mask, _ = read_bin(proc / grid / "mask.bin")
    man = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    h32 = np.asarray(h, dtype=np.float32)
    land = mask == 0
    dx = float(hh["dx"])
    stats = {
        "grid": grid,
        "nrows": int(h.shape[0]),
        "ncols": int(h.shape[1]),
        "dx": dx,
        "steps": int(man["steps"]),
        "t": float(man["t"]),
        "E_mass": float(man["E_mass"]),
        "V_m3": float(np.sum(h[land], dtype=np.float64) * dx * dx),
        "V_rain_m3": float(man["V_rain"]),
        "wet_frac_land_tau_h": float(np.mean(h[land] >= 1e-3)),
        "wet_frac_land_0p10": float(np.mean(h[land] >= 0.10)),
        "max_h": float(h.max()),
        "wall_s": wall,
        "backend": man["backend"],
        "threads": int(man["threads"]),
        "neigh": neigh,
    }
    dst = proc / "warmstart" / f"h0_B0_{grid}.bin"
    hdr = {k: hh[k] for k in ("nrows", "ncols", "dx", "crs", "x0", "y0") if k in hh}
    hdr.update({"name": f"h0_B0_{grid}", "dtype": "float32", "units": "m",
                "source": f"B0 warm start, {T_WARM} s from dry", "b0": b0, "stats": stats})
    write_bin(dst, h32, hdr)
    stats["file"] = str(dst)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--exe", required=True, help="floodca_omp (double build)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--grids", default=",".join(GRIDS))
    ap.add_argument("--neigh", type=int, default=4)
    ap.add_argument("--csv", default="results/warmstart_runs.csv")
    args = ap.parse_args()
    proc = Path(args.proc)
    rep = json.loads((proc / "scenarios_N500_report.json").read_text(encoding="utf-8"))
    b0 = rep["b0"]
    out = {"status": "ok", "b0": b0, "T_warm_s": T_WARM, "grids": {}}
    report_path = proc / "warmstart" / "warmstart_report.json"
    if report_path.is_file():
        out["grids"] = json.loads(report_path.read_text(encoding="utf-8")).get("grids", {})
    for g in args.grids.split(","):
        out["grids"][g] = run_grid(proc, g, args.exe, args.threads, b0, args.neigh, args.csv)
        report_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out["grids"][g]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
