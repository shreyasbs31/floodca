"""E9 anisotropy from an initial water column (no point source, no physics change).

Flat 1024x1024 plane, z = 10 m, all LAND, closed boundaries, no rain.
A 5x5 block at the centre starts at 1 m depth. The fixed volume cannot wet
40 percent of the grid at the 1e-3 m cutoff (see docs/DECISIONS.md), so the
run stops when the wet-cell count sets no new maximum for three chunks.
r_diag / r_axis is 1 for a circle. SPEC 10.4's r_axis / r_diag is stored too.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

N = 1024
THRESH = 1e-3
CHUNK = 400
STALL_CHUNKS = 3
STEP_CAP = 200000


def _write_grid(grid: Path) -> tuple[int, int]:
    grid.mkdir(parents=True, exist_ok=True)
    z = np.full((N, N), 10.0, dtype=np.float32)
    mask = np.zeros((N, N), dtype=np.uint8)
    z.tofile(grid / "dem.bin")
    mask.tofile(grid / "mask.bin")
    hdr = {
        "name": "dem_aniso", "nrows": N, "ncols": N, "dx": 90.0, "dtype": "float32",
        "crs": "EPSG:32643", "x0": 0.0, "y0": float(N * 90),
    }
    (grid / "dem.json").write_text(json.dumps(hdr) + "\n", encoding="utf-8")
    c0 = N // 2
    h0 = np.zeros((N, N), dtype=np.float32)
    h0[c0 - 2:c0 + 3, c0 - 2:c0 + 3] = 1.0
    h0.tofile(grid / "h0.bin")
    return c0, c0


def _radius(h: np.ndarray, r0: int, c0: int, dr: int, dc: int) -> tuple[float, bool]:
    r, c, last = r0, c0, 0
    hit = False
    while True:
        r += dr
        c += dc
        if r < 0 or c < 0 or r >= h.shape[0] or c >= h.shape[1]:
            hit = True
            break
        if h[r, c] > THRESH:
            last += 1
        else:
            break
    euclid = last * (2.0 ** 0.5 if dr and dc else 1.0)
    return euclid, hit


def _metrics(h: np.ndarray, r0: int, c0: int) -> dict:
    dirs = {
        "east": (0, 1), "west": (0, -1), "south": (1, 0), "north": (-1, 0),
        "se": (1, 1), "sw": (1, -1), "ne": (-1, 1), "nw": (-1, -1),
    }
    raw = {k: _radius(h, r0, c0, *v) for k, v in dirs.items()}
    axis = [raw[k][0] for k in ("east", "west", "south", "north")]
    diag = [raw[k][0] for k in ("se", "sw", "ne", "nw")]
    r_axis, r_diag = float(np.mean(axis)), float(np.mean(diag))
    wet_count = int((h > THRESH).sum())
    return {
        "wet_count": wet_count,
        "wet_frac": float(wet_count / h.size),
        "r_axis_cells": r_axis,
        "r_diag_cells": r_diag,
        "ratio_diag_over_axis": r_diag / r_axis if r_axis else None,
        "ratio_axis_over_diag": r_axis / r_diag if r_diag else None,
        "hit_boundary": any(v[1] for v in raw.values()),
        "directions_cells": {k: v[0] for k, v in raw.items()},
    }


def _read_h(path: Path) -> np.ndarray:
    hdr = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    dt = np.float64 if hdr.get("dtype") == "float64" else np.float32
    return np.fromfile(path, dtype=dt).reshape(int(hdr["nrows"]), int(hdr["ncols"]))


def _sim_time(text: str):
    last = None
    for line in text.splitlines():
        if " t=" not in line:
            continue
        for tok in line.split():
            if tok.startswith("t="):
                last = float(tok.split("=", 1)[1])
    return last


def run_k(exe: Path, grid: Path, work: Path, k: int, threads: int) -> dict:
    r0, c0 = N // 2, N // 2
    warm = work / f"warm_K{k}.bin"
    shutil.copyfile(grid / "h0.bin", warm)
    out = work / f"K{k}"
    steps = 0
    sim_time = None
    stall = 0
    peak = 0
    h = np.fromfile(warm, dtype=np.float32).reshape(N, N)
    metrics = _metrics(h, r0, c0)
    peak = metrics["wet_count"]
    stop = "cap"
    while steps < STEP_CAP and not metrics["hit_boundary"]:
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(exe), "--grid", str(grid), "--warmstart", str(warm), "--steps", str(CHUNK),
               "--neigh", str(k), "--U", "1", "--boundary", "closed", "--threads", str(threads),
               "--out", str(out), "--exp", "E9aniso"]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)
        print(proc.stdout, end="", flush=True)
        sim_time = _sim_time(proc.stdout)
        h = _read_h(out / "h_final.bin")
        h.astype(np.float32).tofile(warm)
        steps += CHUNK
        metrics = _metrics(h, r0, c0)
        if metrics["wet_count"] > peak:
            peak = metrics["wet_count"]
            stall = 0
        else:
            stall += 1
        print(f"K={k} steps={steps} wet={metrics['wet_frac']:.6f} "
              f"diag_over_axis={metrics['ratio_diag_over_axis']} stall={stall}",
              flush=True)
        if steps >= 5 * CHUNK and stall >= STALL_CHUNKS:
            stop = "wet_area_stopped_growing"
            break
    else:
        if metrics["hit_boundary"]:
            stop = "boundary"
    metrics["steps"] = steps
    metrics["sim_time_s"] = sim_time
    metrics["K"] = k
    metrics["stop_rule"] = stop
    metrics["peak_wet_count"] = peak
    return metrics


def main() -> int:
    exe = Path(sys.argv[1] if len(sys.argv) > 1 else "build/p4b_cpu/floodca_omp")
    threads = int(sys.argv[2] if len(sys.argv) > 2 else "48")
    root = Path("results/e9_aniso")
    grid = root / "grid"
    _write_grid(grid)
    rows = [run_k(exe, grid, root, k, threads) for k in (4, 8)]
    out = {"setup": {"n": N, "z_m": 10.0, "block": "5x5", "h0_m": 1.0, "thresh_m": THRESH,
                     "dx_m": 90.0, "boundary": "closed", "rain": "none",
                     "stop_rule": "wet cell count sets no new maximum for 3 chunks of 400 steps",
                     "ratio_reported": "r_diag/r_axis (1.0 isotropic); ratio_axis_over_diag is SPEC 10.4",
                     "note": "Initial column instead of the SPEC 10.4 point source. Physics unchanged. "
                             "A 40 percent wet target is unreachable at this volume."},
           "runs": rows}
    Path("results/e9_anisotropy.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
