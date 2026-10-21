"""Weak-scaling grids W<P> (SPEC Sec 3.3): grid M mirror-tiled into a dims[0] x dims[1]
arrangement, with dims from MPI_Dims_create(P, 2). Tile (i, j) is M flipped up-down if
i is odd and left-right if j is odd, so terrain stays continuous across seams. dem,
mask, n, C, lc, the B0 rain field and the B0 warm start are tiled the same way.
These grids are synthetic and labelled as such.

Writes <proc>/W<P>/{dem,mask,n,C,lc}.bin, <proc>/rain/W<P>/rain_<B0>.bin,
<proc>/warmstart/h0_B0_W<P>.bin and the symlink dir <proc>/warmstart/grid_W<P>.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from floodca_prep.io_bin import read_bin, write_bin

FIELDS = ["dem", "mask", "n", "C", "lc"]


def dims_create(p: int) -> tuple[int, int]:
    """MPI_Dims_create(p, 2): the most balanced factorisation, larger factor first."""
    best = (p, 1)
    for a in range(1, int(p ** 0.5) + 1):
        if p % a == 0:
            best = (p // a, a)
    return best


def mirror_tile(a: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
    rows = []
    for i in range(dims[0]):
        r = a[::-1, :] if i % 2 else a
        rows.append(np.concatenate([r[:, ::-1] if j % 2 else r for j in range(dims[1])], axis=1))
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def write_w(proc: Path, p: int, b0_date: str) -> dict:
    dims = dims_create(p)
    name = f"W{p}"
    sources = {f: proc / "M" / f"{f}.bin" for f in FIELDS}
    sources["rain"] = proc / "rain" / "M" / f"rain_{b0_date}.bin"
    sources["h0"] = proc / "warmstart" / "h0_B0_M.bin"
    dests = {f: proc / name / f"{f}.bin" for f in FIELDS}
    dests["rain"] = proc / "rain" / name / f"rain_{b0_date}.bin"
    dests["h0"] = proc / "warmstart" / f"h0_B0_{name}.bin"
    land = 0
    for field, src in sources.items():
        arr, hdr = read_bin(src)
        big = mirror_tile(arr, dims)
        out = dict(hdr)
        out.update({"name": f"{field}_{name}", "nrows": int(big.shape[0]), "ncols": int(big.shape[1]),
                    "synthetic": f"grid M mirror-tiled {dims[0]}x{dims[1]} (SPEC 3.3 W<P>)",
                    "tiled_from": str(src)})
        for k in ("sha256", "created", "stats"):
            out.pop(k, None)
        dests[field].parent.mkdir(parents=True, exist_ok=True)
        write_bin(dests[field], big, out)
        if field == "mask":
            land = int((big == 0).sum())
    ws = proc / "warmstart" / f"grid_{name}"
    ws.mkdir(parents=True, exist_ok=True)
    for f in FIELDS[:4]:
        for ext in ("bin", "json"):
            link = ws / f"{f}.{ext}"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.realpath(proc / name / f"{f}.{ext}"), link)
    return {"P": p, "dims": list(dims), "nrows": dims[0] * 1024, "ncols": dims[1] * 1024, "land_cells": land}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--ps", default="1,2,4,8,16")
    args = ap.parse_args()
    proc = Path(args.proc)
    b0 = json.loads((proc / "scenarios_N500_report.json").read_text(encoding="utf-8"))["b0"]
    report = {}
    for p in (int(x) for x in args.ps.split(",")):
        report[f"W{p}"] = write_w(proc, p, b0["base_date"])
        print(json.dumps(report[f"W{p}"]), flush=True)
    (proc / "tiles_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
