"""E3 calibration grids (SPEC Sec 10.3): 90 m grid directories that differ only in
the runoff coefficient, C = clip(C_table * k_C, 0, 1) at 30 m, then the 3x3 mean,
exactly as floodca_prep.landcover builds 90/C.bin. dem, mask and n are symlinks to
the production 90 m files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from floodca_prep.io_bin import read_bin, write_bin
from floodca_prep.landcover import _agg_mean_3, _lookup_n_c, load_lc_table


def c90_for(proc: Path, params_csv: Path, k_c: float) -> np.ndarray:
    lc30, _ = read_bin(proc / "30" / "lc.bin")
    _, c30 = _lookup_n_c(lc30, load_lc_table(params_csv), 1.0, k_c)
    return np.clip(_agg_mean_3(c30), 0.0, 1.0)


def build(proc: Path, params_csv: Path, k_c: float) -> Path:
    d = proc / "e3" / f"grid_90_kC{k_c:g}"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("dem", "mask", "n"):
        for ext in (".bin", ".json"):
            dst = d / f"{name}{ext}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink((proc / "90" / f"{name}{ext}").resolve(), dst)
    c90 = c90_for(proc, params_csv, k_c)
    _, hdr = read_bin(proc / "90" / "C.bin")
    hdr = {k: v for k, v in hdr.items() if k not in ("sha256", "created")}
    hdr.update({"name": f"C_90_kC{k_c:g}", "k_C": k_c})
    write_bin(d / "C.bin", c90.astype(np.float32), hdr)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--params", default="config/landcover_params.csv")
    ap.add_argument("--k-c", default="0.7,1.3")
    args = ap.parse_args()
    proc, params = Path(args.proc), Path(args.params)
    ref, _ = read_bin(proc / "90" / "C.bin")
    if not np.array_equal(c90_for(proc, params, 1.0).astype(np.float32), ref):
        raise SystemExit("k_C = 1.0 does not reproduce 90/C.bin; landcover inputs changed")
    out = {}
    for k in (float(x) for x in args.k_c.split(",")):
        d = build(proc, params, k)
        c, _ = read_bin(d / "C.bin")
        out[f"{k:g}"] = {"dir": str(d), "C_mean": float(c.mean()), "C_max": float(c.max())}
    print(json.dumps({"k_C_1_reproduces_C90": True, "grids": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
