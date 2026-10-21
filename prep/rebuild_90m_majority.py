"""Rewrite 90/mask.bin and 90/lc.bin from the 30 m products with the fixed 3x3
majority aggregation. Equivalent to rerunning floodca_prep.dem and
floodca_prep.landcover (which compute mask90 and lc90 from mask30 and lc30), but
leaves the 30 m, S, M, and L files untouched."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floodca_prep import dem as dem_mod
from floodca_prep import landcover as lc_mod
from floodca_prep.io_bin import read_bin, write_bin


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    args = ap.parse_args()
    proc = Path(args.proc)
    out = {}
    for name, fn in (("mask", dem_mod._agg_majority_3), ("lc", lc_mod._agg_majority_3)):
        a30, _ = read_bin(proc / "30" / f"{name}.bin")
        old, hdr90 = read_bin(proc / "90" / f"{name}.bin")
        new = fn(a30).astype(old.dtype)
        assert new.shape == old.shape
        write_bin(proc / "90" / f"{name}.bin", new, hdr90)
        out[name] = {"changed_cells": int((new != old).sum()), "cells": int(new.size)}
        if name == "mask":
            out[name].update({k: int((new == v).sum()) for k, v in (("land", 0), ("sea", 1), ("wall", 2))})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
