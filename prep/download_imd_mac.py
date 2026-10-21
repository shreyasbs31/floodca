#!/usr/bin/env python3
"""Download IMD 0.25 deg daily rainfall on a laptop (SPEC Sec 4.3).

The PARAM Utkarsh login node cannot reach IMD HTTPS (P-0 VERIFY). Run this
on your Mac, then rsync the yearly files to scratch.

Setup (once, on the Mac):
    python3 -m venv ~/.venvs/floodca
    source ~/.venvs/floodca/bin/activate
    pip install imdlib tqdm

Run:
    python3 prep/download_imd_mac.py --out data/raw/imd

Copy to the cluster (from the laptop):
    rsync -avP data/raw/imd/ \\
        "$FCA_LOGIN:$FCA_DATA/raw/imd/"

Expected objects: yearly IMD rain files for 1991 to 2023 under that folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/imd", help="local destination directory")
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=2023)
    args = ap.parse_args()
    try:
        import imdlib as imd
    except ImportError:
        sys.stderr.write("imdlib is not installed. pip install imdlib\n")
        return 1
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(f"imd.get_data rain {args.start}-{args.end} -> {out}")
    imd.get_data("rain", args.start, args.end, fn_format="yearwise", file_dir=str(out))
    n = len(list(out.rglob("*")))
    print(f"done. files_under_out={n} path={out}")
    print("Next: rsync this folder to $FCA_DATA/raw/imd/ on the cluster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
