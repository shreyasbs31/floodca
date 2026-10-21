#!/usr/bin/env python3
"""Download IMD yearly rain files. Login-node HTTPS fails (P-0); CPU compute
nodes can reach imdlib (job 242680, 2005.grd). Run this via Slurm, not login.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/imd")
    ap.add_argument("--start", type=int, default=2005)
    ap.add_argument("--end", type=int, default=2005)
    args = ap.parse_args()
    try:
        import imdlib as imd
    except ImportError:
        sys.stderr.write("imdlib is not installed in this environment\n")
        return 1
    import calendar
    import time

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for year in range(args.start, args.end + 1):
        grd = out / "rain" / f"{year}.grd"
        expect = (366 if calendar.isleap(year) else 365) * 129 * 135 * 4
        for attempt in range(1, 4):
            if grd.is_file() and grd.stat().st_size == expect:
                break
            if grd.is_file():
                grd.unlink()
            print(f"year={year} attempt={attempt}", flush=True)
            try:
                imd.get_data("rain", year, year, fn_format="yearwise", file_dir=str(out))
            except Exception as exc:  # imdlib raises bare Exceptions on server errors
                print(f"year={year} error={exc!r}", flush=True)
                time.sleep(10)
        ok = grd.is_file() and grd.stat().st_size == expect
        print(f"year={year} ok={ok} bytes={grd.stat().st_size if grd.is_file() else 0}", flush=True)
        if not ok:
            failed.append(year)
    print(f"done failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
