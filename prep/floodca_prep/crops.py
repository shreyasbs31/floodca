"""Extra 30 m crops for the E7 crossover sizes (SPEC Sec 3.3, 11.3).

E7 uses sizes 128, 256, 512, 768, 1024, 1536, 2048, 3072 and full30. 512, 1024 and
2048 are grids S, M and L; this writes the others as C<size> with the same
centre-of-land rule as floodca_prep.dem, slicing dem, mask, n, C, lc and the B0
rain field from the 30 m products so every crop sees identical inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from floodca_prep.dem import _crop_land_centre
from floodca_prep.io_bin import read_bin, write_bin

FIELDS = ["dem", "mask", "n", "C", "lc"]


def write_crop(proc: Path, size: int, b0_date: str) -> dict:
    dem30, h30 = read_bin(proc / "30" / "dem.bin")
    mask30, _ = read_bin(proc / "30" / "mask.bin")
    _, _, rs, cs, shift, sea_frac = _crop_land_centre(dem30, mask30, size)
    name = f"C{size}"
    dx = float(h30["dx"])
    x0 = float(h30["x0"]) + cs * dx
    y0 = float(h30["y0"]) - rs * dx
    sources = {f: proc / "30" / f"{f}.bin" for f in FIELDS}
    sources["rain"] = proc / "rain" / "30" / f"rain_{b0_date}.bin"
    for field, src in sources.items():
        arr, hdr = read_bin(src)
        sub = arr[rs : rs + size, cs : cs + size]
        out = dict(hdr)
        out.update({"name": f"{field}_{name}", "nrows": size, "ncols": size, "x0": x0, "y0": y0,
                    "crop_of": str(src), "row0": rs, "col0": cs})
        out.pop("sha256", None)
        out.pop("created", None)
        dst = (proc / "rain" / name / f"rain_{b0_date}.bin") if field == "rain" \
            else (proc / name / f"{field}.bin")
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_bin(dst, sub, out)
    return {"row0": rs, "col0": cs, "shift_east_px": shift, "sea_frac": sea_frac,
            "nrows": size, "ncols": size, "land_cells": int((mask30[rs:rs + size, cs:cs + size] == 0).sum())}


def write_centre_crop_of(proc: Path, src_grid: str, size: int, b0_date: str) -> dict:
    """Centre window of an existing benchmark grid, including its B0 warm start.

    Used for the extra E7 sizes 32 and 64, which are crops of S rather than a new
    land-centre crop of the 30 m mosaic.
    """
    dem, h = read_bin(proc / src_grid / "dem.bin")
    nrows, ncols = int(dem.shape[0]), int(dem.shape[1])
    if size > nrows or size > ncols:
        raise RuntimeError(f"{src_grid} is {nrows}x{ncols}, smaller than {size}")
    rs = (nrows - size) // 2
    cs = (ncols - size) // 2
    name = f"C{size}"
    dx = float(h["dx"])
    x0 = float(h["x0"]) + cs * dx
    y0 = float(h["y0"]) - rs * dx
    sources = {f: proc / src_grid / f"{f}.bin" for f in FIELDS}
    sources["rain"] = proc / "rain" / src_grid / f"rain_{b0_date}.bin"
    sources["h0"] = proc / "warmstart" / f"h0_B0_{src_grid}.bin"
    for field, src in sources.items():
        arr, hdr = read_bin(src)
        sub = arr[rs:rs + size, cs:cs + size]
        out = dict(hdr)
        out.update({
            "name": f"{field}_{name}", "nrows": size, "ncols": size, "x0": x0, "y0": y0,
            "crop_of": str(src), "row0": rs, "col0": cs, "src_grid": src_grid,
        })
        out.pop("sha256", None)
        out.pop("created", None)
        if field == "rain":
            dst = proc / "rain" / name / f"rain_{b0_date}.bin"
        elif field == "h0":
            dst = proc / "warmstart" / f"h0_B0_{name}.bin"
        else:
            dst = proc / name / f"{field}.bin"
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_bin(dst, sub, out)
    mask, _ = read_bin(proc / name / "mask.bin")
    return {
        "src_grid": src_grid, "row0": rs, "col0": cs, "nrows": size, "ncols": size,
        "land_cells": int((mask == 0).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--sizes", default="128,256,768,1536")
    ap.add_argument("--from-grid", default="", help="if set, centre-crop this grid instead of the 30 m mosaic")
    args = ap.parse_args()
    proc = Path(args.proc)
    b0 = json.loads((proc / "scenarios_N500_report.json").read_text(encoding="utf-8"))["b0"]
    report_path = proc / "crops_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    for s in (int(x) for x in args.sizes.split(",")):
        if args.from_grid:
            report[f"C{s}"] = write_centre_crop_of(proc, args.from_grid, s, b0["base_date"])
        else:
            report[f"C{s}"] = write_crop(proc, s, b0["base_date"])
        print(f"C{s}", json.dumps(report[f"C{s}"]), flush=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
