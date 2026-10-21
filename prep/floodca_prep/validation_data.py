"""GFD validation rasters on the model grid (SPEC Sec 4.4, 10.1).

Primary permanent-water mask is the GFD image band jrc_perm_water, not the
standalone JRC GSW max_extent GeoTIFF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from floodca_prep.io_bin import read_bin, write_bin
from floodca_prep.validation_metrics import land_flood_fraction


def _header(name, arr, dx, x0, y0, dtype, source, extra=None):
    h = {
        "name": name,
        "nrows": int(arr.shape[0]),
        "ncols": int(arr.shape[1]),
        "dx": float(dx),
        "dtype": dtype,
        "crs": "EPSG:32643",
        "x0": float(x0),
        "y0": float(y0),
        "nodata": -9999,
        "source": source,
    }
    if extra:
        h.update(extra)
    return h


def _reproject_band(src, band_index, dst, dst_transform, resampling):
    reproject(
        source=rasterio.band(src, band_index),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:32643",
        resampling=resampling,
    )


def process(gfd_tif: Path, proc_root: Path, event_id: int = 2734) -> dict:
    mask90, mhdr = read_bin(proc_root / "90" / "mask.bin")
    x0, y0 = float(mhdr["x0"]), float(mhdr["y0"])
    nrows90, ncols90 = mask90.shape
    transform90 = from_origin(x0, y0, 90.0, 90.0)

    minx = x0
    maxy = y0
    miny = y0 - nrows90 * 90.0
    maxx = x0 + ncols90 * 90.0
    dx250 = 250.0
    ncols250 = int(np.ceil((maxx - minx) / dx250))
    nrows250 = int(np.ceil((maxy - miny) / dx250))
    transform250 = from_origin(minx, maxy, dx250, dx250)

    flooded90 = np.zeros((nrows90, ncols90), dtype=np.uint8)
    perm90 = np.zeros((nrows90, ncols90), dtype=np.uint8)
    flooded250 = np.zeros((nrows250, ncols250), dtype=np.uint8)
    perm250 = np.zeros((nrows250, ncols250), dtype=np.uint8)
    descriptions: tuple = ()
    with rasterio.open(gfd_tif) as src:
        if src.count < 2:
            raise RuntimeError(
                f"{gfd_tif} has {src.count} band(s); need flooded and jrc_perm_water"
            )
        descriptions = src.descriptions
        names = [d or f"band{i + 1}" for i, d in enumerate(src.descriptions)]
        i_flood = names.index("flooded") + 1 if "flooded" in names else 1
        i_perm = names.index("jrc_perm_water") + 1 if "jrc_perm_water" in names else 2
        if i_perm == i_flood:
            i_perm = 2
        _reproject_band(src, i_flood, flooded90, transform90, Resampling.nearest)
        _reproject_band(src, i_perm, perm90, transform90, Resampling.nearest)
        _reproject_band(src, i_flood, flooded250, transform250, Resampling.nearest)
        _reproject_band(src, i_perm, perm250, transform250, Resampling.nearest)

    extra = {
        "aoi": mhdr.get("aoi", "udupi_coast"),
        "gfd_id": event_id,
        "gfd_bands": list(descriptions),
        "perm_source": "GFD jrc_perm_water band (not JRC GSW max_extent)",
    }
    src = f"GFD MODIS event {event_id}"
    out = proc_root / "validation"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"gfd_{event_id}"
    write_bin(
        out / f"{tag}_flooded_90.bin",
        flooded90,
        _header(f"{tag}_flooded_90", flooded90, 90.0, x0, y0, "uint8", src, extra),
    )
    write_bin(
        out / f"{tag}_jrc_perm_water_90.bin",
        perm90,
        _header(f"{tag}_jrc_perm_90", perm90, 90.0, x0, y0, "uint8", src, extra),
    )
    write_bin(
        out / f"{tag}_flooded_250.bin",
        flooded250,
        _header(f"{tag}_flooded_250", flooded250, 250.0, minx, maxy, "uint8", src, extra),
    )
    write_bin(
        out / f"{tag}_jrc_perm_water_250.bin",
        perm250,
        _header(f"{tag}_jrc_perm_250", perm250, 250.0, minx, maxy, "uint8", src, extra),
    )

    land = mask90 == 0
    report = {
        "event_id": event_id,
        "gfd_tif": str(gfd_tif),
        "perm_source": extra["perm_source"],
        "clear_perc_band": None,
        "clear_perc_note": "clear_perc was not in the two-band GEE export; SPEC 10.1 skip until present",
        "nrows90": int(nrows90),
        "ncols90": int(ncols90),
        "nrows250": int(nrows250),
        "ncols250": int(ncols250),
        "flooded_frac_90_all_cells": float((flooded90 == 1).mean()),
        "land_flood_frac_90": land_flood_fraction(flooded90, perm90, land),
        "perm_frac_90": float((perm90 == 1).mean()),
        "flooded_frac_250_all_cells": float((flooded250 == 1).mean()),
        "max_extent_not_used": True,
    }
    report_name = "validation_report.json" if event_id == 2734 else f"validation_report_{event_id}.json"
    (out / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _find_gfd(path: Path, event_id: int) -> Path:
    if path.is_file():
        return path
    gee = path.parent
    hits = sorted(gee.glob(f"*gfd*{event_id}*.tif")) + sorted(gee.glob(f"*GFD*{event_id}*.tif"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"GFD GeoTIFF not found at {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gfd", default="data/raw/gee/floodca_gfd_udupi_2734.tif")
    ap.add_argument("--out", default="data/processed/udupi_coast")
    ap.add_argument("--event-id", type=int, default=2734)
    args = ap.parse_args()
    gfd = _find_gfd(Path(args.gfd), args.event_id)
    report = process(gfd, Path(args.out), args.event_id)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
