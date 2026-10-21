"""Copernicus DEM mosaic, UTM 30/90 m grids, SEA/WALL mask, benchmark crops (SPEC 4.2)."""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import yaml
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

from floodca_prep.io_bin import write_bin

LAND, SEA, WALL = 0, 1, 2
NODATA = -9999.0


def _load_aoi(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _snap_down(v: float, step: float) -> float:
    return math.floor(v / step) * step


def _snap_up(v: float, step: float) -> float:
    return math.ceil(v / step) * step


def _utm_aoi_grid(aoi: dict, dx: float = 30.0, snap_m: float = 90.0):
    transformer = Transformer.from_crs("EPSG:4326", aoi["crs"], always_xy=True)
    corners = [
        (aoi["lon_min"], aoi["lat_min"]),
        (aoi["lon_min"], aoi["lat_max"]),
        (aoi["lon_max"], aoi["lat_min"]),
        (aoi["lon_max"], aoi["lat_max"]),
    ]
    xs, ys = zip(*[transformer.transform(x, y) for x, y in corners])
    minx = _snap_down(min(xs), snap_m)
    miny = _snap_down(min(ys), snap_m)
    maxx = _snap_up(max(xs), snap_m)
    maxy = _snap_up(max(ys), snap_m)
    ncols = int(round((maxx - minx) / dx))
    nrows = int(round((maxy - miny) / dx))
    if ncols % 3 or nrows % 3:
        raise RuntimeError(f"grid not divisible by 3: nrows={nrows} ncols={ncols}")
    transform = from_origin(minx, maxy, dx, dx)
    return {
        "minx": minx,
        "miny": miny,
        "maxx": maxx,
        "maxy": maxy,
        "nrows": nrows,
        "ncols": ncols,
        "transform": transform,
        "x0": minx,
        "y0": maxy,
    }


def _mosaic(raw_dir: Path):
    tifs = sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.tiff"))
    if not tifs:
        raise FileNotFoundError(f"no GeoTIFFs in {raw_dir}")
    srcs = [rasterio.open(p) for p in tifs]
    try:
        mosaic, transform = merge(srcs)
        crs = srcs[0].crs
        nodata = srcs[0].nodata
    finally:
        for s in srcs:
            s.close()
    return mosaic[0], transform, crs, nodata


def _crop_wgs84(arr, transform, crs, aoi: dict, buffer_deg: float):
    west = aoi["lon_min"] - buffer_deg
    south = aoi["lat_min"] - buffer_deg
    east = aoi["lon_max"] + buffer_deg
    north = aoi["lat_max"] + buffer_deg
    window = rasterio.windows.from_bounds(west, south, east, north, transform=transform)
    window = window.round_offsets().round_lengths()
    row_off = max(int(window.row_off), 0)
    col_off = max(int(window.col_off), 0)
    row_end = min(int(window.row_off + window.height), arr.shape[0])
    col_end = min(int(window.col_off + window.width), arr.shape[1])
    cropped = arr[row_off:row_end, col_off:col_end]
    new_transform = rasterio.windows.transform(
        rasterio.windows.Window(col_off, row_off, col_end - col_off, row_end - row_off),
        transform,
    )
    return cropped, new_transform, crs


def _reproject_to_grid(src, src_transform, src_crs, grid, src_nodata):
    dst = np.full((grid["nrows"], grid["ncols"]), NODATA, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(src.astype(np.float32)),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"] if "crs" in grid else "EPSG:32643",
        src_nodata=src_nodata,
        dst_nodata=NODATA,
        resampling=Resampling.bilinear,
    )
    return dst


def _fill_nodata(z: np.ndarray) -> tuple[np.ndarray, int]:
    invalid = ~np.isfinite(z) | (z == NODATA)
    nfill = int(invalid.sum())
    if nfill == 0 or invalid.all():
        return z.copy(), nfill
    filled = z.copy()
    _, indices = distance_transform_edt(invalid, return_distances=True, return_indices=True)
    filled[invalid] = z[tuple(indices[:, invalid])]
    return filled, nfill


def _sea_from_west(z: np.ndarray, thresh: float) -> np.ndarray:
    nrows, ncols = z.shape
    sea = np.zeros((nrows, ncols), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(nrows):
        if np.isfinite(z[r, 0]) and z[r, 0] <= thresh:
            sea[r, 0] = True
            q.append((r, 0))
    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols and not sea[rr, cc]:
                if np.isfinite(z[rr, cc]) and z[rr, cc] <= thresh:
                    sea[rr, cc] = True
                    q.append((rr, cc))
    return sea


def _mask30(z: np.ndarray, sea: np.ndarray, raw_invalid: np.ndarray) -> np.ndarray:
    mask = np.full(z.shape, LAND, dtype=np.uint8)
    mask[sea] = SEA
    # Remaining original nodata that NN-fill could not make land/sea: WALL.
    # After NN fill every cell has a value; WALL is unused unless fill failed.
    still_bad = ~np.isfinite(z) | (z == NODATA)
    mask[still_bad] = WALL
    # Keep SEA over WALL if a filled cell is connected ocean.
    mask[sea] = SEA
    _ = raw_invalid
    return mask


def _agg_mean_3(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h // 3, 3, w // 3, 3).mean(axis=(1, 3)).astype(np.float32)


def _agg_majority_3(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    x = a.reshape(h // 3, 3, w // 3, 3).transpose(0, 2, 1, 3).reshape(h // 3, w // 3, 9)
    counts = np.stack([(x == k).sum(axis=2) for k in (0, 1, 2)], axis=-1)
    return counts.argmax(axis=-1).astype(np.uint8)


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


def _write_grid(out_dir: Path, name: str, arr: np.ndarray, header: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_bin(out_dir / f"{name}.bin", arr, header)


def _crop_land_centre(dem, mask, size, max_sea=0.05):
    land = mask == LAND
    if not land.any():
        raise RuntimeError("no LAND cells for crop")
    rows, cols = np.where(land)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    nrows, ncols = mask.shape
    shift_east = 0
    while True:
        rs = cr - size // 2
        re = rs + size
        cs = cc - size // 2 + shift_east
        ce = cs + size
        if rs < 0:
            rs, re = 0, size
        if re > nrows:
            re, rs = nrows, nrows - size
        if cs < 0:
            cs, ce = 0, size
        if ce > ncols:
            ce, cs = ncols, ncols - size
        if rs < 0 or cs < 0:
            raise RuntimeError(f"grid smaller than crop {size}")
        sub_m = mask[rs:re, cs:ce]
        sub_z = dem[rs:re, cs:ce]
        sea_frac = float((sub_m == SEA).mean())
        if sea_frac < max_sea:
            return sub_z, sub_m, rs, cs, shift_east, sea_frac
        shift_east += 32
        if cs + shift_east + size > ncols:
            return sub_z, sub_m, rs, cs, shift_east, sea_frac


def process(aoi_path: Path, raw_dir: Path, out_root: Path) -> dict:
    aoi = _load_aoi(aoi_path)
    grid = _utm_aoi_grid(aoi)
    grid["crs"] = aoi["crs"]
    mosaic, m_transform, m_crs, m_nodata = _mosaic(raw_dir)
    cropped, c_transform, c_crs = _crop_wgs84(
        mosaic, m_transform, m_crs, aoi, float(aoi.get("buffer_deg", 0.02))
    )
    dem30_raw = _reproject_to_grid(cropped, c_transform, c_crs, grid, m_nodata)
    invalid = ~np.isfinite(dem30_raw) | (dem30_raw == NODATA)
    dem30, nfill = _fill_nodata(dem30_raw)
    sea = _sea_from_west(dem30, float(aoi.get("sea_elev_max", 0.5)))
    mask30 = _mask30(dem30, sea, invalid)
    dem90 = _agg_mean_3(dem30)
    mask90 = _agg_majority_3(mask30)

    source = "Copernicus GLO-30"
    extra = {"aoi": aoi.get("name", "aoi")}
    out30 = out_root / "30"
    out90 = out_root / "90"
    _write_grid(
        out30,
        "dem",
        dem30.astype(np.float32),
        _header("dem_30", dem30, 30.0, grid["x0"], grid["y0"], "float32", source, extra),
    )
    _write_grid(
        out30,
        "mask",
        mask30,
        _header("mask_30", mask30, 30.0, grid["x0"], grid["y0"], "uint8", source, extra),
    )
    x0_90, y0_90 = grid["x0"], grid["y0"]
    _write_grid(
        out90,
        "dem",
        dem90,
        _header("dem_90", dem90, 90.0, x0_90, y0_90, "float32", source, extra),
    )
    _write_grid(
        out90,
        "mask",
        mask90,
        _header("mask_90", mask90, 90.0, x0_90, y0_90, "uint8", source, extra),
    )

    crops = {}
    for name, size in (("S", 512), ("M", 1024), ("L", 2048)):
        zc, mc, rs, cs, shift, sea_frac = _crop_land_centre(dem30, mask30, size)
        cdir = out_root / name
        dx = 30.0
        x0 = grid["x0"] + cs * dx
        y0 = grid["y0"] - rs * dx
        _write_grid(
            cdir,
            "dem",
            zc.astype(np.float32),
            _header(f"dem_{name}", zc, dx, x0, y0, "float32", source, extra),
        )
        _write_grid(
            cdir,
            "mask",
            mc,
            _header(f"mask_{name}", mc, dx, x0, y0, "uint8", source, extra),
        )
        crops[name] = {
            "row0": rs,
            "col0": cs,
            "shift_east_px": shift,
            "sea_frac": sea_frac,
            "nrows": int(zc.shape[0]),
            "ncols": int(zc.shape[1]),
        }

    n = mask30.size
    report = {
        "aoi": aoi.get("name"),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "crs": aoi["crs"],
        "nrows30": int(dem30.shape[0]),
        "ncols30": int(dem30.shape[1]),
        "nrows90": int(dem90.shape[0]),
        "ncols90": int(dem90.shape[1]),
        "x0": grid["x0"],
        "y0": grid["y0"],
        "minx": grid["minx"],
        "maxx": grid["maxx"],
        "miny": grid["miny"],
        "maxy": grid["maxy"],
        "filled_cells": nfill,
        "filled_frac": nfill / float(n),
        "sea_cells": int((mask30 == SEA).sum()),
        "sea_frac": float((mask30 == SEA).mean()),
        "wall_cells": int((mask30 == WALL).sum()),
        "land_cells": int((mask30 == LAND).sum()),
        "z_min_land": float(dem30[mask30 == LAND].min()) if (mask30 == LAND).any() else None,
        "z_max_land": float(dem30[mask30 == LAND].max()) if (mask30 == LAND).any() else None,
        "pit_fill": False,
        "crops": crops,
        "note": "No pit filling. SEA is 4-connected to the west edge at elev <= 0.5 m.",
    }
    (out_root / "dem_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aoi", default="config/aoi_udupi.yaml")
    ap.add_argument("--raw", default="data/raw/dem")
    ap.add_argument("--out", default="data/processed/udupi_coast")
    args = ap.parse_args()
    report = process(Path(args.aoi), Path(args.raw), Path(args.out))
    print(json.dumps({k: report[k] for k in ("nrows30", "ncols30", "filled_cells", "sea_frac", "crops")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
