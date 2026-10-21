"""Georeferencing check: GFD event 2734 against the 90 m Udupi grid.

Runs on a compute node. Writes results/alignment_report.json and
results/figures/validation_alignment.png. Does not change any grid or score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from floodca_prep.dem import _agg_majority_3
from floodca_prep.io_bin import read_bin


def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(a)
    rs = slice(max(dr, 0), a.shape[0] + min(dr, 0))
    rd = slice(max(-dr, 0), a.shape[0] + min(-dr, 0))
    cs = slice(max(dc, 0), a.shape[1] + min(dc, 0))
    cd = slice(max(-dc, 0), a.shape[1] + min(-dc, 0))
    out[rd, cd] = a[rs, cs]
    return out


def _overlap(perm: np.ndarray, sea: np.ndarray) -> dict:
    p = perm == 1
    both = int((p & sea).sum())
    nperm = int(p.sum())
    nsea = int(sea.sum())
    union = int((p | sea).sum())
    return {
        "perm_cells": nperm,
        "perm_on_sea": both,
        "frac_perm_on_sea": both / nperm if nperm else None,
        "frac_sea_that_is_perm": both / nsea if nsea else None,
        "jaccard_perm_sea": both / union if union else None,
    }


def _reproject(src, band: int, shape, transform) -> np.ndarray:
    dst = np.zeros(shape, dtype=np.uint8)
    reproject(
        source=rasterio.band(src, band),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs="EPSG:32643",
        resampling=Resampling.nearest,
    )
    return dst


def _elev(z: np.ndarray, sel: np.ndarray) -> dict:
    v = z[sel].astype(np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "p10": float(np.percentile(v, 10)),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "max": float(v.max()),
        "frac_z_eq_0": float(np.mean(v == 0.0)),
        "frac_z_lt_5": float(np.mean(v < 5.0)),
        "frac_z_lt_10": float(np.mean(v < 10.0)),
    }


def _counts(a: np.ndarray) -> dict:
    vals, cnts = np.unique(a, return_counts=True)
    if len(vals) > 12:
        return {"n_unique": int(len(vals)), "min": float(vals[0]), "max": float(vals[-1])}
    return {str(int(v) if float(v) == int(v) else v): int(c) for v, c in zip(vals, cnts)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", type=int, default=2734)
    args = ap.parse_args(argv)
    event = args.event
    proc = Path("data/processed/udupi_coast")
    dem, dhdr = read_bin(proc / "90" / "dem.bin")
    mask, mhdr = read_bin(proc / "90" / "mask.bin")
    perm_path = proc / "validation" / f"gfd_{event}_jrc_perm_water_90.bin"
    flood_path = proc / "validation" / f"gfd_{event}_flooded_90.bin"
    perm90 = read_bin(perm_path)[0] if perm_path.is_file() else None
    flood90 = read_bin(flood_path)[0] if flood_path.is_file() else None
    hmax_path = Path("results/v0_smoke/hmax.bin") if event == 2734 else Path(f"results/val_{event}/hmax.bin")
    hmax, hhdr = read_bin(hmax_path) if hmax_path.is_file() else (None, {})
    mask30, _ = read_bin(proc / "30" / "mask.bin")
    rebuilt = _agg_majority_3(mask30).astype(mask.dtype)
    sea = mask == 1
    land = mask == 0
    x0, y0, dx = float(dhdr["x0"]), float(dhdr["y0"]), float(dhdr["dx"])
    transform = from_origin(x0, y0, dx, dx)

    gfd_path = Path(f"data/raw/gee/floodca_gfd_udupi_{event}.tif")
    with rasterio.open(gfd_path) as src:
        names = [d or f"band{i + 1}" for i, d in enumerate(src.descriptions)]
        i_flood = names.index("flooded") + 1 if "flooded" in names else 1
        i_perm = names.index("jrc_perm_water") + 1 if "jrc_perm_water" in names else 2
        if i_perm == i_flood:
            i_perm = 2 if i_flood == 1 else 1
        band_counts = {names[i - 1]: _counts(src.read(i)) for i in range(1, src.count + 1)}
        fresh_perm = _reproject(src, i_perm, dem.shape, transform)
        fresh_flood = _reproject(src, i_flood, dem.shape, transform)
        utm_bounds = list(transform_bounds(src.crs, "EPSG:32643", *src.bounds, densify_pts=21))
        model_bounds = [x0, y0 - dem.shape[0] * dx, x0 + dem.shape[1] * dx, y0]
        covers = (
            utm_bounds[0] <= model_bounds[0] + 1.0
            and utm_bounds[1] <= model_bounds[1] + 1.0
            and utm_bounds[2] >= model_bounds[2] - 1.0
            and utm_bounds[3] >= model_bounds[3] - 1.0
        )
        gfd = {
            "event_id": event,
            "path": str(gfd_path),
            "crs": str(src.crs),
            "bounds": list(src.bounds),
            "bounds_epsg32643": utm_bounds,
            "covers_model_grid": bool(covers),
            "north_up": bool(src.transform.e < 0 and src.transform.b == 0 and src.transform.d == 0),
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "res": list(src.res),
            "dtypes": list(src.dtypes),
            "bands": names,
            "band_index_flooded": i_flood,
            "band_index_perm": i_perm,
            "raw_value_counts": band_counts,
        }
        half = {}
        for name, ox, oy in (
            ("zero", 0.0, 0.0),
            ("east_45m", 45.0, 0.0),
            ("west_45m", -45.0, 0.0),
            ("north_45m", 0.0, 45.0),
            ("south_45m", 0.0, -45.0),
            ("east_90m", 90.0, 0.0),
            ("north_90m", 0.0, 90.0),
        ):
            shifted = _reproject(src, i_perm, dem.shape, from_origin(x0 + ox, y0 + oy, dx, dx))
            half[name] = _overlap(shifted, sea)

    shifts = {}
    best = None
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            ov = _overlap(_shift(fresh_perm, dr, dc), sea)
            shifts[f"row{dr:+d}_col{dc:+d}"] = ov
            score = ov["jaccard_perm_sea"] or 0.0
            if best is None or score > best[0]:
                best = (score, dr, dc)

    report = {
        "model_grid": {
            "crs": dhdr.get("crs"),
            "x0": x0,
            "y0": y0,
            "dx": dx,
            "nrows": int(dem.shape[0]),
            "ncols": int(dem.shape[1]),
            "bounds": [x0, y0 - dem.shape[0] * dx, x0 + dem.shape[1] * dx, y0],
            "row0_is_north": True,
            "dem_header_matches_mask_header": (
                dhdr.get("x0") == mhdr.get("x0") and dhdr.get("y0") == mhdr.get("y0")
                and dhdr.get("nrows") == mhdr.get("nrows") and dhdr.get("ncols") == mhdr.get("ncols")
            ),
        },
        "gfd_geotiff": gfd,
        "stored_perm90_equals_fresh_reproject": (
            None if perm90 is None else bool(np.array_equal(perm90, fresh_perm))
        ),
        "stored_flood90_equals_fresh_reproject": (
            None if flood90 is None else bool(np.array_equal(flood90, fresh_flood))
        ),
        "mask90_equals_fixed_3x3_majority_of_30m": bool(np.array_equal(mask, rebuilt)),
        "shapes": {
            "dem": list(dem.shape),
            "perm90": None if perm90 is None else list(perm90.shape),
            "flood90": None if flood90 is None else list(flood90.shape),
            "hmax": None if hmax is None else list(hmax.shape),
            "transpose_would_match": list(dem.shape) == list(dem.shape[::-1]),
        },
        "hmax_header": {"path": str(hmax_path), "dtype": hhdr.get("dtype"), "created": hhdr.get("created"),
                        "nrows": hhdr.get("nrows"), "ncols": hhdr.get("ncols")},
        "reprojected_value_counts": {"perm": _counts(fresh_perm), "flooded": _counts(fresh_flood)},
        "overlap_unshifted": _overlap(fresh_perm, sea),
        "overlap_flood_vs_sea": _overlap(fresh_flood, sea),
        "overlap_flipud": _overlap(np.flipud(fresh_perm), sea),
        "overlap_fliplr": _overlap(np.fliplr(fresh_perm), sea),
        "integer_shifts": shifts,
        "best_integer_shift_by_jaccard": {"drow": best[1], "dcol": best[2], "jaccard": best[0]},
        "half_pixel_and_one_cell_origin_shifts": half,
        "elevation_of_permanent_water": {
            "all_perm": _elev(dem, fresh_perm == 1),
            "perm_on_sea": _elev(dem, (fresh_perm == 1) & sea),
            "perm_on_land": _elev(dem, (fresh_perm == 1) & land),
        },
        "observed_flood_on_land": _elev(dem, (fresh_flood == 1) & land & (fresh_perm != 1)),
    }
    unshifted = report["overlap_unshifted"]["jaccard_perm_sea"] or 0.0
    flipud = report["overlap_flipud"]["jaccard_perm_sea"] or 0.0
    flood_sea = report["overlap_flood_vs_sea"]["jaccard_perm_sea"] or 0.0
    perm_vals = set(report["reprojected_value_counts"]["perm"])
    flood_vals = set(report["reprojected_value_counts"]["flooded"])
    report["binary_01"] = perm_vals <= {"0", "1"} and flood_vals <= {"0", "1"}
    report["band_order_suspect"] = flood_sea > unshifted and flood_sea > 0.0
    report["georef_ok"] = bool(
        gfd["crs"] == "EPSG:4326"
        and gfd["north_up"]
        and gfd["covers_model_grid"]
        and abs(best[1]) <= 2
        and abs(best[2]) <= 2
        and flipud < unshifted
        and report["binary_01"]
        and not report["band_order_suspect"]
    )
    out = Path("results/alignment_report.json") if event == 2734 else Path(f"results/alignment_report_{event}.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "georef_ok", "binary_01", "band_order_suspect", "gfd_geotiff",
        "best_integer_shift_by_jaccard", "overlap_unshifted", "overlap_flipud",
        "overlap_flood_vs_sea", "elevation_of_permanent_water", "observed_flood_on_land",
        "reprojected_value_counts")}, indent=2))

    gy, gx = np.gradient(dem.astype(np.float64))
    hill = np.clip(0.55 - 0.45 * (gx + gy) / (np.percentile(np.abs(gx) + np.abs(gy), 98) + 1e-9), 0, 1)
    rgb = np.dstack([hill, hill, hill])
    rgb[sea] = (0.75, 0.84, 0.90)
    rgb[fresh_perm == 1] = (0.15, 0.35, 0.75)
    rgb[(fresh_flood == 1) & land & (fresh_perm != 1)] = (0.85, 0.15, 0.15)
    model = np.zeros(dem.shape, dtype=bool) if hmax is None else (hmax >= 0.10) & land
    edge = model & ~(_shift(model.astype(np.uint8), 0, 1).astype(bool)
                     & _shift(model.astype(np.uint8), 0, -1).astype(bool)
                     & _shift(model.astype(np.uint8), 1, 0).astype(bool)
                     & _shift(model.astype(np.uint8), -1, 0).astype(bool))
    rgb[edge] = (0.95, 0.85, 0.10)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8.2))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=(0.75, 0.84, 0.90), edgecolor="none", label="sea"),
        Patch(facecolor=(0.15, 0.35, 0.75), edgecolor="none", label="permanent water"),
        Patch(facecolor=(0.85, 0.15, 0.15), edgecolor="none", label="observed flood"),
        Patch(facecolor=(0.95, 0.85, 0.10), edgecolor="none", label="model outline, 0.10 m"),
    ], fontsize=11, loc="lower right", framealpha=0.9)
    figdir = Path("results/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    fig_name = "validation_alignment.png" if event == 2734 else f"validation_alignment_{event}.png"
    paper_name = "F15_alignment.png" if event == 2734 else f"F15_alignment_{event}.png"
    fig.savefig(figdir / fig_name, dpi=120)
    paper = Path("paper/shared/figures")
    paper.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper / paper_name, dpi=120)
    print("wrote", figdir / fig_name, "and", paper / paper_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
