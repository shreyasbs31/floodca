"""IMD rainfall processing (SPEC Sec 4.3, 4.4 V0 forcing)."""
from __future__ import annotations

import argparse
import calendar
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# IMD 0.25 deg daily rainfall is conventionally accumulated ending 08:30 IST
# (03:00 UTC). Recorded as the conventional reporting day, not read from a file header.
IMD_DAY_CONVENTION = "daily total ending 08:30 IST (03:00 UTC)"
NLAT = 129
NLON = 135
LAT0, LAT1, DLAT = 6.5, 38.5, 0.25
LON0, LON1, DLON = 66.5, 100.0, 0.25

_MISSING_V0 = {
    "status": "missing_imd",
    "need": "2005.grd under data/raw/imd/ or data/raw/imd/rain/",
    "mac": (
        "python3 prep/download_imd_mac.py --start 2005 --end 2005 "
        "--out data/raw/imd"
    ),
    "rsync": (
        "rsync -avP data/raw/imd/ "
        "${FCA_LOGIN}:${FCA_DATA}/raw/imd/"
    ),
}


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
        "units": "m",
        "imd_day_convention": IMD_DAY_CONVENTION,
    }
    if extra:
        h.update(extra)
    return h


def _find_grd(raw_dir: Path, year: int) -> Path | None:
    candidates = [
        raw_dir / "rain" / f"{year}.grd",
        raw_dir / f"{year}.grd",
        raw_dir / "rain" / f"Rainfall_ind{year}_rfp25.grd",
        raw_dir / f"Rainfall_ind{year}_rfp25.grd",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _available_years(raw_dir: Path) -> list[int]:
    return [y for y in range(1991, 2024) if _find_grd(raw_dir, y) is not None]


def _fill_nearest(a):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    invalid = ~np.isfinite(a) | (a < 0)
    if not invalid.any() or invalid.all():
        out = a.copy()
        out[invalid] = 0.0
        return out
    filled = a.copy()
    _, idx = distance_transform_edt(invalid, return_distances=True, return_indices=True)
    filled[invalid] = a[tuple(idx[:, invalid])]
    filled[~np.isfinite(filled)] = 0.0
    return filled


def _monotonic_latlon(lat, lon, field):
    import numpy as np

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field = field[::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        field = field[:, ::-1]
    return lat, lon, field


def _interp_to_grid(field_latlon, lat, lon, hdr):
    """field_latlon is (nlat, nlon) in mm/day. Returns metres on the model grid."""
    import numpy as np
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    lat, lon, field_latlon = _monotonic_latlon(lat, lon, field_latlon)
    nrows, ncols = int(hdr["nrows"]), int(hdr["ncols"])
    dx = float(hdr["dx"])
    x0, y0 = float(hdr["x0"]), float(hdr["y0"])
    xs = x0 + (np.arange(ncols) + 0.5) * dx
    ys = y0 - (np.arange(nrows) + 0.5) * dx
    xx, yy = np.meshgrid(xs, ys)
    transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    lon_m, lat_m = transformer.transform(xx, yy)
    interp = RegularGridInterpolator(
        (lat, lon),
        field_latlon,
        bounds_error=False,
        fill_value=0.0,
        method="linear",
    )
    pts = np.column_stack([lat_m.ravel(), lon_m.ravel()])
    out_mm = interp(pts).reshape(nrows, ncols)
    out_mm = np.clip(out_mm, 0.0, None)
    return (out_mm / 1000.0).astype(np.float32)


def _native_year(path: Path, year: int):
    import numpy as np
    import pandas as pd

    ndays = 366 if calendar.isleap(year) else 365
    raw = np.fromfile(path, dtype="<f4")
    expected = ndays * NLAT * NLON
    if raw.size != expected:
        raise RuntimeError(f"{path} size {raw.size} != {expected} for year {year}")
    rain = raw.reshape(ndays, NLAT, NLON).astype(np.float64)
    rain = np.where(rain < 0, np.nan, rain)
    lat = np.arange(LAT0, LAT1 + DLAT / 2.0, DLAT)
    lon = np.arange(LON0, LON1 + DLON / 2.0, DLON)
    if lat.size != NLAT or lon.size != NLON:
        raise RuntimeError(f"IMD lat/lon size {lat.size} {lon.size}")
    times = pd.date_range(f"{year}-01-01", periods=ndays, freq="D")
    return rain, lat, lon, times


def _open_year(raw_dir: Path, year: int):
    import numpy as np
    import pandas as pd

    grd = _find_grd(raw_dir, year)
    if grd is None:
        raise FileNotFoundError(f"no IMD grd for {year} under {raw_dir}")
    try:
        import imdlib as imd

        file_dir = str(grd.parent.parent if grd.parent.name == "rain" else grd.parent)
        fn_format = "yearwise" if grd.name == f"{year}.grd" else None
        obj = imd.open_data("rain", year, year, fn_format=fn_format, file_dir=file_dir)
        ds = obj.get_xarray()
        rain = np.asarray(ds["rain"].values, dtype=np.float64)
        rain = np.where(rain < 0, np.nan, rain)
        lat = np.asarray(ds["lat"].values)
        lon = np.asarray(ds["lon"].values)
        times = pd.to_datetime(ds["time"].values)
        return rain, lat, lon, times
    except Exception:
        return _native_year(grd, year)


def _aoi_bbox(aoi_yaml: Path) -> tuple[float, float, float, float]:
    import yaml

    aoi = yaml.safe_load(aoi_yaml.read_text(encoding="utf-8"))
    pad = 0.25
    return (
        float(aoi["lon_min"]) - pad,
        float(aoi["lon_max"]) + pad,
        float(aoi["lat_min"]) - pad,
        float(aoi["lat_max"]) + pad,
    )


def write_v0_forcing(
    raw_dir: Path,
    proc_root: Path,
    aoi_yaml: Path,
    start: str = "2005-09-19",
    end: str = "2005-10-11",
    write_30: bool = False,
    name: str = "V0",
) -> dict:
    year = int(start[:4])
    if int(end[:4]) != year:
        raise RuntimeError(f"forcing window {start} to {end} must fall in one IMD year")
    if _find_grd(raw_dir, year) is None:
        report = dict(_MISSING_V0)
        report["need"] = f"{year}.grd under data/raw/imd/ or data/raw/imd/rain/"
        out_dir = proc_root / f"forcing_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (proc_root / f"forcing_{name}_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report
    import numpy as np
    import pandas as pd
    from floodca_prep.io_bin import write_bin

    rain, lat, lon, times = _open_year(raw_dir, year)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    hdr90 = json.loads((proc_root / "90" / "dem.json").read_text(encoding="utf-8"))
    hdr30 = json.loads((proc_root / "30" / "dem.json").read_text(encoding="utf-8"))
    lon_min, lon_max, lat_min, lat_max = _aoi_bbox(aoi_yaml)
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = (lon >= lon_min) & (lon <= lon_max)

    out_dir = proc_root / f"forcing_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    times_norm = pd.DatetimeIndex(times).normalize()
    day_index = 0
    for t in pd.date_range(start_ts, end_ts, freq="D"):
        match = np.where(times_norm == t.normalize())[0]
        if len(match) == 0:
            raise RuntimeError(f"IMD {year} missing date {t.date()}")
        sl = np.asarray(rain[int(match[0])], dtype=np.float64)
        sub = sl[np.ix_(lat_sel, lon_sel)]
        p_mean = float(np.nanmean(sub))
        p_max = float(np.nanmax(sub))
        sl = _fill_nearest(sl)
        r90 = _interp_to_grid(sl, lat, lon, hdr90)
        date_s = t.strftime("%Y-%m-%d")
        extra = {"date": date_s, "day_index": day_index, "p_mean_mm": p_mean}
        f90 = (out_dir / f"rain_{date_s}_90.bin").resolve()
        write_bin(
            f90,
            r90,
            _header(f"rain_{date_s}_90", r90, 90.0, hdr90["x0"], hdr90["y0"], "float32", "IMD 0.25 deg", extra),
        )
        row = {
            "day_index": day_index,
            "date": date_s,
            "file": str(f90),
            "p_mean_mm": p_mean,
            "p_max_cell_mm": p_max,
        }
        if write_30:
            r30 = _interp_to_grid(sl, lat, lon, hdr30)
            f30 = (out_dir / f"rain_{date_s}_30.bin").resolve()
            write_bin(
                f30,
                r30,
                _header(f"rain_{date_s}_30", r30, 30.0, hdr30["x0"], hdr30["y0"], "float32", "IMD 0.25 deg", extra),
            )
            row["file_30"] = str(f30)
        rows.append(row)
        day_index += 1

    csv_path = proc_root / f"forcing_{name}.csv"
    fields = ["day_index", "date", "file", "p_mean_mm", "p_max_cell_mm"]
    if write_30:
        fields.insert(3, "file_30")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    report = {
        "status": "ok",
        "imd_day_convention": IMD_DAY_CONVENTION,
        "start": start,
        "end": end,
        "n_days": len(rows),
        "T_sim_s": len(rows) * 86400,
        "csv": str(csv_path.resolve()),
        "p_mean_mm_sum": float(sum(r["p_mean_mm"] for r in rows)),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (proc_root / f"forcing_{name}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def write_rain_fields(raw_dir: Path, dates: list[str], targets: dict[str, tuple[dict, Path]]) -> list[dict]:
    """Writes rain_<date>.bin (metres, float32) for every date on every target grid.

    targets maps a grid name to (dem.json header, output dir). Same fill and bilinear
    interpolation as the V0 forcing. Returns one row per (date, grid).
    """
    import numpy as np
    import pandas as pd
    from floodca_prep.io_bin import write_bin

    by_year: dict[int, list[str]] = {}
    for d in sorted(set(dates)):
        by_year.setdefault(int(d[:4]), []).append(d)
    rows = []
    for year, ds in sorted(by_year.items()):
        rain, lat, lon, times = _open_year(raw_dir, year)
        times_norm = pd.DatetimeIndex(times).normalize()
        for d in ds:
            match = np.where(times_norm == pd.Timestamp(d))[0]
            if len(match) == 0:
                raise RuntimeError(f"IMD {year} missing date {d}")
            sl = _fill_nearest(np.asarray(rain[int(match[0])], dtype=np.float64))
            for name, (hdr, out_dir) in targets.items():
                out_dir.mkdir(parents=True, exist_ok=True)
                r = _interp_to_grid(sl, lat, lon, hdr)
                path = out_dir / f"rain_{d}.bin"
                write_bin(
                    path,
                    r,
                    _header(f"rain_{d}", r, hdr["dx"], hdr["x0"], hdr["y0"], "float32", "IMD 0.25 deg",
                            {"date": d, "grid": name}),
                )
                rows.append({"date": d, "grid": name, "file": str(path),
                             "mean_mm": float(r.mean() * 1000.0)})
    return rows


def write_event_pool(raw_dir: Path, proc_root: Path, aoi_yaml: Path) -> dict:
    years = _available_years(raw_dir)
    need = list(range(1991, 2024))
    if years != need:
        return {
            "status": "skipped",
            "years": years,
            "reason": "need yearly IMD files 1991 to 2023 for the event pool",
        }
    import numpy as np
    import pandas as pd

    import xarray as xr

    lon_min, lon_max, lat_min, lat_max = _aoi_bbox(aoi_yaml)
    series = []
    fields = {}
    sub_lat = sub_lon = None
    for year in years:
        rain, lat, lon, times = _open_year(raw_dir, year)
        lat_sel = (lat >= lat_min) & (lat <= lat_max)
        lon_sel = (lon >= lon_min) & (lon <= lon_max)
        sub_lat, sub_lon = np.asarray(lat)[lat_sel], np.asarray(lon)[lon_sel]
        times_norm = pd.DatetimeIndex(times).normalize()
        for i, t in enumerate(times_norm):
            sl = np.asarray(rain[i], dtype=np.float64)
            sub = sl[np.ix_(lat_sel, lon_sel)]
            p_mean = float(np.nanmean(sub)) if np.isfinite(sub).any() else float("nan")
            p_max = float(np.nanmax(sub)) if np.isfinite(sub).any() else 0.0
            date_s = t.strftime("%Y-%m-%d")
            series.append({"date": date_s, "p_mean_mm": p_mean, "p_max_cell_mm": p_max})
            if 6 <= t.month <= 9:
                fields[date_s] = sub
    df_all = pd.DataFrame(series)
    df_all.to_csv(proc_root / "imd_aoi_series_1991_2023.csv", index=False)
    month = pd.to_datetime(df_all["date"]).dt.month
    df = df_all[(month >= 6) & (month <= 9)]
    wet = df[df["p_mean_mm"] >= 2.5]
    q = float(wet["p_mean_mm"].quantile(0.95))
    pool = wet[wet["p_mean_mm"] > q].copy()
    threshold = 0.95
    if len(pool) < 50:
        q = float(wet["p_mean_mm"].quantile(0.90))
        pool = wet[wet["p_mean_mm"] > q].copy()
        threshold = 0.90
    pool_path = proc_root / "event_pool.csv"
    pool.to_csv(pool_path, index=False)
    dates = list(pool["date"])
    ds = xr.Dataset(
        {"rain": (("date", "lat", "lon"), np.stack([fields[d] for d in dates]).astype(np.float32))},
        coords={"date": dates, "lat": sub_lat, "lon": sub_lon},
        attrs={"units": "mm/day", "source": "IMD 0.25 deg", "imd_day_convention": IMD_DAY_CONVENTION},
    )
    ds.to_netcdf(proc_root / "event_fields.nc", engine="scipy")
    report = {
        "status": "ok",
        "years": [years[0], years[-1]],
        "n_monsoon_days": int(len(df)),
        "n_wet_days": int(len(wet)),
        "n_events": int(len(pool)),
        "threshold_quantile": threshold,
        "q_mm": q,
        "csv": str(pool_path),
        "max_p_mean_date": str(pool.loc[pool["p_mean_mm"].idxmax(), "date"]),
        "max_p_mean_mm": float(pool["p_mean_mm"].max()),
    }
    (proc_root / "event_pool_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/imd")
    ap.add_argument("--out", default="data/processed/udupi_coast")
    ap.add_argument("--aoi", default="config/aoi_udupi.yaml")
    ap.add_argument("--start", default="2005-09-19")
    ap.add_argument("--end", default="2005-10-11")
    ap.add_argument("--also-30", action="store_true")
    ap.add_argument("--name", default="V0", help="output stem: forcing_<name>.csv, does not touch other stems")
    ap.add_argument("--skip-pool", action="store_true", help="do not rewrite the 1991-2023 event pool")
    args = ap.parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    aoi = Path(args.aoi)
    v0 = write_v0_forcing(raw, out, aoi, args.start, args.end, args.also_30, args.name)
    pool = None if args.skip_pool else write_event_pool(raw, out, aoi)
    print(json.dumps({"forcing": v0, "event_pool": pool}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
