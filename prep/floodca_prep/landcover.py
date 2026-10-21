"""WorldCover to Manning n and runoff C (SPEC Sec 4.6)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.warp import Resampling, reproject

from floodca_prep.io_bin import read_bin, write_bin


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


def load_lc_table(path: Path) -> dict[int, tuple[float, float]]:
    table: dict[int, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = int(row["worldcover_code"])
            table[code] = (float(row["manning_n"]), float(row["runoff_C"]))
    return table


def _agg_majority_3(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    x = a.reshape(h // 3, 3, w // 3, 3).transpose(0, 2, 1, 3).reshape(h // 3, w // 3, 9)
    codes, inverse = np.unique(x, return_inverse=True)
    inv = inverse.reshape(h // 3, w // 3, 9)
    counts = np.stack([(inv == i).sum(axis=2) for i in range(len(codes))], axis=-1)
    return codes[counts.argmax(axis=-1)].astype(a.dtype)


def _agg_mean_3(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h // 3, 3, w // 3, 3).mean(axis=(1, 3)).astype(np.float32)


def _lookup_n_c(lc: np.ndarray, table: dict[int, tuple[float, float]], k_n: float, k_c: float):
    default = table.get(-1, (0.035, 0.40))
    n = np.empty(lc.shape, dtype=np.float32)
    C = np.empty(lc.shape, dtype=np.float32)
    for code in np.unique(lc):
        n0, c0 = table.get(int(code), default)
        sel = lc == code
        n[sel] = n0
        C[sel] = c0
    n *= k_n
    C = np.clip(C * k_c, 0.0, 1.0)
    return n, C


def process(
    wc_tif: Path,
    proc_root: Path,
    params_csv: Path,
    k_n: float = 1.0,
    k_c: float = 1.0,
) -> dict:
    dem, dem_hdr = read_bin(proc_root / "30" / "dem.bin")
    nrows, ncols = dem.shape
    x0, y0, dx = dem_hdr["x0"], dem_hdr["y0"], dem_hdr["dx"]
    dst_transform = rasterio.transform.from_origin(x0, y0, dx, dx)
    lc30 = np.zeros((nrows, ncols), dtype=np.uint8)
    with rasterio.open(wc_tif) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=lc30,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:32643",
            src_nodata=src.nodata if src.nodata is not None else 0,
            dst_nodata=0,
            resampling=Resampling.mode,
        )
    table = load_lc_table(params_csv)
    n30, c30 = _lookup_n_c(lc30, table, k_n, k_c)
    lc90 = _agg_majority_3(lc30)
    n90 = _agg_mean_3(n30)
    c90 = _agg_mean_3(c30)
    c90 = np.clip(c90, 0.0, 1.0)
    extra = {"aoi": dem_hdr.get("aoi", "udupi_coast"), "k_n": k_n, "k_C": k_c}
    src_name = "ESA WorldCover v200"
    out30 = proc_root / "30"
    out90 = proc_root / "90"
    write_bin(out30 / "lc.bin", lc30, _header("lc_30", lc30, 30.0, x0, y0, "uint8", src_name, extra))
    write_bin(out30 / "n.bin", n30, _header("n_30", n30, 30.0, x0, y0, "float32", src_name, extra))
    write_bin(out30 / "C.bin", c30, _header("C_30", c30, 30.0, x0, y0, "float32", src_name, extra))
    write_bin(out90 / "lc.bin", lc90, _header("lc_90", lc90, 90.0, x0, y0, "uint8", src_name, extra))
    write_bin(out90 / "n.bin", n90, _header("n_90", n90, 90.0, x0, y0, "float32", src_name, extra))
    write_bin(out90 / "C.bin", c90, _header("C_90", c90, 90.0, x0, y0, "float32", src_name, extra))

    report = json.loads((proc_root / "dem_report.json").read_text(encoding="utf-8"))
    for name, crop in report["crops"].items():
        rs, cs = int(crop["row0"]), int(crop["col0"])
        h, w = int(crop["nrows"]), int(crop["ncols"])
        cx0 = x0 + cs * 30.0
        cy0 = y0 - rs * 30.0
        cdir = proc_root / name
        sl_n = n30[rs : rs + h, cs : cs + w]
        sl_c = c30[rs : rs + h, cs : cs + w]
        sl_l = lc30[rs : rs + h, cs : cs + w]
        write_bin(cdir / "n.bin", sl_n, _header(f"n_{name}", sl_n, 30.0, cx0, cy0, "float32", src_name, extra))
        write_bin(cdir / "C.bin", sl_c, _header(f"C_{name}", sl_c, 30.0, cx0, cy0, "float32", src_name, extra))
        write_bin(cdir / "lc.bin", sl_l, _header(f"lc_{name}", sl_l, 30.0, cx0, cy0, "uint8", src_name, extra))

    summary = {
        "lc30_unique": [int(v) for v in np.unique(lc30)],
        "n_min": float(n30.min()),
        "n_max": float(n30.max()),
        "C_min": float(c30.min()),
        "C_max": float(c30.max()),
        "k_n": k_n,
        "k_C": k_c,
        "source": str(wc_tif),
    }
    (proc_root / "landcover_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _find_worldcover(path: Path) -> Path:
    if path.is_file():
        return path
    gee = path.parent if path.parent.name else Path("data/raw/gee")
    hits = sorted(gee.glob("*worldcover*.tif"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"WorldCover GeoTIFF not found at {path} or {gee}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worldcover", default="data/raw/gee/floodca_worldcover_v200_udupi.tif")
    ap.add_argument("--params", default="config/landcover_params.csv")
    ap.add_argument("--out", default="data/processed/udupi_coast")
    ap.add_argument("--defaults", default="config/default.yaml")
    args = ap.parse_args()
    k_n, k_c = 1.0, 1.0
    p = Path(args.defaults)
    if p.exists():
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        k_n = float(cfg.get("k_n", 1.0))
        k_c = float(cfg.get("k_C", 1.0))
    wc = _find_worldcover(Path(args.worldcover))
    summary = process(wc, Path(args.out), Path(args.params), k_n, k_c)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
