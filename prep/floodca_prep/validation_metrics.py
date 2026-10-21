"""SPEC Sec 10.1 comparison helpers.

Permanent water exclusion uses the GFD jrc_perm_water band, never the
standalone JRC GSW max_extent layer.
"""
from __future__ import annotations

import numpy as np
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


def exclusion_mask(
    jrc_perm_water: np.ndarray,
    sea_or_wall: np.ndarray,
    clear_perc: np.ndarray | None = None,
    clear_min: float = 50.0,
) -> np.ndarray:
    """True where a cell is eligible for CSI and related metrics."""
    valid = (jrc_perm_water != 1) & (~sea_or_wall)
    if clear_perc is not None:
        valid &= np.isfinite(clear_perc) & (clear_perc >= clear_min)
    return valid


def land_flood_fraction(
    flooded: np.ndarray,
    jrc_perm_water: np.ndarray,
    land: np.ndarray,
) -> float:
    land = land.astype(bool)
    if not land.any():
        return 0.0
    lf = (flooded == 1) & (jrc_perm_water != 1) & land
    return float(lf.sum() / land.sum())


def resample_binary_average(
    src: np.ndarray,
    src_dx: float,
    src_x0: float,
    src_y0: float,
    dst_shape: tuple[int, int],
    dst_dx: float,
    dst_x0: float,
    dst_y0: float,
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(src.astype(np.float32)),
        destination=dst,
        src_transform=from_origin(src_x0, src_y0, src_dx, src_dx),
        src_crs="EPSG:32643",
        dst_transform=from_origin(dst_x0, dst_y0, dst_dx, dst_dx),
        dst_crs="EPSG:32643",
        resampling=Resampling.average,
    )
    return dst


def resample_nearest_u8(
    src: np.ndarray,
    src_dx: float,
    src_x0: float,
    src_y0: float,
    dst_shape: tuple[int, int],
    dst_dx: float,
    dst_x0: float,
    dst_y0: float,
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=np.uint8)
    reproject(
        source=np.ascontiguousarray(src.astype(np.uint8)),
        destination=dst,
        src_transform=from_origin(src_x0, src_y0, src_dx, src_dx),
        src_crs="EPSG:32643",
        dst_transform=from_origin(dst_x0, dst_y0, dst_dx, dst_dx),
        dst_crs="EPSG:32643",
        resampling=Resampling.nearest,
    )
    return dst


def confusion_counts(
    obs_flooded: np.ndarray,
    model_flooded: np.ndarray,
    valid: np.ndarray,
) -> dict[str, int]:
    o = (obs_flooded == 1) & valid
    m = (model_flooded == 1) & valid
    h = int((o & m).sum())
    f = int((~o & m).sum())
    miss = int((o & ~m).sum())
    n = int((~o & ~m & valid).sum())
    return {"H": h, "F": f, "M": miss, "N": n}


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    h, f, m = c["H"], c["F"], c["M"]
    denom_csi = h + f + m
    denom_pod = h + m
    denom_far = h + f
    denom_bias = h + m
    denom_f1 = 2 * h + f + m
    return {
        "CSI": (h / denom_csi) if denom_csi else 0.0,
        "POD": (h / denom_pod) if denom_pod else 0.0,
        "FAR": (f / denom_far) if denom_far else 0.0,
        "Bias": ((h + f) / denom_bias) if denom_bias else 0.0,
        "F1": ((2 * h) / denom_f1) if denom_f1 else 0.0,
    }


def model_flood_on_obs_grid(
    model_flag_90: np.ndarray,
    hdr90: dict,
    obs_shape: tuple[int, int],
    obs_dx: float,
    obs_x0: float,
    obs_y0: float,
    frac_threshold: float = 0.5,
) -> np.ndarray:
    frac = resample_binary_average(
        model_flag_90,
        float(hdr90["dx"]),
        float(hdr90["x0"]),
        float(hdr90["y0"]),
        obs_shape,
        obs_dx,
        obs_x0,
        obs_y0,
    )
    return (frac >= frac_threshold).astype(np.uint8)
