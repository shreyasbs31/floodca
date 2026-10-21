"""Read and write FloodCA .bin grids plus JSON headers (SPEC Sec 7.3)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "uint8": np.uint8,
    "int32": np.int32,
}


def write_bin(path: str | Path, array: np.ndarray, header: dict[str, Any]) -> None:
    path = Path(path)
    arr = np.ascontiguousarray(array)
    data = arr.tobytes(order="C")
    digest = hashlib.sha256(data).hexdigest()
    header = dict(header)
    header["sha256"] = digest
    header["created"] = header.get("created") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    path.write_bytes(data)
    path.with_suffix(".json").write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")


def read_bin(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    header = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    dtype = DTYPE_MAP[header["dtype"]]
    data = np.frombuffer(path.read_bytes(), dtype=dtype)
    nrows = int(header["nrows"])
    ncols = int(header["ncols"])
    arr = data.reshape((nrows, ncols))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if header.get("sha256") and header["sha256"] != digest:
        raise ValueError(f"sha256 mismatch for {path}")
    return arr, header
