"""Ensemble scenario generation (SPEC Sec 6.1, 6.3).

Writes scenarios_N<N>.csv (scenario_id, base_date, lambda, D_h, phi, seed) from the
event pool with numpy.random.default_rng(20260915), then precomputes one rain field
per unique base_date on the 90 m grid (rain/90/rain_<date>.bin). The backend applies
lambda at load time. The B0 field is also written for every benchmark grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

MASTER_SEED = 20260915
FIELDS = ["scenario_id", "base_date", "lambda", "D_h", "phi", "seed"]
BENCH_GRIDS = ["S", "M", "L", "30", "90"]
VAL_EVENT = ("2005-09-22", "2005-10-11")  # GFD 2734 began and ended dates


def read_pool(pool_csv: Path) -> list[dict]:
    with pool_csv.open(encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def b0_date(pool: list[dict]) -> str:
    """SPEC 6.3: the pool date with the largest p_mean_mm (first one on ties)."""
    best = max(range(len(pool)), key=lambda i: (float(pool[i]["p_mean_mm"]), -i))
    return pool[best]["date"]


def generate(pool_dates: list[str], n: int, master_seed: int = MASTER_SEED) -> list[dict]:
    import numpy as np

    rng = np.random.default_rng(master_seed)
    idx = rng.integers(0, len(pool_dates), size=n)
    lam = rng.uniform(0.8, 1.5, size=n)
    dur = rng.choice(np.array([6, 12, 24]), size=n)
    phi = rng.uniform(0.25, 0.75, size=n)
    seeds = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    return [
        {
            "scenario_id": k,
            "base_date": pool_dates[int(idx[k])],
            "lambda": f"{float(lam[k]):.17g}",
            "D_h": int(dur[k]),
            "phi": f"{float(phi[k]):.17g}",
            "seed": int(seeds[k]),
        }
        for k in range(n)
    ]


def event_dates(first: str = VAL_EVENT[0], last: str = VAL_EVENT[1]) -> list[str]:
    from datetime import date, timedelta

    d0, d1 = date.fromisoformat(first), date.fromisoformat(last)
    return [(d0 + timedelta(days=k)).isoformat() for k in range((d1 - d0).days + 1)]


def generate_validation(dates: list[str], n: int = 64, master_seed: int = MASTER_SEED) -> list[dict]:
    """SPEC 6.5: base_date uniform over the observed event dates, lambda ~ U(0.9, 1.1);
    D and phi as in Sec 6.1. The stream is seeded with [master_seed, 1] so that it is
    derived from, but independent of, the main ensemble."""
    import numpy as np

    rng = np.random.default_rng([master_seed, 1])
    idx = rng.integers(0, len(dates), size=n)
    lam = rng.uniform(0.9, 1.1, size=n)
    dur = rng.choice(np.array([6, 12, 24]), size=n)
    phi = rng.uniform(0.25, 0.75, size=n)
    seeds = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    return [
        {
            "scenario_id": k,
            "base_date": dates[int(idx[k])],
            "lambda": f"{float(lam[k]):.17g}",
            "D_h": int(dur[k]),
            "phi": f"{float(phi[k]):.17g}",
            "seed": int(seeds[k]),
        }
        for k in range(n)
    ]


def to_csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc", default="data/processed/udupi_coast")
    ap.add_argument("--raw", default="data/raw/imd")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--no-rain", action="store_true", help="only write the scenario CSV")
    ap.add_argument("--validation", action="store_true",
                    help="write the Sec 6.5 validation ensemble scenarios_val_N<N>.csv instead")
    args = ap.parse_args()
    proc = Path(args.proc)
    if args.validation:
        return main_validation(proc, Path(args.raw), args.n, args.no_rain)

    pool = read_pool(proc / "event_pool.csv")
    dates = [r["date"] for r in pool]
    rows = generate(dates, args.n)
    text = to_csv_text(rows)
    out_csv = proc / f"scenarios_N{args.n}.csv"
    out_csv.write_text(text, encoding="utf-8")
    unique = sorted({r["base_date"] for r in rows})
    b0 = b0_date(pool)
    report = {
        "status": "ok",
        "n": args.n,
        "master_seed": MASTER_SEED,
        "pool_size": len(pool),
        "unique_base_dates": len(unique),
        "csv": str(out_csv),
        "csv_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "b0": {"base_date": b0, "lambda": 1.0, "D_h": 24, "phi": 0.5},
    }
    if not args.no_rain:
        from floodca_prep.rainfall import write_rain_fields

        def hdr(g: str) -> dict:
            return json.loads((proc / g / "dem.json").read_text(encoding="utf-8"))

        rows90 = write_rain_fields(
            Path(args.raw), sorted(set(unique) | {b0}), {"90": (hdr("90"), proc / "rain" / "90")}
        )
        rowsb0 = write_rain_fields(
            Path(args.raw), [b0], {g: (hdr(g), proc / "rain" / g) for g in BENCH_GRIDS if g != "90"}
        )
        report["rain_fields_90"] = len(rows90)
        report["b0_fields"] = {r["grid"]: r["file"] for r in rowsb0}
        report["b0_fields"]["90"] = str(proc / "rain" / "90" / f"rain_{b0}.bin")
        report["b0_mean_mm"] = {r["grid"]: r["mean_mm"] for r in rowsb0}
    (proc / f"scenarios_N{args.n}_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


def main_validation(proc: Path, raw: Path, n: int, no_rain: bool) -> int:
    dates = event_dates()
    rows = generate_validation(dates, n)
    text = to_csv_text(rows)
    out_csv = proc / f"scenarios_val_N{n}.csv"
    out_csv.write_text(text, encoding="utf-8")
    unique = sorted({r["base_date"] for r in rows})
    report = {
        "status": "ok",
        "n": n,
        "seed": [MASTER_SEED, 1],
        "event": list(VAL_EVENT),
        "event_dates": len(dates),
        "unique_base_dates": len(unique),
        "csv": str(out_csv),
        "csv_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if not no_rain:
        from floodca_prep.rainfall import write_rain_fields

        hdr = json.loads((proc / "90" / "dem.json").read_text(encoding="utf-8"))
        rows90 = write_rain_fields(raw, unique, {"90": (hdr, proc / "rain" / "90")})
        report["rain_fields_90"] = len(rows90)
    (proc / f"scenarios_val_N{n}_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                          encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
