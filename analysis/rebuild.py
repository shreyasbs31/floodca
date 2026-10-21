"""Regenerate the printed numbers in the FloodCA paper from results/.

Each check carries the string as printed and the value recomputed from the
saved rows. Rounding follows the decimal places of that printed string.
Nothing here edits a results file.
"""
from __future__ import annotations

import csv
import json
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

THREADS = (1, 2, 4, 8, 16, 20, 32, 40)
RANKS = (1, 2, 4, 8, 16, 32, 40)

# Printed OpenMP cells, transcribed from the paper. Order is
# T, speedup vs serial, efficiency vs serial, speedup vs 1 thread, efficiency vs 1 thread.
OMP = {
    "S": {
        1: ("62.6", "1.10", "1.10", "1.00", "1.00"),
        2: ("31.5", "2.18", "1.09", "1.99", "0.993"),
        4: ("16.1", "4.25", "1.06", "3.88", "0.970"),
        8: ("8.17", "8.40", "1.05", "7.66", "0.958"),
        16: ("4.24", "16.2", "1.01", "14.8", "0.924"),
        20: ("3.47", "19.8", "0.989", "18.1", "0.903"),
        32: ("2.37", "28.9", "0.904", "26.4", "0.825"),
        40: ("2.06", "33.4", "0.834", "30.5", "0.761"),
    },
    "M": {
        1: ("246", "1.10", "1.10", "1.00", "1.00"),
        2: ("126", "2.14", "1.07", "1.95", "0.974"),
        4: ("63.7", "4.25", "1.06", "3.86", "0.966"),
        8: ("32.3", "8.37", "1.05", "7.62", "0.952"),
        16: ("16.5", "16.4", "1.03", "14.9", "0.933"),
        20: ("13.4", "20.2", "1.01", "18.4", "0.918"),
        32: ("8.46", "32.0", "1.00", "29.1", "0.910"),
        40: ("7.04", "38.5", "0.961", "35.0", "0.875"),
    },
    "L": {
        1: ("975", "1.10", "1.10", "1.00", "1.00"),
        2: ("502", "2.13", "1.07", "1.94", "0.972"),
        4: ("253", "4.22", "1.06", "3.86", "0.964"),
        8: ("129", "8.28", "1.03", "7.55", "0.944"),
        16: ("67.3", "15.9", "0.992", "14.5", "0.906"),
        20: ("54.9", "19.5", "0.974", "17.8", "0.889"),
        32: ("34.5", "31.0", "0.969", "28.3", "0.884"),
        40: ("28.0", "38.1", "0.953", "34.8", "0.870"),
    },
}

MPI = {
    1: ("1,069", "1", "1", "0", "0"),
    2: ("549", "1.95", "0.974", "2.92", "2.07"),
    4: ("290", "3.69", "0.922", "4.97", "9.54"),
    8: ("146", "7.30", "0.912", "6.57", "10.3"),
    16: ("74.3", "14.4", "0.899", "8.85", "14.4"),
    32: ("38.2", "28.0", "0.875", "8.28", "20.0"),
    40: ("31.1", "34.4", "0.860", "10.2", "21.5"),
}

CROSS = {
    "C32": ("1,024", "0.378", "0.0980", "8", "0.0849", "1.16"),
    "C64": ("4,096", "1.24", "0.178", "10", "0.0858", "2.08"),
    "C128": ("16,384", "4.84", "0.346", "20", "0.0919", "3.76"),
    "C256": ("65,536", "18.5", "0.981", "40", "0.107", "9.18"),
    "S": ("262,144", "68.6", "2.06", "40", "0.203", "10.1"),
    "C768": ("589,824", "152", "4.15", "40", "0.379", "11.0"),
    "M": ("1,048,576", "271", "7.04", "40", "0.588", "12.0"),
    "C1536": ("2,359,296", "608", "15.9", "40", "1.23", "12.9"),
    "L": ("4,184,956", "1,069", "28.0", "40", "2.08", "13.5"),
}

FLUX = {
    ("C128", 20): ("819", "0.154", "94.0"),
    ("C128", 40): ("410", "0.330", "402"),
    ("C256", 20): ("3,277", "0.458", "69.9"),
    ("C256", 40): ("1,638", "0.460", "140"),
    ("C768", 40): ("14,746", "1.76", "59.7"),
    ("C1536", 40): ("58,982", "6.39", "54.2"),
    ("S", 40): ("6,554", "0.878", "67.0"),
    ("M", 40): ("26,214", "2.96", "56.4"),
    ("L", 40): ("104,624", "10.8", "51.5"),
}

SCORES = {
    ("2734", "model_h0.10", "all_valid"): ("0.0272", "0.0405", "0.923", "0.529"),
    ("2734", "model_h0.25", "all_valid"): ("0.00224", "0.00261", "0.984", "0.167"),
    ("2734", "baseline_near_perm_vs_model_h0.10", "all_valid"): ("0.129", "0.714", "0.864", "5.25"),
    ("2734", "baseline_lowest_405_vs_model_h0.10", "all_valid"): ("0.00861", "0.0131", "0.975", "0.529"),
    ("4198", "model_h0.10", "all_valid"): ("0.00285", "0.00433", "0.992", "0.525"),
    ("4198", "model_h0.25", "all_valid"): ("0.00121", "0.00144", "0.993", "0.193"),
    ("4198", "baseline_near_perm_vs_model_h0.10", "all_valid"): ("0.100", "0.687", "0.895", "6.52"),
    ("4198", "baseline_lowest_364_vs_model_h0.10", "all_valid"): ("0.00190", "0.00289", "0.995", "0.525"),
}

LAND = [
    ("10", "0.12", "0.20"),
    ("20", "0.07", "0.30"),
    ("30", "0.035", "0.35"),
    ("40", "0.04", "0.35"),
    ("50", "0.015", "0.85"),
    ("60", "0.025", "0.60"),
    ("70", "0.02", "0.90"),
    ("80", "0.03", "1.00"),
    ("90", "0.06", "0.10"),
    ("95", "0.12", "0.10"),
    ("100", "0.035", "0.30"),
    ("-1", "0.035", "0.40"),
]


def fmt_like(raw, expected: str) -> str:
    """Round raw to the decimal places of the printed string."""
    want_comma = "," in expected
    body = expected.replace(",", "").replace("%", "").strip()
    if body.lower() in {"n/a", "na"}:
        return expected
    if "e" in body.lower():
        mant = body.lower().split("e")[0]
        places = len(mant.split(".")[1]) if "." in mant else 0
        shown = f"{float(raw):.{places}e}"
        coef, exp = shown.split("e")
        text = f"{coef}e{int(exp)}"
        return text
    if "." in body:
        places = len(body.split(".")[1])
    else:
        places = 0
    quantum = Decimal("1") if places == 0 else Decimal("0." + "0" * (places - 1) + "1")
    rounded = Decimal(str(raw)).quantize(quantum, rounding=ROUND_HALF_UP)
    if places == 0:
        text = f"{int(rounded)}"
    else:
        text = f"{rounded:.{places}f}"
    if want_comma:
        neg = text.startswith("-")
        core = text[1:] if neg else text
        whole, dot, frac = core.partition(".")
        whole = f"{int(whole):,}"
        text = ("-" if neg else "") + whole + (dot + frac if dot else "")
    return text


def _load_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_bench():
    rows = []
    for row in _load_csv(RES / "bench.csv"):
        if row.get("warmup") != "0" or row.get("contaminated") not in ("0", ""):
            continue
        row["_g"] = row["grid_name"].replace("grid_", "")
        h2d = _f(row, "t_h2d_s") or 0.0
        d2h = _f(row, "t_d2h_s") or 0.0
        wall = _f(row, "wall_total_s")
        row["_e2e"] = None if wall is None else wall + h2d + d2h
        rows.append(row)
    return rows


BENCH = load_bench()


def pick(pred, prefer_main=True):
    chosen = [row for row in BENCH if pred(row)]
    if prefer_main and any(row["host"] == "main-node" for row in chosen):
        mains = [row for row in chosen if row["host"] == "main-node"]
        if mains and len(mains) < len(chosen):
            return mains
    return chosen


def median_of(pred, field="wall_total_s", prefer_main=True):
    values = []
    for row in pick(pred, prefer_main=prefer_main):
        if field == "_e2e":
            value = row.get("_e2e")
        else:
            value = _f(row, field)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return statistics.median(values)


def e4(grid, backend, threads, field="wall_total_s"):
    return median_of(
        lambda row: row["exp_id"] == "E4"
        and row["backend"] == backend
        and row["_g"] == grid
        and int(float(row["threads"])) == threads
        and row["precision"] == "float"
        and row["neigh"] == "4",
        field,
    )


def timing_rows(grid, backend):
    return lambda threads, field="wall_total_s": e4(grid, backend, threads, field)


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, paper_item, quantity, expected, raw, note=""):
        self.rows.append({
            "paper_item": paper_item,
            "quantity": quantity,
            "expected": expected,
            "raw": raw,
            "note": note,
        })


def _json(name):
    return json.loads((RES / name).read_text())


def build():
    out = Checks()
    _omp(out)
    _mpi(out)
    _cross(out)
    _flux(out)
    _reuse(out)
    _scores(out)
    _ensemble(out)
    _sparse(out)
    _dp(out)
    _energy(out)
    _params(out)
    _landcover(out)
    _abstract(out)
    _extra(out)
    return out.rows


def _omp(out: Checks):
    for grid, cells in OMP.items():
        item = {"S": "Table VI", "M": "Table VII", "L": "Table VIII"}[grid]
        serial = e4(grid, "serial", 1)
        one = e4(grid, "omp", 1)
        for threads, printed in cells.items():
            wall = e4(grid, "omp", threads)
            speed_s = None if serial is None or wall is None else serial / wall
            eff_s = None if speed_s is None else speed_s / threads
            speed_1 = None if one is None or wall is None else one / wall
            eff_1 = None if speed_1 is None else speed_1 / threads
            labels = ("T_s", "speedup_vs_serial", "efficiency_vs_serial",
                      "speedup_vs_1_thread", "efficiency_vs_1_thread")
            raws = (wall, speed_s, eff_s, speed_1, eff_1)
            for label, expected, raw in zip(labels, printed, raws):
                out.add(item, f"{grid} threads {threads} {label}", expected, raw)


def _mpi(out: Checks):
    def pred(ranks):
        return lambda row: (
            row["exp_id"] == "E5" and row["backend"] == "mpi" and row["_g"] == "L"
            and int(float(row["ranks"])) == ranks and row["precision"] == "float"
            and row["neigh"] == "4" and row.get("halo_mode") == "sendrecv2"
        )
    base = median_of(pred(1))
    for ranks, printed in MPI.items():
        wall = median_of(pred(ranks))
        halo = median_of(pred(ranks), "t_halo_s")
        reduce = median_of(pred(ranks), "t_allreduce_s")
        speed = None if base is None or wall in (None, 0) else base / wall
        eff = None if speed is None else speed / ranks
        halo_pct = None if halo is None or wall in (None, 0) else 100.0 * halo / wall
        red_pct = None if reduce is None or wall in (None, 0) else 100.0 * reduce / wall
        raws = (wall, speed, eff, halo_pct, red_pct)
        labels = ("wall_s", "speedup", "efficiency", "halo_percent", "allreduce_percent")
        for label, expected, raw in zip(labels, printed, raws):
            out.add("Table IX", f"ranks {ranks} {label}", expected, raw)
    out.add("Table IX", "bytes sent per step at 4 ranks", "65,600",
            median_of(pred(4), "bytes_sent_per_step"))
    minimal = median_of(
        lambda row: row["exp_id"] == "E9" and row["backend"] == "mpi" and row["_g"] == "L"
        and int(float(row["ranks"])) == 4 and row["neigh"] == "4"
        and row.get("halo_mode") == "minimal",
        "bytes_sent_per_step",
    )
    out.add("Table IX", "minimal halo bytes at 4 ranks K=4", "65,536", minimal)


def _best_omp(grid):
    best_wall, best_threads = None, None
    for threads in (1, 2, 4, 8, 10, 16, 20, 32, 40):
        wall = median_of(
            lambda row, threads=threads: row["exp_id"] in ("E4", "E7")
            and row["backend"] == "omp" and row["_g"] == grid
            and int(float(row["threads"])) == threads
            and row["precision"] == "float" and row["neigh"] == "4"
        )
        if wall is None:
            continue
        if best_wall is None or wall < best_wall:
            best_wall, best_threads = wall, threads
    return best_wall, best_threads


def _cross(out: Checks):
    for grid, printed in CROSS.items():
        cells, serial_s, omp_s, threads_s, cuda_s, ratio_s = printed
        serial = median_of(
            lambda row, grid=grid: row["exp_id"] in ("E4", "E7")
            and row["backend"] == "serial" and row["_g"] == grid
            and row["precision"] == "float" and row["neigh"] == "4"
        )
        omp, threads = _best_omp(grid)
        cuda = median_of(
            lambda row, grid=grid: row["exp_id"] == "E7" and row["backend"] == "cuda"
            and row["_g"] == grid and row["precision"] == "float" and row["neigh"] == "4"
            and row.get("block") == "16x16" and row.get("reduce_impl") == "cub",
            "_e2e",
        )
        active = None
        for row in pick(lambda row, grid=grid: row["_g"] == grid and row["precision"] == "float"):
            active = _f(row, "active_cells")
            if active:
                break
        ratio = None if omp is None or cuda in (None, 0) else omp / cuda
        out.add("Table X", f"{grid} cells", cells, active)
        out.add("Table X", f"{grid} serial_s", serial_s, serial)
        out.add("Table X", f"{grid} best_omp_s", omp_s, omp)
        out.add("Table X", f"{grid} best_omp_threads", threads_s, threads)
        out.add("Table X", f"{grid} cuda_e2e_s", cuda_s, cuda)
        out.add("Table X", f"{grid} ratio", ratio_s, ratio)


def _flux_median(grid, threads):
    exp = "E7" if grid.startswith("C") else "E4"
    return median_of(
        lambda row, grid=grid, threads=threads, exp=exp: row["exp_id"] == exp
        and row["backend"] == "omp" and row["_g"] == grid
        and int(float(row["threads"])) == threads
        and row["precision"] == "float" and row["neigh"] == "4",
        "t_flux_s",
    )


def _flux(out: Checks):
    for (grid, threads), printed in FLUX.items():
        cells_s, flux_s, ns_s = printed
        flux = _flux_median(grid, threads)
        active = None
        exp = "E7" if grid.startswith("C") else "E4"
        for row in pick(lambda row, grid=grid, exp=exp: row["exp_id"] == exp and row["_g"] == grid):
            active = _f(row, "active_cells")
            if active:
                break
        per = None if active is None else active / threads
        ns = None if flux is None or not per else flux / 2000.0 / per * 1e9
        out.add("Table XI", f"{grid} {threads} cells_per_thread", cells_s, per)
        out.add("Table XI", f"{grid} {threads} flux_s", flux_s, flux)
        out.add("Table XI", f"{grid} {threads} ns", ns_s, ns)


def _reuse(out: Checks):
    specs = (
        ("OpenMP 40", "omp", 40, 1, "27.9", "19.1", "31.5"),
        ("MPI 4", "mpi", 1, 4, "290", "196", "32.5"),
        ("MPI 8", "mpi", 1, 8, "146", "98.4", "32.8"),
        ("CUDA", "cuda", 1, 1, "2.01", "1.54", "23.2"),
    )
    for name, backend, threads, ranks, u1_s, u50_s, save_s in specs:
        def pred(u, backend=backend, threads=threads, ranks=ranks):
            return lambda row: (
                row["exp_id"] == "E8" and row["backend"] == backend and row["_g"] == "L"
                and row["precision"] == "float" and row["neigh"] == "4"
                and int(float(row["U"])) == u
                and int(float(row["threads"])) == threads
                and int(float(row["ranks"])) == ranks
            )
        u1 = median_of(pred(1))
        u50 = median_of(pred(50))
        saving = None if u1 in (None, 0) or u50 is None else 100.0 * (u1 - u50) / u1
        out.add("Table XII", f"{name} U=1 s", u1_s, u1)
        out.add("Table XII", f"{name} U=50 s", u50_s, u50)
        out.add("Table XII", f"{name} saving percent", save_s, saving)


def _scores(out: Checks):
    rows = { (r["event_id"], r["score"], r["zone"]): r
             for r in _load_csv(RES / "validation_compare.csv") }
    labels = {
        ("2734", "model_h0.10", "all_valid"): "2734 at 0.10 m",
        ("2734", "model_h0.25", "all_valid"): "2734 at 0.25 m",
        ("2734", "baseline_near_perm_vs_model_h0.10", "all_valid"): "2734 water buffer",
        ("2734", "baseline_lowest_405_vs_model_h0.10", "all_valid"): "2734 lowest N",
        ("4198", "model_h0.10", "all_valid"): "4198 at 0.10 m",
        ("4198", "model_h0.25", "all_valid"): "4198 at 0.25 m",
        ("4198", "baseline_near_perm_vs_model_h0.10", "all_valid"): "4198 water buffer",
        ("4198", "baseline_lowest_364_vs_model_h0.10", "all_valid"): "4198 lowest N",
    }
    for key, printed in SCORES.items():
        row = rows[key]
        name = labels[key]
        for quantity, expected, column in zip(
            ("CSI", "POD", "FAR", "Bias"), printed, ("CSI", "POD", "FAR", "Bias")
        ):
            out.add("Table V", f"{name} {quantity}", expected, float(row[column]))
    diag = _json("v0_diagnostics.json")
    diag2 = _json("val_4198_diagnostics.json")
    out.add("Table V", "event 2734 ROC area", "0.794", diag["ROC_AUC"])
    out.add("Table V", "event 4198 ROC area", "0.788", diag2["ROC_AUC"])
    out.add("Table V", "event 2734 valid cells", "155,559", diag["n_valid"])
    out.add("Table V", "event 4198 valid cells", "155,448", diag2["n_valid"])
    out.add("Table V", "event 2734 observed flooded", "766", diag["n_observed_flood_valid"])
    out.add("Table V", "event 4198 observed flooded", "693", diag2["n_observed_flood_valid"])
    for event, hits, false, miss in (("2734", "31", "374", "735"), ("4198", "3", "361", "690")):
        row = rows[(event, "model_h0.10", "all_valid")]
        out.add("Table V", f"{event} hits", hits, float(row["H"]))
        out.add("Table V", f"{event} false alarms", false, float(row["F"]))
        out.add("Table V", f"{event} misses", miss, float(row["M"]))


def _ensemble(out: Checks):
    cpu = {row["config"]: row for row in _load_csv(RES / "e10_cpu.csv")}
    gpu = {row["config"]: row for row in _load_csv(RES / "e10_gpu100.csv")}
    mapping = (
        ("1 by 40", "b5_1x40", "8", "605", "47.6"),
        ("2 by 20", "b5_2x20", "8", "605", "47.6"),
        ("4 by 10", "b5_4x10", "8", "640", "45.0"),
        ("8 by 5", "b5_8x5", "8", "708", "40.7"),
        ("serial", "serial", "1", "2,470", "1.46"),
    )
    for name, key, n_s, wall_s, rate_s in mapping:
        row = cpu[key]
        out.add("Table XIII", f"{name} scenarios", n_s, float(row["scenarios"]))
        out.add("Table XIII", f"{name} wall_s", wall_s, float(row["wall_s"]))
        out.add("Table XIII", f"{name} per hour", rate_s, float(row["scen_per_hour"]))
    b4 = {row["scenario_id"]: row for row in _load_csv(RES / "e10_gpu100" / "B4.csv")}
    cpu_scen = {row["scenario_id"]: row for row in _load_csv(RES / "e10_cpu" / "b5_1x40.csv")}
    printed = {
        "0": ("70.9", "4.45", "15.9"),
        "1": ("58.2", "4.54", "12.8"),
        "2": ("83.6", "6.46", "12.9"),
        "3": ("71.0", "5.95", "11.9"),
        "4": ("80.1", "6.40", "12.5"),
        "5": ("96.9", "7.48", "12.9"),
        "6": ("83.4", "5.35", "15.6"),
        "7": ("60.6", "5.11", "11.9"),
    }
    gpu_sum = 0.0
    for sid, (cpu_s, gpu_s, ratio_s) in printed.items():
        cpu_wall = float(cpu_scen[sid]["wall_s"])
        gpu_wall = float(b4[sid]["wall_s"])
        gpu_sum += gpu_wall
        out.add("Table XIV", f"id {sid} CPU s", cpu_s, cpu_wall)
        out.add("Table XIV", f"id {sid} GPU s", gpu_s, gpu_wall)
        out.add("Table XIV", f"id {sid} ratio", ratio_s, cpu_wall / gpu_wall)
    out.add("Table XIV", "GPU column sum s", "45.7", gpu_sum)
    out.add("Table XIV", "ensemble over GPU sum", "13.2",
            float(cpu["b5_1x40"]["wall_s"]) / gpu_sum)
    out.add("Table XIII", "1 GPU wall s", "591", float(gpu["B4"]["wall_s"]))
    out.add("Table XIII", "1 GPU per hour", "609", float(gpu["B4"]["scen_per_hour"]))
    out.add("Table XIII", "2 GPU wall s", "298", float(gpu["B6a"]["wall_s"]))
    out.add("Table XIII", "2 GPU per hour", "1,208", float(gpu["B6a"]["scen_per_hour"]))
    out.add("abstract", "two GPUs over one", "1.98",
            float(gpu["B6a"]["scen_per_hour"]) / float(gpu["B4"]["scen_per_hour"]))


def _run_median(path, pred, field="wall_total_s"):
    values = []
    for row in _load_csv(path):
        if row.get("warmup") != "0":
            continue
        if pred(row):
            value = _f(row, field)
            if value is not None:
                values.append(value)
    return statistics.median(values) if values else None


def _sparse(out: Checks):
    path = RES / "esparse_runs.csv"
    def sparse(grid, threads):
        return _run_median(
            path,
            lambda row, grid=grid, threads=threads: row["exp_id"] == "Esparse"
            and row["backend"] == "omp" and row["grid_name"].replace("grid_", "") == grid
            and int(float(row["threads"])) == threads and row["host"] == "second-node",
        )
    anchor = _json("anchor_second_node.json")["rows"]
    dense = {
        ("S", 1): e4("S", "omp", 1),
        ("S", 20): e4("S", "omp", 20),
        ("S", 40): e4("S", "omp", 40),
        ("M", 1): anchor["M_1"]["second_node_median_s"],
        ("M", 20): e4("M", "omp", 20),
        ("M", 40): e4("M", "omp", 40),
        ("L", 40): anchor["L_40"]["second_node_median_s"],
    }
    printed = {
        ("S", 1): ("72.7", "62.6", "1.16"),
        ("S", 10): ("7.58", None, None),
        ("S", 20): ("3.96", "3.47", "1.14"),
        ("S", 40): ("2.30", "2.06", "1.12"),
        ("M", 1): ("286", "250.4", "1.14"),
        ("M", 10): ("30.0", None, None),
        ("M", 20): ("15.2", "13.4", "1.13"),
        ("M", 40): ("8.00", "7.04", "1.14"),
        ("L", 10): ("115", None, None),
        ("L", 40): ("30.6", "28.36", "1.08"),
    }
    for key, (sp_s, de_s, ratio_s) in printed.items():
        grid, threads = key
        sp = sparse(grid, threads)
        out.add("Table XV", f"{grid} {threads} sparse s", sp_s, sp)
        if de_s is not None:
            de = dense[key]
            out.add("Table XV", f"{grid} {threads} dense s", de_s, de)
            out.add("Table XV", f"{grid} {threads} ratio", ratio_s,
                    None if sp is None or not de else sp / de)
    s1 = sparse("S", 1)
    s40 = sparse("S", 40)
    m1 = sparse("M", 1)
    m40 = sparse("M", 40)
    out.add("Table XV", "S 40 self-relative efficiency", "0.790",
            None if not s1 or not s40 else (s1 / s40) / 40)
    out.add("Table XV", "M 40 self-relative efficiency", "0.893",
            None if not m1 or not m40 else (m1 / m40) / 40)
    checks = _json("cpu_checks.json")
    out.add("prose", "sparse S start wet percent", "11.3", checks["wet_start_percent_S"])
    out.add("prose", "sparse M start wet percent", "17.3", checks["wet_start_percent_M"])
    out.add("prose", "sparse L start wet percent", "13.0", checks["wet_start_percent_L"])
    out.add("prose", "sparse S end wet percent", "88.6", checks["wet_end_percent_S"])
    out.add("prose", "sparse M end wet percent", "87.3", checks["wet_end_percent_M"])
    out.add("prose", "sparse L end wet percent", "85.1", checks["wet_end_percent_L"])
    out.add("prose", "S 40 sparse outlier s", "4.65", checks["sparse_max_s_S_40"])
    out.add("prose", "S 40 sparse IQR percent", "3.0", 100.0 * checks["sparse_iqr_over_median_S_40"])


def _single(grid, threads, host):
    return median_of(
        lambda row, grid=grid, threads=threads, host=host: row["exp_id"] in ("E4", "E7")
        and row["backend"] == "omp" and row["_g"] == grid
        and int(float(row["threads"])) == threads
        and row["precision"] == "float" and row["neigh"] == "4"
        and row["host"] == host,
        prefer_main=False,
    )


def _dp(out: Checks):
    path = RES / "edpx_runs.csv"
    def double(grid, threads):
        return _run_median(
            path,
            lambda row, grid=grid, threads=threads: row["exp_id"] == "EDPX"
            and row["backend"] == "omp"
            and row["grid_name"].replace("grid_", "") == grid
            and int(float(row["threads"])) == threads
            and row["precision"] == "double"
            and row["host"] == "second-node",
        )
    anchor = _json("anchor_second_node.json")["rows"]["L_40"]["second_node_median_s"]
    singles = {
        ("C32", 8): _single("C32", 8, "second-node"),
        ("C64", 10): _single("C64", 10, "main-node"),
        ("C128", 20): _single("C128", 20, "main-node"),
        ("C256", 20): _single("C256", 20, "main-node"),
        ("C256", 40): _single("C256", 40, "main-node"),
        ("L", 40): anchor,
    }
    printed = {
        ("C32", 8): ("0.0957", "0.0974", "0.982"),
        ("C64", 10): ("0.170", "0.178", "0.954"),
        ("C128", 20): ("0.339", "0.346", "0.979"),
        ("C256", 20): ("0.952", "1.013", "0.940"),
        ("C256", 40): ("1.004", "0.981", "1.023"),
        ("L", 40): ("29.08", "28.360", "1.025"),
    }
    for key, (d_s, s_s, r_s) in printed.items():
        grid, threads = key
        dbl = double(grid, threads)
        single = singles[key]
        out.add("Table XVI", f"{grid} {threads} double s", d_s, dbl)
        out.add("Table XVI", f"{grid} {threads} single s", s_s, single)
        out.add("Table XVI", f"{grid} {threads} ratio", r_s,
                None if dbl is None or not single else dbl / single)


def _energy(out: Checks):
    rows = [row for row in _load_csv(RES / "e11_energy.csv")
            if row.get("exit_code") == "0" and row.get("slurm_job_id") == "243067"]
    by_label = {row["label"]: row for row in rows}
    specs = (
        ("E11_E7L_serial", "serial", "1,068", "5,410", None),
        ("E11_E7L_omp40", "OpenMP", "30.1", "4,890", None),
        ("E11_E7L_cuda", "CUDA", "4.18", "16.7", "297"),
    )
    for label, name, dur_s, cpu_s, gpu_s in specs:
        row = by_label[label]
        out.add("Table XVII", f"{name} duration s", dur_s, float(row["duration_s"]))
        out.add("Table XVII", f"{name} CPU J", cpu_s, float(row["energy_cpu_net_j"]))
        if gpu_s is not None:
            out.add("Table XVII", f"{name} GPU J", gpu_s, float(row["energy_gpu_net_j"]))


def _params(out: Checks):
    text = (ROOT / "config" / "default.yaml").read_text()
    values = {}
    for line in text.splitlines():
        if ":" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    out.add("Table II", "alpha", "0.5", float(values["alpha"]))
    out.add("Table II", "depth cutoff m", "0.001", float(values["tau_h"]))
    out.add("Table II", "level cutoff m", "0.00001", float(values["tau_l"]))
    out.add("Table II", "dt min s", "0.01", float(values["dt_min"]))
    out.add("Table II", "dt max s", "60", float(values["dt_max"]))
    out.add("Table II", "U", "1", float(values["U"]))
    out.add("Table II", "K", "4", float(values["K"]))
    out.add("Table II", "open boundary slope", "0.001", float(values["S_bnd"]))
    out.add("Table II", "g", "9.81", float(values["g"]))


def _landcover(out: Checks):
    rows = {row["worldcover_code"]: row for row in _load_csv(ROOT / "config" / "landcover_params.csv")}
    for code, n_s, c_s in LAND:
        row = rows[code]
        out.add("Table III", f"code {code} n", n_s, float(row["manning_n"]))
        out.add("Table III", f"code {code} C", c_s, float(row["runoff_C"]))


def _abstract(out: Checks):
    equiv = _load_csv(RES / "e1_equivalence.csv")
    diffs = []
    for row in equiv:
        for key, value in row.items():
            if "diff" in key.lower() or key.lower().startswith("max_abs"):
                number = _f(row, key)
                if number is not None:
                    diffs.append(abs(number))
    # Fall back to every numeric field whose name contains h, if the header differs.
    if not diffs:
        for row in equiv:
            for key, value in row.items():
                number = _f(row, key)
                if number is not None and "h" in key.lower() and number < 1:
                    diffs.append(abs(number))
    out.add("abstract", "double precision agreement m", "7.37e-14", max(diffs) if diffs else None)
    active_l = None
    for row in BENCH:
        if row["_g"] == "L":
            active_l = _f(row, "active_cells")
            break
    out.add("abstract", "L active cells millions", "4.2",
            None if active_l is None else active_l / 1e6)
    l40 = e4("L", "omp", 40)
    l1 = e4("L", "omp", 1)
    out.add("abstract", "OpenMP self-relative efficiency at 40", "0.87",
            None if not l1 or not l40 else (l1 / l40) / 40)
    mpi1 = median_of(lambda row: row["exp_id"] == "E5" and row["_g"] == "L"
                     and int(float(row["ranks"])) == 1 and row["backend"] == "mpi")
    mpi40 = median_of(lambda row: row["exp_id"] == "E5" and row["_g"] == "L"
                      and int(float(row["ranks"])) == 40 and row["backend"] == "mpi")
    out.add("abstract", "MPI efficiency at 40", "0.86",
            None if not mpi1 or not mpi40 else (mpi1 / mpi40) / 40)
    ratios = _json("derived_ratios.json")
    out.add("abstract", "bandwidth ratio", "3.2", ratios["bandwidth_ratio"])
    out.add("abstract", "FP32 ratio", "2.5", ratios["fp32_ratio"])
    out.add("abstract", "FP64 ratio", "2.4", ratios["fp64_ratio"])
    cuda = median_of(
        lambda row: row["exp_id"] == "E7" and row["backend"] == "cuda" and row["_g"] == "C32"
        and row.get("block") == "16x16" and row.get("reduce_impl") == "cub",
        "wall_total_s",
    )
    out.add("abstract", "kernel step at 1024 cells us", "42",
            None if cuda is None else cuda / 2000.0 * 1e6)
    omp = _best_omp("C32")[0]
    e2e = median_of(
        lambda row: row["exp_id"] == "E7" and row["backend"] == "cuda" and row["_g"] == "C32"
        and row.get("block") == "16x16" and row.get("reduce_impl") == "cub",
        "_e2e",
    )
    out.add("abstract", "lead at 1024 cells", "1.16",
            None if omp is None or not e2e else omp / e2e)
    omp_l, _ = _best_omp("L")
    e2e_l = median_of(
        lambda row: row["exp_id"] == "E7" and row["backend"] == "cuda" and row["_g"] == "L"
        and row.get("block") == "16x16" and row.get("reduce_impl") == "cub",
        "_e2e",
    )
    out.add("abstract", "V100 over best OpenMP", "13.5",
            None if omp_l is None or not e2e_l else omp_l / e2e_l)
    diag = _json("v0_diagnostics.json")
    diag2 = _json("val_4198_diagnostics.json")
    out.add("abstract", "ROC area event 2734 to 2 decimals", "0.79", diag["ROC_AUC"])
    out.add("abstract", "ROC area event 4198 to 2 decimals", "0.79", diag2["ROC_AUC"])
    # The isolated GPU reduction (0.481 s of 2.056 s, 23.4 percent) is not a
    # column in the saved timers. The device t_dt column includes the wait.
    out.add("abstract", "GPU time-step share percent", "23", None,
            "isolated reduction time is not a column in the saved results")
    dt = e4("L", "omp", 40, "t_dt_s")
    out.add("prose", "OpenMP time-step share percent", "32.1",
            None if dt is None or not l40 else 100.0 * dt / l40)
    def mpi_share(ranks):
        wall = median_of(lambda row, ranks=ranks: row["exp_id"] == "E8" and row["backend"] == "mpi"
                         and row["_g"] == "L" and int(float(row["ranks"])) == ranks
                         and int(float(row["U"])) == 1 and row["precision"] == "float")
        step = median_of(lambda row, ranks=ranks: row["exp_id"] == "E8" and row["backend"] == "mpi"
                         and row["_g"] == "L" and int(float(row["ranks"])) == ranks
                         and int(float(row["U"])) == 1 and row["precision"] == "float",
                         "t_dt_s")
        if wall in (None, 0) or step is None:
            return None
        return 100.0 * step / wall
    out.add("prose", "MPI 4 rank time-step share percent", "33.2", mpi_share(4))
    out.add("prose", "MPI 8 rank time-step share percent", "33.6", mpi_share(8))


def _extra(out: Checks):
    align = _json("alignment_report.json")
    align2 = _json("alignment_report_4198.json")
    land = 1211522
    out.add("prose", "event 2734 observed flood percent of land", "0.479",
            100.0 * align["observed_flood_on_land"]["n"] / land,
            "printed as model land in the runs; the matching count is the observed flood")
    out.add("prose", "event 4198 observed flood percent of land", "0.444",
            100.0 * align2["observed_flood_on_land"]["n"] / land,
            "printed as model land in the runs; the matching count is the observed flood")
    out.add("prose", "observed flood median elevation m", "557",
            align["observed_flood_on_land"]["median"],
            "whole-grid median elevation is not stored")
    e12 = _json("e12_summary.json")["main_N500"]
    out.add("prose", "members until land MAD below 0.01", "8",
            e12["010"]["N_first_MAD_below_0.01"])
    out.add("prose", "members until wet MAD below 0.01", "250",
            e12["010"]["N_first_MAD_wet_below_0.01"])
    p0 = 1.0 - e12["010"]["cells_P_gt_0"] / e12["010"]["land_cells"]
    out.add("prose", "land cells with probability 0 at 0.10 m percent", "90.4", 100.0 * p0)
    out.add("prose", "mean probability 0.10 m", "0.0568", e12["010"]["mean_P_land"])
    out.add("prose", "mean probability 0.25 m", "0.0317", e12["025"]["mean_P_land"])
    out.add("prose", "mean probability 0.50 m", "0.0147", e12["050"]["mean_P_land"])
    prec = _json("precision_check.json")
    out.add("prose", "float double max abs hmax m", "0.0183", prec["max_abs_hmax_diff_m"])
    drift = _json("edtdrift.json")
    out.add("prose", "held steps", "1,960", drift["n_reuse_steps"])
    out.add("prose", "held step excess min percent", "0.003",
            100.0 * (drift["min_reused_over_fresh"] - 1.0))
    out.add("prose", "held step excess max percent", "0.66",
            100.0 * (drift["max_reused_over_fresh"] - 1.0))
    out.add("prose", "held step excess p99 percent", "0.59",
            100.0 * (drift["p99_reused_over_fresh"] - 1.0))
    same = _json("evec_samenode.json")
    out.add("prose", "fast-math median s", "28.088", same["fast_math_median_s"])
    out.add("prose", "fast-math ratio", "1.0097", same["ratio_anchor_over_fast"])
    out.add("prose", "fast-math difference percent", "0.97",
            100.0 * (same["ratio_anchor_over_fast"] - 1.0))
    hmax = _json("evec_hmax.json")
    out.add("prose", "fast-math max abs hmax m", "0.000230", hmax["max_abs_hmax_diff_m"])
    aniso = {row["K"]: row for row in _load_csv(RES / "e9_anisotropy.csv")}
    out.add("prose", "K=4 anisotropy ratio", "1.018", float(aniso["4"]["ratio_diag_over_axis"]))
    out.add("prose", "K=8 anisotropy ratio", "0.986", float(aniso["8"]["ratio_diag_over_axis"]))
    out.add("prose", "K=4 wet cells", "17,765", float(aniso["4"]["wet_count"]))
    out.add("prose", "K=8 wet cells", "17,893", float(aniso["8"]["wet_count"]))
    # Mass maxima in double precision.
    masses = {"serial": [], "omp": [], "mpi": [], "cuda": []}
    for row in _load_csv(RES / "bench.csv"):
        if row.get("precision") != "double":
            continue
        value = _f(row, "mass_err_rel")
        if value is None:
            continue
        backend = row.get("backend")
        if backend in masses:
            masses[backend].append(abs(value))
    printed_mass = {
        "serial": "6.63e-14",
        "omp": "6.56e-15",
        "mpi": "6.91e-15",
        "cuda": "2.77e-13",
    }
    for backend, expected in printed_mass.items():
        raw = max(masses[backend]) if masses[backend] else None
        out.add("prose", f"{backend} mass error maximum", expected, raw)
    # K=8 over K=4.
    def kwall(backend, neigh, **extra):
        return median_of(
            lambda row, backend=backend, neigh=neigh, extra=extra: row["exp_id"] == "E9"
            and row["backend"] == backend and row["_g"] == "L" and row["precision"] == "float"
            and row["neigh"] == str(neigh)
            and all(str(row.get(key)) == str(value) for key, value in extra.items())
        )
    pairs = (
        ("OpenMP", "omp", {"threads": "40", "halo_mode": ""}, "1.70"),
        ("CUDA", "cuda", {"block": "16x16"}, "1.41"),
        ("MPI", "mpi", {"ranks": "4", "halo_mode": "sendrecv2"}, "1.70"),
    )
    for name, backend, extra, expected in pairs:
        a = kwall(backend, 4, **extra)
        b = kwall(backend, 8, **extra)
        out.add("prose", f"{name} K=8 over K=4", expected, None if not a or not b else b / a)


def write_expected(path: Path | None = None):
    path = path or (ROOT / "analysis" / "expected_values.csv")
    rows = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_item", "quantity", "expected", "note"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "paper_item": row["paper_item"],
                "quantity": row["quantity"],
                "expected": row["expected"],
                "note": row["note"],
            })
    return path


def evaluate():
    results = []
    for row in build():
        if row["raw"] is None:
            got = "MISSING"
            status = "FAIL"
        else:
            got = fmt_like(row["raw"], row["expected"])
            status = "PASS" if got == row["expected"] else "FAIL"
        results.append({**row, "got": got, "status": status})
    return results


if __name__ == "__main__":
    for item in evaluate():
        if item["status"] != "PASS":
            print(f"FAIL {item['paper_item']} {item['quantity']}: expected {item['expected']} got {item['got']} {item['note']}")
