"""Energy sampler (SPEC Sec 9.2) wrapped around one command.

    energy_sampler.py --label LABEL --exp E11 --csv results/e11_energy.csv \
        [--idle 30] [--scenarios N] [--cells-steps X] -- <command> [args...]

1. Idle baseline: RAPL package energy and nvidia-smi power over --idle seconds.
2. Runs the command; RAPL energy_uj of every top-level package is read before and
   after (counter wrap handled with max_energy_range_uj), and nvidia-smi samples
   timestamp,index,power.draw every 100 ms in the background; GPU energy is the
   trapezoidal integral over the run window, summed over all GPUs of the node.
3. Appends one row: gross and net (minus idle power x duration) CPU and GPU energy,
   energy per scenario and per million cell updates when given, and EDP.
CPU energy is "n/a" when RAPL is unreadable; GPU columns are empty without GPUs.
The command's stdout and stderr pass through; the exit code is the command's.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp", "slurm_job_id", "host", "exp_id", "label", "duration_s", "idle_s",
          "cpu_idle_w", "gpu_idle_w", "energy_cpu_gross_j", "energy_cpu_net_j", "energy_gpu_gross_j",
          "energy_gpu_net_j", "energy_total_net_j", "scenarios", "j_per_scenario_net",
          "cell_steps", "j_per_mcu_net", "edp_net_js", "gpu_samples", "exit_code", "command"]


def _rapl_domains() -> list[str]:
    return sorted(d for d in glob.glob("/sys/class/powercap/intel-rapl:*") if d.count(":") == 1)


def _rapl_read(domains: list[str]) -> list[int] | None:
    try:
        return [int(Path(d, "energy_uj").read_text()) for d in domains]
    except (OSError, ValueError):
        return None


def _rapl_delta_j(domains: list[str], a: list[int] | None, b: list[int] | None) -> float | None:
    if a is None or b is None or not domains:
        return None
    tot = 0
    for d, x, y in zip(domains, a, b):
        if y < x:
            y += int(Path(d, "max_energy_range_uj").read_text())
        tot += y - x
    return tot * 1e-6


class GpuSampler:
    def __init__(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(prefix="gpu_power_", suffix=".csv", delete=False)
        self.proc = None
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,index,power.draw", "--format=csv,noheader,nounits",
                 "-lms", "100"], stdout=self.tmp, stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None

    def stop(self) -> list[tuple[float, int, float]]:
        if self.proc is None:
            return []
        self.proc.terminate()
        self.proc.wait()
        self.tmp.close()
        out = []
        for line in Path(self.tmp.name).read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                t = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out.append((t, int(parts[1]), float(parts[2])))
            except ValueError:
                continue
        os.unlink(self.tmp.name)
        return out


def _gpu_energy(samples: list[tuple[float, int, float]], t0: float, t1: float) -> float | None:
    if not samples:
        return None
    tot = 0.0
    for idx in sorted({s[1] for s in samples}):
        pts = sorted((t, p) for t, i, p in samples if i == idx and t0 <= t <= t1)
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            tot += 0.5 * (pa + pb) * (tb - ta)
    return tot


def _measure(cmd: list[str] | None, idle: float):
    dom = _rapl_domains()
    gs = GpuSampler()
    time.sleep(0.5)
    r0, t0 = _rapl_read(dom), time.time()
    rc = 0
    if cmd is None:
        time.sleep(idle)
    else:
        rc = subprocess.call(cmd)
    r1, t1 = _rapl_read(dom), time.time()
    time.sleep(0.3)
    samples = gs.stop()
    return t1 - t0, _rapl_delta_j(dom, r0, r1), _gpu_energy(samples, t0, t1), len(samples), rc


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        print(__doc__)
        return 2
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--exp", default="E11")
    ap.add_argument("--csv", default="results/e11_energy.csv")
    ap.add_argument("--idle", type=float, default=30.0)
    ap.add_argument("--scenarios", type=int, default=0)
    ap.add_argument("--cells-steps", type=float, default=0.0)
    a = ap.parse_args(argv[:cut])
    cmd = argv[cut + 1:]

    idle_s, e_cpu_idle, e_gpu_idle, _, _ = _measure(None, a.idle)
    dur, e_cpu, e_gpu, ns, rc = _measure(cmd, 0.0)
    cpu_idle_w = e_cpu_idle / idle_s if e_cpu_idle is not None else None
    gpu_idle_w = e_gpu_idle / idle_s if e_gpu_idle is not None else None
    cpu_net = e_cpu - cpu_idle_w * dur if e_cpu is not None and cpu_idle_w is not None else None
    gpu_net = e_gpu - gpu_idle_w * dur if e_gpu is not None and gpu_idle_w is not None else None
    tot = sum(v for v in (cpu_net, gpu_net) if v is not None) if (cpu_net, gpu_net) != (None, None) else None

    def f(v):
        return "n/a" if v is None else f"{v:.6g}"

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "host": socket.gethostname(),
           "exp_id": a.exp, "label": a.label, "duration_s": f"{dur:.3f}", "idle_s": f"{idle_s:.3f}",
           "cpu_idle_w": f(cpu_idle_w), "gpu_idle_w": "" if e_gpu is None else f(gpu_idle_w),
           "energy_cpu_gross_j": f(e_cpu), "energy_cpu_net_j": f(cpu_net),
           "energy_gpu_gross_j": "" if e_gpu is None else f(e_gpu),
           "energy_gpu_net_j": "" if e_gpu is None else f(gpu_net), "energy_total_net_j": f(tot),
           "scenarios": a.scenarios or "",
           "j_per_scenario_net": f(tot / a.scenarios) if a.scenarios and tot is not None else "",
           "cell_steps": f"{a.cells_steps:.6g}" if a.cells_steps else "",
           "j_per_mcu_net": f(tot / (a.cells_steps / 1e6)) if a.cells_steps and tot is not None else "",
           "edp_net_js": f(tot * dur) if tot is not None else "", "gpu_samples": ns, "exit_code": rc,
           "command": " ".join(cmd)}
    new = not Path(a.csv).is_file()
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"energy label={a.label} duration={dur:.1f}s cpu_net={f(cpu_net)}J gpu_net={f(gpu_net)}J", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
