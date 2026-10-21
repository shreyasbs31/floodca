#!/bin/bash
# Shared by bench/p7_*.sbatch (SPEC Sec 9.3, 11.1). Sourced after `cd ${SLURM_SUBMIT_DIR}`.
# Builds production binaries on the benchmark node, then runs each configuration as
# 1 discarded warm-up + 5 timed repeats of B0 from its warm start, 2000 steps, U=1,
# --no-output, one CSV row per run in ${CSV}.
set -euo pipefail
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=close OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
export FCA_NODE_TYPE="H1"
mkdir -p results/slurm

CSV="${CSV:-${SLURM_SUBMIT_DIR}/results/bench.csv}"
P7_STEPS="${P7_STEPS:-2000}"
P7_REPEATS="${P7_REPEATS:-5}"
PROC=data/processed/udupi_coast
WS="${PROC}/warmstart"
B0_DATE=$(python -c "import json;print(json.load(open('${PROC}/scenarios_N500_report.json'))['b0']['base_date'])")

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) partition=${SLURM_JOB_PARTITION} nodes=${SLURM_JOB_NODELIST}"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node[0-9]"
nvidia-smi --query-gpu=index,name,driver_version,utilization.gpu,memory.used --format=csv 2>/dev/null || true

# p7_build <dir> <double ON|OFF> <cuda ON|OFF>
p7_build() {
  local dir=$1 dbl=$2 cuda=$3
  cmake -S . -B "${dir}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE="${dbl}" -DFCA_WITH_OMP=ON \
    -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA="${cuda}" -DFCA_BUILD_TESTS=OFF > "${dir}.cmake.log" 2>&1 \
    || { cat "${dir}.cmake.log"; return 1; }
  cmake --build "${dir}" -j "$(nproc)" > "${dir}.build.log" 2>&1 || { tail -50 "${dir}.build.log"; return 1; }
  echo "built ${dir} (double=${dbl}, cuda=${cuda})"
}

# Co-tenancy check before each configuration (SPEC 9.3). The 1-minute load average is
# recorded but includes our own build and previous runs, so contamination is decided
# by CPU use of other users' processes (> 5% each) and by GPU utilisation while none
# of our GPU work is running.
p7_precheck() {
  export FCA_LOAD_AVG=$(cut -d' ' -f1 /proc/loadavg)
  local other="" busy
  if command -v nvidia-smi > /dev/null 2>&1; then
    # __FCA_NODE_SECOND__ has no usable device for a CPU-only job. nvidia-smi then exits 6
    # ("No devices were found"), which would abort the run under pipefail.
    other=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | sort -n | tail -1 | tr -d ' ' || true)
  fi
  export FCA_GPU_OTHER_UTIL="${other}"
  busy=$(ps -eo user:32,pcpu --no-headers \
    | awk -v me="${USER}" '$1 != me && $1 != "root" && $2 > 5 {n++} END {print n + 0}')
  export FCA_CONTAMINATED=0
  if [[ "${busy}" -gt 0 ]]; then
    FCA_CONTAMINATED=1
    echo "precheck: ${busy} busy processes of other users"
  fi
  if [[ "${other}" =~ ^[0-9]+$ && "${other}" -gt 5 ]]; then
    FCA_CONTAMINATED=1
    echo "precheck: GPU utilisation ${other}% before our run"
  fi
}

# p7_run <exp> <grid> <binary> [extra args...]: warm-up + repeats on one grid.
p7_run() {
  local exp=$1 grid=$2 exe=$3
  shift 3
  local args=(--grid "${WS}/grid_${grid}" --rain "${PROC}/rain/${grid}/rain_${B0_DATE}.bin"
              --warmstart "${WS}/h0_B0_${grid}.bin" --lambda 1 --D_h 24 --phi 0.5
              --hyeto triangle --steps "${P7_STEPS}" --neigh "${P7_K:-4}" --U 1
              --no-output --csv "${CSV}" --exp "${exp}")
  p7_precheck
  local r
  for r in $(seq 0 "${P7_REPEATS}"); do
    export FCA_REPEAT="${r}" FCA_WARMUP=$([[ ${r} == 0 ]] && echo 1 || echo 0)
    "${exe}" "${args[@]}" "$@" | grep -E "^done" | sed "s/^/${exp} ${grid} r${r} /"
  done
  unset FCA_REPEAT FCA_WARMUP
}

# p7_need_grid <grid>: dry symlink dir from floodca_prep.warmstart must exist.
p7_need_grid() {
  local g=$1
  [[ -f "${WS}/grid_${g}/dem.bin" && -f "${WS}/h0_B0_${g}.bin" && -f "${PROC}/rain/${g}/rain_${B0_DATE}.bin" ]] \
    || { echo "missing inputs for grid ${g}"; return 1; }
}
