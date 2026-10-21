#!/bin/bash
#SBATCH --job-name=floodca-e10cpu
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E10 throughput, CPU half. No GPUs are requested.
# B5 master-worker, N=8, at 1x40, 2x20, 4x10 and 8x5 (ranks x threads).
# One serial scenario, not eight.
# 40 Slurm tasks, one CPU each, so Open MPI can launch up to 8 ranks.
# Job 243816 had one task: 1x40 finished (605.093 s) and 2 ranks were refused.
# That 1x40 row is kept. Later configs are not repeated if they are already in the csv.
#
# Estimate before this script was submitted, from executed logs, not from this job:
#   Job 243061, double, Platinum 8268, 8 scenarios, 90 m, 24 h:
#     1x48 wall 409.54046 s (51.2 s/scenario), 4x12 wall 510.2057 s, 3x16 wall 477.77947 s.
#   Planning rate on the Gold 6248 in float: 70 s/scenario at 40 threads
#   (above that 51 s measurement: fewer cores, lower clock, precision not remeasured).
#   N=8: 1x40 about 560 s; each split config about 1.3 times that (the 4-rank
#   double run was 1.25 times), so four configs about 46 min.
#   One serial scenario about 70 * (1068.584/28.019) = 44 min.
#   Build about 5 min. Total about 96 min, inside a 2.5 h limit.
#
# If the CPU model is not the Gold 6248, rows are marked H2 and are not for
# GPU-node ratios. This job does not source a GPU preflight and does not call nvidia-smi.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONUNBUFFERED=1
export OMP_PROC_BIND=false OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"

MODEL=$(awk -F: '/model name/ {gsub(/^ /,"",$2); print $2; exit}' /proc/cpuinfo)
if [[ "${MODEL}" == *"6248"* ]]; then
  NODE_TYPE=H1
else
  NODE_TYPE=H2
fi
echo "E10_CPU_NODE model=${MODEL} node_type=${NODE_TYPE} partition=${SLURM_JOB_PARTITION} host=$(hostname)"
if [[ "${NODE_TYPE}" != H1 ]]; then
  echo "E10_CPU_NODE_H2 do not mix these walls into GPU-node ratios"
fi

BD="build/p10e10cpu_${SLURM_JOB_ID}"
cmake -S . -B "${BD}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF -DFCA_WITH_OMP=ON \
  -DFCA_WITH_MPI=ON -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF > "${BD}.cmake.log" 2>&1
cmake --build "${BD}" -j "$(nproc)" --target floodca_ens_mpi_omp --target floodca_serial \
  > "${BD}.build.log" 2>&1

PROC=data/processed/udupi_coast
OUT=results/e10_cpu
mkdir -p "${OUT}"
SUM=results/e10_cpu.csv
if [[ ! -s "${SUM}" ]]; then
  echo "config,node_type,cpu_model,scenarios,ranks,threads,wall_s,scen_per_hour,slurm_job_id" > "${SUM}"
fi
# H2 rows stay out of results/bench.csv so GPU-node summaries do not absorb them.
BENCH_CSV=""
if [[ "${NODE_TYPE}" == H1 ]]; then
  BENCH_CSV=results/bench.csv
fi

run_b5() {
  local tag=$1 np=$2 thr=$3
  if awk -F, -v tag="${tag}" 'NR > 1 && $1 == tag { found = 1 } END { exit !found }' "${SUM}"; then
    echo "skip ${tag}: already in ${SUM}"
    return 0
  fi
  echo "===== B5 ${tag}: ${np} x ${thr} N=8 ====="
  local extra=()
  if [[ -n "${BENCH_CSV}" ]]; then
    extra=(--csv "${BENCH_CSV}" --exp E10)
  fi
  OMP_NUM_THREADS="${thr}" mpirun -np "${np}" --bind-to none -x OMP_NUM_THREADS="${thr}" \
    -x OMP_PROC_BIND=false \
    "${BD}/floodca_ens_mpi_omp" \
    --grid "${PROC}/90" --scenario-file "${PROC}/scenarios_N500.csv" --rain-dir "${PROC}/rain/90" \
    --nscen 8 --tsim 86400 --neigh 4 --U 1 --threads "${thr}" --master-computes 1 \
    --poll-every 50 --no-output --out "${OUT}/${tag}" --ens-csv "${OUT}/${tag}.csv" \
    "${extra[@]}" | tee "${OUT}/${tag}.log"
  awk -v tag="${tag}" -v np="${np}" -v thr="${thr}" -v nt="${NODE_TYPE}" -v model="${MODEL}" \
      -v job="${SLURM_JOB_ID}" '
    /done backend=ens_mpi_omp/ {
      wall=""; rate=""; scen=""
      for (i = 1; i <= NF; i++) {
        split($i, a, "=")
        if (a[1] == "wall") wall=a[2]
        if (a[1] == "scen_per_hour") rate=a[2]
        if (a[1] == "scenarios") scen=a[2]
      }
      print tag "," nt "," model "," scen "," np "," thr "," wall "," rate "," job
    }
  ' "${OUT}/${tag}.log" >> "${SUM}"
}

run_b5 b5_1x40 1 40
run_b5 b5_2x20 2 20
run_b5 b5_4x10 4 10
run_b5 b5_8x5 8 5

echo "===== serial N=1 ====="
read -r sid date lam dh phi < <(awk -F, 'NR==2 {print $1, $2, $3, $4, $5}' "${PROC}/scenarios_N500.csv")
serial_args=(--grid "${PROC}/90" --rain "${PROC}/rain/90/rain_${date}.bin"
  --lambda "${lam}" --D_h "${dh}" --phi "${phi}" --hyeto triangle --tsim 86400
  --neigh 4 --U 1 --threads 1 --no-output)
if [[ -n "${BENCH_CSV}" ]]; then
  serial_args+=(--csv "${BENCH_CSV}" --exp E10)
fi
"${BD}/floodca_serial" "${serial_args[@]}" | tee "${OUT}/serial.log"
awk -v nt="${NODE_TYPE}" -v model="${MODEL}" -v job="${SLURM_JOB_ID}" '
  /done backend=serial/ {
    wall=""
    for (i = 1; i <= NF; i++) {
      split($i, a, "=")
      if (a[1] == "wall") wall=a[2]
    }
    rate = 3600.0 / wall
    print "serial," nt "," model ",1,1,1," wall "," rate "," job
  }
' "${OUT}/serial.log" >> "${SUM}"
echo "P10_E10_CPU_OK"
