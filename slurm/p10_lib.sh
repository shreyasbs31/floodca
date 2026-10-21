#!/bin/bash
# Shared by the P-10 H1 benchmark scripts. Sourced after `cd ${SLURM_SUBMIT_DIR}`.
# Does not modify bench/p7_lib.sh: the queued P-7 jobs source that file at runtime.
source slurm/p7_lib.sh

# p10_build <dir> <double ON|OFF> <cuda ON|OFF> <mpi ON|OFF>
p10_build() {
  local dir=$1 dbl=$2 cuda=$3 mpi=$4
  cmake -S . -B "${dir}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE="${dbl}" -DFCA_WITH_OMP=ON \
    -DFCA_WITH_MPI="${mpi}" -DFCA_WITH_CUDA="${cuda}" -DFCA_BUILD_TESTS=OFF \
    > "${dir}.cmake.log" 2>&1 || { cat "${dir}.cmake.log"; return 1; }
  cmake --build "${dir}" -j "$(nproc)" > "${dir}.build.log" 2>&1 \
    || { tail -50 "${dir}.build.log"; return 1; }
  echo "built ${dir} (double=${dbl}, cuda=${cuda}, mpi=${mpi})"
}

# p10_run <exp> <grid> <np> <exe> [extra args...]
# np=0 runs the binary directly (serial, OpenMP, CUDA). np>0 uses mpirun --bind-to core.
# Extra arguments are appended, so a later --U or --neigh overrides the defaults.
# Protocol is p7_run: 1 discarded warm-up + P7_REPEATS timed repeats, B0 warm start.
p10_run() {
  local exp=$1 grid=$2 np=$3 exe=$4
  shift 4
  local args=(--grid "${WS}/grid_${grid}" --rain "${PROC}/rain/${grid}/rain_${B0_DATE}.bin"
              --warmstart "${WS}/h0_B0_${grid}.bin" --lambda 1 --D_h 24 --phi 0.5
              --hyeto triangle --steps "${P7_STEPS}" --neigh "${P7_K:-4}" --U 1
              --no-output --csv "${CSV}" --exp "${exp}")
  p7_precheck
  local r
  for r in $(seq 0 "${P7_REPEATS}"); do
    export FCA_REPEAT="${r}" FCA_WARMUP=$([[ ${r} == 0 ]] && echo 1 || echo 0)
    if [[ "${np}" -gt 0 ]]; then
      mpirun -np "${np}" --bind-to core "${exe}" "${args[@]}" "$@"
    else
      "${exe}" "${args[@]}" "$@"
    fi | grep -E "^done" | sed "s/^/${exp} ${grid} np${np} r${r} /"
  done
  unset FCA_REPEAT FCA_WARMUP
}
