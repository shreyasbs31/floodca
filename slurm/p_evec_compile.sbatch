#!/bin/bash
#SBATCH --job-name=floodca-eveccmp
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Compiler report only. No timing. CMakeLists keeps -O3 -march=native in the
# Release flags; the extra flags below are appended. On a Cascade Lake CPU
# node, native is the same ISA family as the Gold 6248. This is not a wall clock.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
OUT=results/evec
mkdir -p "${OUT}"
echo "host $(hostname) gcc $(gcc -dumpversion)"
grep -m1 "model name" /proc/cpuinfo || true
echo "libmvec:"; ldconfig -p 2>/dev/null | grep libmvec || ls /lib64/libmvec* /usr/lib64/libmvec* 2>/dev/null || echo "NO_LIBMVEC"

build_one() {
  local tag="$1" flags="$2"
  local bd="build/eveccmp_${tag}_${SLURM_JOB_ID}"
  local miss="${SLURM_SUBMIT_DIR}/${OUT}/vec_missed_${tag}.txt"
  rm -f "${miss}"
  cmake -S . -B "${bd}" -DCMAKE_BUILD_TYPE=Release -DFCA_DOUBLE=OFF \
    -DFCA_WITH_OMP=ON -DFCA_WITH_MPI=OFF -DFCA_WITH_CUDA=OFF -DFCA_BUILD_TESTS=OFF \
    -DCMAKE_C_FLAGS="${flags} -fopt-info-vec-missed=${miss}"
  echo "FLAGS ${tag}: $(grep '^C_FLAGS' "${bd}/CMakeFiles/floodca_core_omp.dir/flags.make")"
  cmake --build "${bd}" -j "${SLURM_CPUS_PER_TASK}" --target floodca_omp
  if [[ -f "${miss}" ]]; then
    echo "WROTE ${miss} bytes $(wc -c < "${miss}")"
  else
    echo "MISSING ${miss}"
  fi
}

build_one standard ""
build_one fastmath "-ffast-math -fopenmp-simd"
echo "EVEC_COMPILE_OK"
