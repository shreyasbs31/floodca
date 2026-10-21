#!/bin/bash
#SBATCH --job-name=floodca-p4
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export FCA_GIT="$(git rev-parse HEAD 2>/dev/null || true)"
mkdir -p build results/slurm results/e1_257x263

echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) nlist=${SLURM_JOB_NODELIST}"
echo "CC=$CC cmake=$(cmake --version | head -n1) OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CC}" \
  -DCMAKE_C_FLAGS="-ffp-contract=off -fno-fast-math" \
  -DFCA_DOUBLE=ON \
  -DFCA_WITH_OMP=ON \
  -DFCA_WITH_MPI=ON \
  -DFCA_WITH_CUDA=ON \
  -DFCA_BUILD_TESTS=ON
rm -f build/test_physics build/CMakeFiles/test_physics.dir/tests/c/test_physics.c.o
cmake --build build -j "${SLURM_CPUS_PER_TASK}" \
  --target test_physics test_mass test_lake_at_rest floodca_serial floodca_omp

echo "===== test_physics ====="
./build/test_physics
echo "===== test_mass ====="
./build/test_mass
echo "===== test_lake_at_rest ====="
./build/test_lake_at_rest

echo "===== E1 257x263 ====="
export BUILD="${SLURM_SUBMIT_DIR}/build"
export E1_THREADS="${SLURM_CPUS_PER_TASK}"
bash tests/c/test_equivalence.sh

echo "===== E1 grid M (1000 steps, threads 1 and ${SLURM_CPUS_PER_TASK}) ====="
MDIR=data/processed/udupi_coast/M
MOUT=results/e1_gridM
python - <<'PY'
import json, os, struct
d = "data/processed/udupi_coast/M"
meta = json.load(open(os.path.join(d, "dem.json")))
n = int(meta["nrows"]) * int(meta["ncols"])
mask = open(os.path.join(d, "mask.bin"), "rb").read()
h = [0.08 if (mask[i] == 0) else 0.0 for i in range(n)]
open(os.path.join(d, "h.bin"), "wb").write(struct.pack("<" + "f" * n, *h))
print("wrote", d + "/h.bin", "n", n)
PY
mkdir -p "${MOUT}/serial" "${MOUT}/omp"
./build/floodca_serial --grid "${MDIR}" --steps 1000 --neigh 4 --U 1 \
    --out "${MOUT}/serial" --exp E1
./build/floodca_omp --grid "${MDIR}" --steps 1000 --neigh 4 --U 1 --threads "${SLURM_CPUS_PER_TASK}" \
    --out "${MOUT}/omp" --exp E1
python tests/py/compare_hmax.py "${MOUT}/serial/hmax.bin" "${MOUT}/omp/hmax.bin" 1e-10
echo "P4_OMP_OK"
