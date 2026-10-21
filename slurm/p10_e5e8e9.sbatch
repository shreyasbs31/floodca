#!/bin/bash
#SBATCH --job-name=floodca-p10h1
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Packed H1 job (SPEC 11.3), one exclusive Gold 6248 node, float, B0, 2000 steps,
# 1 warm-up + 5 repeats unless noted:
#   E5 strong scaling on L, P in {1,2,4,8,16,32,40}. XL is deferred.
#   E5 weak scaling on W1,W2,W4,W8,W16 (one GPU-partition node; multi-node E5 is out of scope).
#   E8 timing on L: OpenMP 40 threads, MPI P=4 and P=8, CUDA; U in {1,5,10,25,50}.
#   E9: K=8 vs K=4 on L for OpenMP, MPI P=4 (sendrecv2 and the K=4 minimal halo), CUDA.
#   E11: one instrumented run of the E7 grid-L serial, OpenMP-40 and CUDA configurations.
# The SPEC 10.4 anisotropy test is not run: the test-only point source is not implemented,
# and src/kernels/physics.h is not modified in this job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/p10_lib.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p results
cat > results/e9_anisotropy.txt <<'EOF'
E9 anisotropy test (SPEC 10.4) was not run in job floodca-p10h1.
The test needs a point source that is compiled only into test builds. That term is not in the code, and the physics was not changed.
K=4 versus K=8 timing on grid L (OpenMP, MPI, CUDA) is the E9 result from this job.
EOF

BD="build/p10h1_${SLURM_JOB_ID}"
p10_build "${BD}" OFF ON ON
SER="${BD}/floodca_serial"
OMP="${BD}/floodca_omp"
MPI="${BD}/floodca_mpi"
CUDA="${BD}/floodca_cuda"

echo "===== E5 strong, grid L ====="
p7_need_grid L
for p in 1 2 4 8 16 32 40; do
  p10_run E5 L "${p}" "${MPI}"
done

echo "===== E5 weak, W<P> ====="
for p in 1 2 4 8 16; do
  p7_need_grid "W${p}"
  p10_run E5 "W${p}" "${p}" "${MPI}"
done

echo "===== E8, grid L, U sweep ====="
p7_need_grid L
for u in 1 5 10 25 50; do
  p10_run E8 L 0 "${OMP}" --threads 40 --U "${u}"
  p10_run E8 L 4 "${MPI}" --U "${u}"
  p10_run E8 L 8 "${MPI}" --U "${u}"
  p10_run E8 L 0 "${CUDA}" --block 16x16 --reduce cub --U "${u}"
done

echo "===== E9, K=4 and K=8 ====="
for k in 4 8; do
  p10_run E9 L 0 "${OMP}" --threads 40 --neigh "${k}"
  p10_run E9 L 4 "${MPI}" --neigh "${k}" --halo sendrecv2
  p10_run E9 L 0 "${CUDA}" --block 16x16 --reduce cub --neigh "${k}"
done
p10_run E9 L 4 "${MPI}" --neigh 4 --halo minimal

echo "===== E11 instrumented E7-L runs (one each, after an unmeasured warm-up) ====="
stampdir=results/e11_stamps
mkdir -p "${stampdir}"
e11_one() {
  local label=$1
  shift
  if [[ -f "${stampdir}/${label}.ok" ]]; then
    echo "skip ${label}"
    return 0
  fi
  "$@" --exp E11_warmup || true
  python scripts/energy_sampler.py --label "${label}" --exp E11 --idle 30 --scenarios 1 \
    --csv results/e11_energy.csv -- "$@" --exp E11
  touch "${stampdir}/${label}.ok"
}
base=(--grid "${WS}/grid_L" --rain "${PROC}/rain/L/rain_${B0_DATE}.bin"
      --warmstart "${WS}/h0_B0_L.bin" --lambda 1 --D_h 24 --phi 0.5 --hyeto triangle
      --steps 2000 --neigh 4 --U 1 --no-output --csv "${CSV}")
e11_one E11_E7L_serial "${SER}" "${base[@]}"
e11_one E11_E7L_omp40 "${OMP}" "${base[@]}" --threads 40
e11_one E11_E7L_cuda "${CUDA}" "${base[@]}" --block 16x16 --reduce cub
echo "P10_H1_OK"
