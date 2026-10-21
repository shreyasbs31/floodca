#!/bin/bash
#SBATCH --job-name=floodca-p7e7
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E7 (SPEC 11.3): CPU-GPU crossover on one exclusive H1 node, float, K=4. Serial,
# OpenMP at ${E7_OMP_THREADS} threads (best of these is "best OpenMP"; S, M, L use
# the full E4 sweep), and CUDA 16x16 + CUB. CUDA "kernel only" is
# wall_total (step loop, including the per-step dt readback); "end-to-end" adds
# t_h2d and t_d2h. Serial and OpenMP rows on S, M, L come from E4 (same node type,
# same configuration), so E7_CPU_GRIDS lists only the sizes E4 does not cover.
cd "${SLURM_SUBMIT_DIR}"
source slurm/p7_lib.sh
export CUDA_VISIBLE_DEVICES=0

CPU_GRIDS="${E7_CPU_GRIDS:-C128 C256 C768 C1536}"
GPU_GRIDS="${E7_GPU_GRIDS:-C128 C256 S C768 M C1536 L}"
OMP_T="${E7_OMP_THREADS:-10 20 40}"
BD="build/p7cross_${SLURM_JOB_ID}"
p7_build "${BD}" OFF ON
for g in ${GPU_GRIDS}; do
  p7_need_grid "${g}"
  p7_run E7 "${g}" "${BD}/floodca_cuda" --block 16x16 --reduce cub
done
for g in ${CPU_GRIDS}; do
  for t in ${OMP_T}; do
    p7_run E7 "${g}" "${BD}/floodca_omp" --threads "${t}"
  done
  p7_run E7 "${g}" "${BD}/floodca_serial"
done
echo "P7_E7_OK"
