#!/bin/bash
#SBATCH --job-name=floodca-p7e6
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# E6 (SPEC 11.3): CUDA performance on one V100 of an exclusive H1 node, grids
# ${E6_GRIDS}, blocks 8/16/32, float vs double, CUB vs custom reduction. Production
# binary (FMA allowed) for the timed repeats; the CUDA-event binary
# (floodca_cuda_timing) runs the same configurations once more for the F9 breakdown.
cd "${SLURM_SUBMIT_DIR}"
source slurm/p7_lib.sh
export CUDA_VISIBLE_DEVICES=0

GRIDS="${E6_GRIDS:-S M L}"
BLOCKS="${E6_BLOCKS:-8x8 16x16 32x32}"
REDUCES="${E6_REDUCES:-cub custom}"
BF="build/p7cudaf_${SLURM_JOB_ID}"
BDB="build/p7cudad_${SLURM_JOB_ID}"
p7_build "${BF}" OFF ON
p7_build "${BDB}" ON ON
for g in ${GRIDS}; do
  p7_need_grid "${g}"
  for bd in "${BF}" "${BDB}"; do
    for b in ${BLOCKS}; do
      for red in ${REDUCES}; do
        p7_run E6 "${g}" "${bd}/floodca_cuda" --block "${b}" --reduce "${red}"
        P7_REPEATS=1
        export FCA_NOTES=cuda_event_timing
        p7_run E6t "${g}" "${bd}/floodca_cuda_timing" --block "${b}" --reduce "${red}"
        P7_REPEATS="${E6_REPEATS:-5}"
        unset FCA_NOTES
      done
    done
  done
done
echo "P7_E6_OK grids=${GRIDS}"
