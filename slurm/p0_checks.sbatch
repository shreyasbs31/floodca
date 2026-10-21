#!/bin/bash
# P-0 follow-up checks on a GPU node: both V100s, OpenMPI transports, RAPL.
#SBATCH --job-name=floodca-p0chk
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_GPU__
#SBATCH --exclude=__FCA_NODE_SECOND__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/slurm
# shellcheck source=/dev/null
source slurm/env.sh
# shellcheck disable=SC1091
source slurm/gpu_driver_preflight.sh
gpu_driver_preflight

echo "===== job ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
hostname
date -u

echo "===== nvidia-smi -L ====="
nvidia-smi -L
echo "===== nvidia-smi ====="
nvidia-smi

echo "===== RAPL ====="
ls -l /sys/class/powercap/intel-rapl* 2>&1 | head
for f in /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl:*:*/energy_uj; do
    if [ -e "$f" ]; then
        echo -n "$f readable="; test -r "$f" && echo yes || echo no
        echo -n "$f value="; cat "$f" 2>&1
    fi
done

echo "===== ompi_info transports ====="
ompi_info | grep -i -E "ucx|openib|btl:|pml:|ofi|verbs|mtl" || true
echo "===== ompi_info pml ====="
ompi_info --param pml all --level 9 2>/dev/null | grep -i -E "ucx|ob1|cm" | head -n 40 || true
echo "===== MCA pml components ====="
ompi_info -a 2>/dev/null | grep -i -E "MCA pml" | head -n 30 || true

echo "===== try OMPI_MCA_pml=ucx (compile+run 1 rank) ====="
cat > /tmp/floodca_ucx_hello.$$.c <<'EOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int r=0,s=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&r);
    MPI_Comm_size(MPI_COMM_WORLD,&s);
    char ver[MPI_MAX_LIBRARY_VERSION_STRING];
    int n=0; MPI_Get_library_version(ver,&n);
    printf("ucx_probe rank %d/%d lib=%s\n", r, s, ver);
    MPI_Finalize();
    return 0;
}
EOF
mpicc -O2 /tmp/floodca_ucx_hello.$$.c -o /tmp/floodca_ucx_hello.$$
set +e
OMPI_MCA_pml=ucx mpirun -np 1 /tmp/floodca_ucx_hello.$$
echo "ucx_exit=$?"
set -e
rm -f /tmp/floodca_ucx_hello.$$ /tmp/floodca_ucx_hello.$$.c

echo "P0CHK_OK"
