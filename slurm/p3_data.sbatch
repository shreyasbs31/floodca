#!/bin/bash
#SBATCH --job-name=floodca-p3data
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/${USER}_mplconfig"
mkdir -p results/slurm paper/shared/figures "${MPLCONFIGDIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} host=$(hostname) python=$(python --version)"

echo "===== landcover ====="
python -m floodca_prep.landcover \
  --worldcover data/raw/gee/floodca_worldcover_v200_udupi.tif \
  --params config/landcover_params.csv \
  --out data/processed/udupi_coast \
  --defaults config/default.yaml

echo "===== GFD 2734 validation (jrc_perm_water band) ====="
python -m floodca_prep.validation_data \
  --gfd data/raw/gee/floodca_gfd_udupi_2734.tif \
  --out data/processed/udupi_coast \
  --event-id 2734

echo "===== Sec 10.1 exclusion mask (obs only, no hmax yet) ====="
python -m analysis.validation --proc data/processed/udupi_coast

echo "===== IMD 2005 probe (compute-node VERIFY) ====="
if [[ ! -f data/raw/imd/2005.grd && ! -f data/raw/imd/rain/2005.grd && ! -f data/raw/imd/Rainfall_ind2005_rfp25.grd ]]; then
    echo "no 2005.grd; trying imdlib from the compute node (timeout 120s)"
    set +e
    timeout 120 python -c "
import imdlib as imd
imd.get_data('rain', 2005, 2005, fn_format='yearwise', file_dir='data/raw/imd')
print('IMD_GET_DATA_OK')
" 
    echo "imdlib_exit=$?"
    echo "curl IMD HTTPS:"
    timeout 15 curl -sI --max-time 10 https://imdpune.gov.in/cmpg/Griddata/rainfall.php || true
    echo "curl IMD HTTP:"
    timeout 15 curl -sI --max-time 10 http://www.imdpune.gov.in/ || true
    set -e
fi

echo "===== IMD V0 forcing (2005-09-19 through 2005-10-11) ====="
python -m floodca_prep.rainfall \
  --raw data/raw/imd \
  --out data/processed/udupi_coast \
  --aoi config/aoi_udupi.yaml \
  --start 2005-09-19 \
  --end 2005-10-11

python -m floodca_prep.scenarios || true
python -m floodca_prep.warmstart || true

if [[ -f data/processed/udupi_coast/90/lc.bin && -f data/processed/udupi_coast/validation/gfd_2734_flooded_90.bin ]]; then
    echo "===== F1 draft ====="
    python -m analysis.fig_f1 \
      --proc data/processed/udupi_coast \
      --out paper/shared/figures/F1_study_area.png
fi

echo "===== pytest data + validation ====="
python -m pytest tests/py/test_io.py tests/py/test_data.py tests/py/test_validation.py -q

echo "===== C rain / day-clamp ====="
mkdir -p build/p3_cpu
cmake -S . -B build/p3_cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_C_FLAGS="-ffp-contract=off -fno-fast-math" \
  -DFCA_DOUBLE=ON \
  -DFCA_WITH_OMP=OFF \
  -DFCA_WITH_MPI=OFF \
  -DFCA_WITH_CUDA=OFF \
  -DFCA_BUILD_TESTS=ON
cmake --build build/p3_cpu -j "${SLURM_CPUS_PER_TASK}" --target test_rain test_mass
./build/p3_cpu/test_rain
./build/p3_cpu/test_mass

if [[ ! -f data/processed/udupi_coast/forcing_V0.csv ]]; then
    echo "P3_DATA_PARTIAL: landcover + GFD 2734 done; V0 forcing blocked on 2005.grd (Mac IMD download)"
    exit 2
fi
echo "P3_DATA_OK"
