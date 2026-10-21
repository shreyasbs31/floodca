#!/bin/bash
#SBATCH --job-name=floodca-e4gap
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Compiler comparison for the serial vs OpenMP 1-thread gap. No timing result
# from this node is an E4 number: the CPU partition is not H1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/${USER}_mplconfig_${SLURM_JOB_ID}"
OUT=results/e4_gap
mkdir -p "${OUT}"
CC="${CC:-gcc}"
"${CC}" --version | head -1 | tee "${OUT}/gcc_version.txt"
INC="-I src/common -I src/kernels"
BASE="-O3 -march=native -std=c11 -Wall -D_ISOC11_SOURCE ${INC}"

"${CC}" -Q --help=optimize ${BASE} > "${OUT}/flags_optimize_serial.txt" 2>&1
"${CC}" -Q --help=optimize ${BASE} -fopenmp > "${OUT}/flags_optimize_fopenmp.txt" 2>&1
diff -u "${OUT}/flags_optimize_serial.txt" "${OUT}/flags_optimize_fopenmp.txt" \
  > "${OUT}/flags_optimize_diff.txt" || true

"${CC}" ${BASE} -fopt-info-vec-optimized -fopt-info-vec-missed \
  -c src/common/model.c -o "${OUT}/model_serial.o" \
  > "${OUT}/vec_serial.txt" 2>&1
"${CC}" ${BASE} -fopenmp -DFCA_OMP -fopt-info-vec-optimized -fopt-info-vec-missed \
  -c src/common/model.c -o "${OUT}/model_omp.o" \
  > "${OUT}/vec_omp.txt" 2>&1
"${CC}" ${BASE} -fopenmp -fopt-info-vec-optimized -fopt-info-vec-missed \
  -c src/common/model.c -o "${OUT}/model_fopenmp_nopragma.o" \
  > "${OUT}/vec_fopenmp_nopragma.txt" 2>&1

python - <<'PY'
from pathlib import Path
out = Path("results/e4_gap")
lines = []
for name in ("vec_serial.txt", "vec_omp.txt", "vec_fopenmp_nopragma.txt"):
    text = (out / name).read_text(encoding="utf-8", errors="replace")
    opt = [ln for ln in text.splitlines() if "optimized" in ln.lower() or "vectorized" in ln.lower()]
    miss = [ln for ln in text.splitlines() if "missed" in ln.lower()]
    lines.append(f"== {name} optimized_or_vectorized={len(opt)} missed={len(miss)}")
    lines.extend(opt[:80])
diff = (out / "flags_optimize_diff.txt").read_text(encoding="utf-8", errors="replace")
lines.append("== optimize flag diff lines " + str(len(diff.splitlines())))
lines.append(diff[:4000])
(out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:120]))
PY

python - <<'PY'
from pathlib import Path
from analysis.plots import fig_f7, load
df = load(Path("results/bench.csv"))
df = df[df["slurm_job_id"].astype(int).isin([242994, 242995])]
fig_f7(df, Path("paper/shared/figures"), Path("results"), ", preliminary (S and M; L OpenMP not finished)")
print("F7_GAP_OK")
PY
echo "P7_GAP_VEC_OK"
