#!/bin/bash
#SBATCH --job-name=floodca-paprev
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Draft and clean LaTeX only. No experiment, no figure redraw.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source slurm/env.sh
python3 paper/audit/check_claims.py
python3 paper/audit/check_claims.py paper/pdp/sections paper/audit/claims_pdp.csv
export TECTONIC=${FCA_SCRATCH}/opt/tectonic/tectonic
export TECTONIC_CACHE_DIR=${FCA_SCRATCH}/opt/tectonic-cache
mkdir -p paper/audit/pdf
report() {
  local tag="$1" pdf="$2" log="$3"
  echo "===== ${tag} ====="
  if [[ -f "${log}" ]]; then
    echo -n "overfull "
    grep -c "Overfull" "${log}" || true
    grep "Output written on main.xdv" "${log}" || true
  fi
  if command -v pdftotext >/dev/null 2>&1; then
    pdftotext "${pdf}" - | grep -c '\[?\]' || true
  fi
}
bash paper/build.sh
cp paper/ieee/main.pdf paper/audit/pdf/draft_ieee.pdf
cp paper/springer/main.pdf paper/audit/pdf/draft_springer.pdf
cp paper/ieee/main.log paper/audit/pdf/draft_ieee.log
cp paper/springer/main.log paper/audit/pdf/draft_springer.log
cp paper/ieee/main.blg paper/audit/pdf/draft_ieee.blg
cp paper/springer/main.blg paper/audit/pdf/draft_springer.blg
report draft_ieee paper/audit/pdf/draft_ieee.pdf paper/audit/pdf/draft_ieee.log
cp paper/pdp/main.pdf paper/audit/pdf/draft_pdp.pdf
cp paper/pdp/main.log paper/audit/pdf/draft_pdp.log
report draft_springer paper/audit/pdf/draft_springer.pdf paper/audit/pdf/draft_springer.log
report draft_pdp paper/audit/pdf/draft_pdp.pdf paper/audit/pdf/draft_pdp.log
restore_draft() { sed -i 's/^\\draftfalse/\\drafttrue/' paper/shared/sections/macros.tex; }
trap restore_draft EXIT
sed -i 's/^\\drafttrue/\\draftfalse/' paper/shared/sections/macros.tex
bash paper/build.sh
cp paper/ieee/main.pdf paper/audit/pdf/clean_ieee.pdf
cp paper/springer/main.pdf paper/audit/pdf/clean_springer.pdf
cp paper/ieee/main.log paper/audit/pdf/clean_ieee.log
cp paper/springer/main.log paper/audit/pdf/clean_springer.log
cp paper/ieee/main.blg paper/audit/pdf/clean_ieee.blg
cp paper/springer/main.blg paper/audit/pdf/clean_springer.blg
report clean_ieee paper/audit/pdf/clean_ieee.pdf paper/audit/pdf/clean_ieee.log
cp paper/pdp/main.pdf paper/audit/pdf/clean_pdp.pdf
cp paper/pdp/main.log paper/audit/pdf/clean_pdp.log
report clean_springer paper/audit/pdf/clean_springer.pdf paper/audit/pdf/clean_springer.log
report clean_pdp paper/audit/pdf/clean_pdp.pdf paper/audit/pdf/clean_pdp.log
cp paper/audit/pdf/draft_ieee.pdf paper/ieee/main.pdf
cp paper/audit/pdf/draft_springer.pdf paper/springer/main.pdf
echo "PAPER_REV_OK"
