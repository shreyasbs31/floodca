#!/bin/bash
#SBATCH --job-name=floodca-papertex
#SBATCH --account=__FCA_ACCOUNT__
#SBATCH --partition=__FCA_PARTITION_CPU__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

# Compile both papers from a clean aux. Not part of the experiment chain.
# Compute nodes can reach the network; do not pass --only-cached.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source slurm/env.sh
export PYTHONPATH="${SLURM_SUBMIT_DIR}/prep:${SLURM_SUBMIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python3 paper/audit/check_claims.py paper/shared/sections paper/audit/claims.csv
python3 paper/audit/check_claims.py paper/pdp/sections paper/audit/claims_pdp.csv
python -m analysis.plots --only F7
python -m analysis.plots --only F8
python -m analysis.plots --only F9 --jobs 242998
python -m analysis.plots --only F10
python -m analysis.plots --only F11
python -m analysis.plots --only F12
python -m analysis.plots --only F13
python analysis/fig_study.py
export TECTONIC=${FCA_SCRATCH}/opt/tectonic/tectonic
export TECTONIC_CACHE_DIR=${FCA_SCRATCH}/opt/tectonic-cache
mkdir -p paper/audit/pdf
report() {
  local tag="$1" pdf="$2" log="$3" blg="$4"
  echo "===== ${tag} ====="
  echo "----- ${tag} blg -----"
  if [[ -f "${blg}" ]]; then
    grep -E "Error|Warning--|I couldn't|I didn't" "${blg}" || echo "BLG_CLEAN"
  else
    echo "BLG_MISSING"
  fi
  echo "----- ${tag} overfull count -----"
  if [[ -f "${log}" ]]; then
    grep -c "Overfull" "${log}" || true
  else
    echo "LOG_MISSING"
  fi
  echo "----- ${tag} pages -----"
  if [[ -f "${log}" ]] && grep -q "Output written on main.xdv" "${log}"; then
    grep "Output written on main.xdv" "${log}"
  elif command -v pdfinfo >/dev/null 2>&1; then
    pdfinfo "${pdf}" | awk '/Pages:/ {print $2}'
  else
    python3 - "${pdf}" << 'PY'
import re, sys
data = open(sys.argv[1], "rb").read()
# Tectonic writes /Type /Pages /Count N in the page tree.
counts = [int(m) for m in re.findall(br"/Type\s*/Pages\s*/Count\s+(\d+)", data)]
print(max(counts) if counts else "PAGES_UNKNOWN")
PY
  fi
  echo "----- ${tag} citation qmark -----"
  if command -v pdftotext >/dev/null 2>&1; then
    if pdftotext "${pdf}" - | grep -n '\[?\]'; then
      echo "QMARK_FOUND"
    else
      echo "NO_QMARK"
    fi
  else
    python3 - "${pdf}" << 'PY'
import sys
data = open(sys.argv[1], "rb").read()
print("QMARK_FOUND" if b"[?]" in data else "NO_QMARK")
PY
  fi
}
# Draft: \trace prints. The flag in macros.tex is \drafttrue.
bash paper/build.sh
cp paper/ieee/main.pdf paper/audit/pdf/draft_ieee.pdf
cp paper/springer/main.pdf paper/audit/pdf/draft_springer.pdf
cp paper/ieee/main.log paper/audit/pdf/draft_ieee.log
cp paper/springer/main.log paper/audit/pdf/draft_springer.log
cp paper/ieee/main.blg paper/audit/pdf/draft_ieee.blg
cp paper/springer/main.blg paper/audit/pdf/draft_springer.blg
cp paper/pdp/main.pdf paper/audit/pdf/draft_pdp.pdf
cp paper/pdp/main.log paper/audit/pdf/draft_pdp.log
cp paper/pdp/main.blg paper/audit/pdf/draft_pdp.blg
report draft_ieee paper/audit/pdf/draft_ieee.pdf paper/audit/pdf/draft_ieee.log paper/audit/pdf/draft_ieee.blg
report draft_springer paper/audit/pdf/draft_springer.pdf paper/audit/pdf/draft_springer.log paper/audit/pdf/draft_springer.blg
report draft_pdp paper/audit/pdf/draft_pdp.pdf paper/audit/pdf/draft_pdp.log paper/audit/pdf/draft_pdp.blg
# Clean: no trace pointers. The reproducibility appendix stays.
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
cp paper/pdp/main.pdf paper/audit/pdf/clean_pdp.pdf
cp paper/pdp/main.log paper/audit/pdf/clean_pdp.log
cp paper/pdp/main.blg paper/audit/pdf/clean_pdp.blg
report clean_ieee paper/audit/pdf/clean_ieee.pdf paper/audit/pdf/clean_ieee.log paper/audit/pdf/clean_ieee.blg
report clean_springer paper/audit/pdf/clean_springer.pdf paper/audit/pdf/clean_springer.log paper/audit/pdf/clean_springer.blg
report clean_pdp paper/audit/pdf/clean_pdp.pdf paper/audit/pdf/clean_pdp.log paper/audit/pdf/clean_pdp.blg
# The class directories follow the draft flag, which the exit trap restores.
cp paper/audit/pdf/draft_ieee.pdf paper/ieee/main.pdf
cp paper/audit/pdf/draft_springer.pdf paper/springer/main.pdf
cp paper/audit/pdf/draft_pdp.pdf paper/pdp/main.pdf
echo "----- abstract words (tex, comments stripped) -----"
python3 - << 'PY'
import re
from pathlib import Path
for name in ("paper/shared/sections/abstract.tex", "paper/pdp/sections/abstract.tex"):
    t = Path(name).read_text()
    t = re.sub(r"%.*", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.'-]*", t)
    print(name, len(words))
PY
echo "----- placeholders and TODOs -----"
grep -RInE "TODO|FIXME|PLACEHOLDER|\\\\todo|\\[email\\]|FILL BEFORE" paper/shared paper/pdp paper/ieee/main.tex paper/springer/main.tex paper/pdp/main.tex || echo "NONE"
echo "PAPER_TEX_OK"
