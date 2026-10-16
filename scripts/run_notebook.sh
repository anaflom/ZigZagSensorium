#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=run-notebook
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-run-notebook-%j.out
#SBATCH --error=logs/slurm-run-notebook-%j.err

# ============================================================================
# Execute a Jupyter notebook top to bottom with nbconvert (SLURM).
#
# The notebook runs from its own folder (so relative paths such as ../scripts
# work) and, by default, its outputs are written back into the same file.
# Do not edit the notebook while the job runs: the file is overwritten at the end.
#
# Usage (run sbatch from the repository root, so logs/ is created there):
#   sbatch scripts/run_notebook.sh notebooks/turnover_intepretability.ipynb
#   sbatch --export=ALL,NOTEBOOK=notebooks/turnover_intepretability.ipynb scripts/run_notebook.sh
#
#   # write the executed copy to a new file instead of overwriting the notebook
#   sbatch --export=ALL,OUTPUT=turnover_intepretability_run.ipynb \
#          scripts/run_notebook.sh notebooks/turnover_intepretability.ipynb
#
#   # other resources: standard sbatch options override the defaults above
#   sbatch --cpus-per-task=16 --mem=64G --time=02:00:00 scripts/run_notebook.sh notebooks/other.ipynb
#
# Variables:
#   NOTEBOOK      notebook path, relative to the repository root or absolute (or first argument)
#   OUTPUT        output file name (written next to the notebook); empty = overwrite in place
#   CELL_TIMEOUT  maximum seconds per cell (default 20000; nbconvert's own default is 600)
#   KERNEL_NAME   Jupyter kernel (default python3)
#   ALLOW_ERRORS  true = keep executing after a failing cell and save its traceback (default false)
# ============================================================================

set -euo pipefail

# sbatch starts the job in the submission directory, so the default is the current directory
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

NOTEBOOK="${1:-${NOTEBOOK:-notebooks/turnover_intepretability.ipynb}}"
OUTPUT="${OUTPUT:-turnover_intepretability_results.ipynb}"
CELL_TIMEOUT="${CELL_TIMEOUT:-20000}"
KERNEL_NAME="${KERNEL_NAME:-python3}"
ALLOW_ERRORS="${ALLOW_ERRORS:-false}"

if [[ -z "${NOTEBOOK}" ]]; then
  echo "ERROR: give the notebook path as first argument or as NOTEBOOK=..." >&2
  exit 2
fi
[[ "${NOTEBOOK}" = /* ]] || NOTEBOOK="${PROJECT_DIR}/${NOTEBOOK}"
if [[ ! -f "${NOTEBOOK}" ]]; then
  echo "ERROR: notebook not found: ${NOTEBOOK}" >&2
  exit 2
fi
NB_DIR="$(cd "$(dirname "${NOTEBOOK}")" && pwd)"
NB_FILE="$(basename "${NOTEBOOK}")"

mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
# utils.py imports zztop; fall back to the local checkout if the installed one is not importable
export PYTHONPATH="${PROJECT_DIR}/zz-top:${PYTHONPATH:-}"

CMD=(
  python3 -m nbconvert --to notebook --execute
  "--ExecutePreprocessor.kernel_name=${KERNEL_NAME}"
  "--ExecutePreprocessor.timeout=${CELL_TIMEOUT}"
)
if [[ -n "${OUTPUT}" ]]; then
  CMD+=(--output "${OUTPUT}")
else
  CMD+=(--inplace)
fi
if [[ "${ALLOW_ERRORS}" == "true" ]]; then
  CMD+=(--allow-errors)
fi
CMD+=("${NB_FILE}")

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-N/A}   Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Notebook: ${NB_DIR}/${NB_FILE}"
echo "Output: ${OUTPUT:-in place}"
echo "Cell timeout: ${CELL_TIMEOUT}s   Allow errors: ${ALLOW_ERRORS}"
echo "Python: $(which python3)"
echo "============================================"

cd "${NB_DIR}"
START=$(date +%s)
set +e
"${CMD[@]}"
EXIT_CODE=$?
set -e

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE} after $(( $(date +%s) - START ))s"
echo "============================================"
exit ${EXIT_CODE}
