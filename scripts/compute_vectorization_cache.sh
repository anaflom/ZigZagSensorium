#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-compute-vector-cache
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-zz-compute-vector-cache-%j.out
#SBATCH --error=logs/slurm-zz-compute-vector-cache-%j.err

# ==========================================================================
# Precompute vectorization caches used by strict-cache analysis scripts.
# ==========================================================================

set -euo pipefail

PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/compute_vectorization_cache.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

MICE="${MICE:-None}"
CACHE_DIR="${CACHE_DIR:-}"
MAX_TRIALS="${MAX_TRIALS:-None}"

VECTORIZATION_METHOD="${VECTORIZATION_METHOD:-Turnover}"
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
CLIP_FRAMES="${CLIP_FRAMES:-None}"

NUM_WORKERS="${NUM_WORKERS:-32}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

mkdir -p "${PROJECT_DIR}/logs"

source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo "Script: ${SCRIPT}"
echo "Data root: ${DATA_ROOT}"
echo "Meta root: ${META_ROOT}"
echo "Method: ${VECTORIZATION_METHOD}"
echo "p_active: ${P_ACTIVE}"
echo "per_trial_thresh: ${PER_TRIAL_THRESH}"
echo "clip_frames: ${CLIP_FRAMES}"
echo "num_workers: ${NUM_WORKERS}"
echo "============================================"

CMD=(
  python3 -u "${SCRIPT}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --vectorization-method "${VECTORIZATION_METHOD}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --clip-frames "${CLIP_FRAMES}"
  --num-workers "${NUM_WORKERS}"
  --progress-every "${PROGRESS_EVERY}"
  --force-recompute "${FORCE_RECOMPUTE}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  CMD+=(--cache-dir "${CACHE_DIR}")
fi

if [[ "${MICE}" != "None" && "${MICE}" != "none" && "${MICE}" != "" ]]; then
  CMD+=(--mice "${MICE}")
fi

if [[ "${MAX_TRIALS}" != "None" && "${MAX_TRIALS}" != "none" && "${MAX_TRIALS}" != "" ]]; then
  CMD+=(--max-trials "${MAX_TRIALS}")
fi

echo ""
echo "Running command:"
printf '  %q' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}
