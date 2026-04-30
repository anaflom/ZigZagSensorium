#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-gen-shuffle
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-zz-gen-shuffle-%j.out
#SBATCH --error=logs/slurm-zz-gen-shuffle-%j.err

# ============================================================================
# Generate shuffled zigzag vectorization caches for ablation studies.
#
# This script runs on CPU only (no GPU needed).
# For each mouse it discovers existing cached shuffle IDs and appends new ones,
# N_SHUFFLES is the *target total* — mice that already have >= N_SHUFFLES cached
# shuffles are skipped entirely.  Repeated runs safely extend the pool.
#
# Runtime note: zigzag per trial is expensive. For safer wall-time behavior,
# defaults are conservative (N_SHUFFLES=1, MAX_TRIALS=240).
#
# Usage examples:
#   sbatch scripts/generate_trials_ablation_shuffle.sh
#
#   sbatch --export=N_SHUFFLES=2,SHUFFLE_TYPE=time,MAX_TRIALS=240 \
#          scripts/generate_trials_ablation_shuffle.sh
#
#   sbatch --export=N_SHUFFLES=3,MICE=dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce \
#          scripts/generate_trials_ablation_shuffle.sh
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/generate_trials_ablation_shuffle.py"
VENV_DIR="${PROJECT_DIR}/.venv-genoa"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

# Mouse selection
DEFAULT_MICE="dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce"
MICE="${MICE:-${DEFAULT_MICE}}"

# Shuffle parameters
N_SHUFFLES="${N_SHUFFLES:-1}"
SHUFFLE_TYPE="${SHUFFLE_TYPE:-phase}"
SEED="${SEED:-42}"
DIFFERENT_SHUFFLE_PER_TRIAL="${DIFFERENT_SHUFFLE_PER_TRIAL:-true}"

# Vectorization / zigzag parameters
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
VECTORIZATION_METHOD="${VECTORIZATION_METHOD:-Turnover}"
CLIP_FRAMES="${CLIP_FRAMES:-240}"
MAX_TRIALS="${MAX_TRIALS:-None}"
MAX_DIM="${MAX_DIM:-2}"
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"

# Cache
CACHE_DIR="${CACHE_DIR:-}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

# --- Environment -------------------------------------------------------------
mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job   : ${SLURM_JOB_ID:-N/A}"
echo "Node  : $(hostname)"
echo "CPUs  : ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Memory: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Python: $(which python3)"
echo "============================================"
echo "Script           : ${SCRIPT}"
echo "Data root        : ${DATA_ROOT}"
echo "Meta root        : ${META_ROOT}"
echo "Cache dir        : ${CACHE_DIR:-<data-root>/<mouse>/cache}"
echo "============================================"
echo "N_SHUFFLES       : ${N_SHUFFLES}"
echo "SHUFFLE_TYPE     : ${SHUFFLE_TYPE}"
echo "SEED             : ${SEED}"
echo "DIFF_PER_TRIAL   : ${DIFFERENT_SHUFFLE_PER_TRIAL}"
echo "============================================"
echo "VECTORIZATION    : ${VECTORIZATION_METHOD}"
echo "P_ACTIVE         : ${P_ACTIVE}"
echo "PER_TRIAL_THRESH : ${PER_TRIAL_THRESH}"
echo "CLIP_FRAMES      : ${CLIP_FRAMES}"
echo "MAX_TRIALS       : ${MAX_TRIALS}"
echo "MAX_DIM          : ${MAX_DIM}"
echo "PROGRESS_EVERY   : ${PROGRESS_EVERY}"
echo "NUM_WORKERS      : ${NUM_WORKERS}"
echo "GRID_SUBDIR      : ${GRID_SUBDIR}"
echo "MICE             : ${MICE}"
echo "FORCE_RECOMPUTE  : ${FORCE_RECOMPUTE}"
echo "============================================"

# --- Build command -----------------------------------------------------------
CMD=(
  python3 -u "${SCRIPT}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --n-shuffles "${N_SHUFFLES}"
  --shuffle-type "${SHUFFLE_TYPE}"
  --seed "${SEED}"
  --different-shuffle-per-trial "${DIFFERENT_SHUFFLE_PER_TRIAL}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --vectorization-method "${VECTORIZATION_METHOD}"
  --max-dim "${MAX_DIM}"
  --progress-every "${PROGRESS_EVERY}"
  --num-workers "${NUM_WORKERS}"
  --grid-subdir "${GRID_SUBDIR}"
  --force-recompute "${FORCE_RECOMPUTE}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  CMD+=(--cache-dir "${CACHE_DIR}")
fi

if [[ "${MICE}" != "None" && "${MICE}" != "none" && "${MICE}" != "" ]]; then
  CMD+=(--mice "${MICE}")
fi

if [[ "${CLIP_FRAMES}" != "None" && "${CLIP_FRAMES}" != "none" && "${CLIP_FRAMES}" != "" ]]; then
  CMD+=(--clip-frames "${CLIP_FRAMES}")
fi

if [[ "${MAX_TRIALS}" != "None" && "${MAX_TRIALS}" != "none" && "${MAX_TRIALS}" != "" ]]; then
  CMD+=(--max-trials "${MAX_TRIALS}")
fi

echo ""
echo "Running command:"
printf '  %q' "${CMD[@]}"
echo ""
echo ""

echo "Starting ${SHUFFLE_TYPE}-shuffle generation ..."
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "SUCCESS (exit code: ${EXIT_CODE})"
else
  echo "FAILED  (exit code: ${EXIT_CODE})"
fi
echo "============================================"

exit "${EXIT_CODE}"
