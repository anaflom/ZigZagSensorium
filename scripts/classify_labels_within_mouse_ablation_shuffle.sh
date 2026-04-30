#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-within-shuf
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-within-shuf-%j.out
#SBATCH --error=logs/slurm-zz-within-shuf-%j.err

# ============================================================================
# Within-mouse classification on pre-cached shuffled zigzag vectorizations.
#
# Requires shuffles to have been generated first by:
#   sbatch scripts/generate_labels_ablation_shuffle.sh
#
# Preflight checks are run before any training:
#   - Each mouse must have >= N_SHUFFLES available randomizations.
#   - Shuffles must have been generated with at least MAX_TRIALS trials.
#   - N_SHUFFLES IDs are randomly sampled from the available pool (seeded).
#
# Usage examples:
#   sbatch scripts/classify_labels_within_mouse_ablation_shuffle.sh
#
#   sbatch --export=N_SHUFFLES=10,SHUFFLE_TYPE=time,SEED=0 \
#          scripts/classify_labels_within_mouse_ablation_shuffle.sh
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_labels_within_mouse_ablation_shuffle.py"
VENV_DIR="${PROJECT_DIR}/.venv-gpu"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

# Mouse selection
MICE="${MICE:-None}"

# Shuffle parameters
N_SHUFFLES="${N_SHUFFLES:-1}"
SHUFFLE_TYPE="${SHUFFLE_TYPE:-time}"
SEED="${SEED:-42}"
DIFFERENT_SHUFFLE_PER_TRIAL="${DIFFERENT_SHUFFLE_PER_TRIAL:-true}"

# Vectorization parameters (must match generation)
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
VECTORIZATION_METHOD="${VECTORIZATION_METHOD:-Turnover}"
CLIP_FRAMES="${CLIP_FRAMES:-240}"
MAX_TRIALS="${MAX_TRIALS:-None}"
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"
MAX_DIM="${MAX_DIM:-2}"

# Cache
CACHE_DIR="${CACHE_DIR:-}"

# Training parameters
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-16}"
EPOCHS_CNN3D="${EPOCHS_CNN3D:-40}"
LR_CNN3D="${LR_CNN3D:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS_DL="${NUM_WORKERS_DL:-0}"

PER_TRIAL_THRESH_NORM="$(echo "${PER_TRIAL_THRESH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PER_TRIAL_THRESH_NORM}" == "true" ]]; then
  OUTPUT_SUFFIX="per-trial"
else
  OUTPUT_SUFFIX="global"
fi

DIFF_PER_TRIAL_NORM="$(echo "${DIFFERENT_SHUFFLE_PER_TRIAL}" | tr '[:upper:]' '[:lower:]')"
if [[ "${DIFF_PER_TRIAL_NORM}" == "true" ]]; then
  SHUFFLE_MODE="different"
else
  SHUFFLE_MODE="same"
fi

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/ablation_shuffle/${SHUFFLE_TYPE}/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="within_p${P_ACTIVE}_method-${VECTORIZATION_METHOD}_shuffle-${SHUFFLE_TYPE}_mode-${SHUFFLE_MODE}_nshuf-${N_SHUFFLES}_clip-${CLIP_FRAMES}_${RUN_TS}"
RUN_TAG_SAFE="$(echo "${RUN_TAG}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
OUT_DIR="${OUTPUT_BASE}/${RUN_TAG_SAFE}"

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
echo "Script        : ${SCRIPT}"
echo "Data root     : ${DATA_ROOT}"
echo "Meta root     : ${META_ROOT}"
echo "Output dir    : ${OUT_DIR}"
echo "Cache dir     : ${CACHE_DIR:-<data-root>/<mouse>/cache}"
echo "============================================"
echo "N_SHUFFLES    : ${N_SHUFFLES}"
echo "SHUFFLE_TYPE  : ${SHUFFLE_TYPE}"
echo "SEED          : ${SEED}"
echo "SHUFFLE_MODE  : ${SHUFFLE_MODE}"
echo "DIFF_PER_TRIAL: ${DIFFERENT_SHUFFLE_PER_TRIAL}"
echo "============================================"
echo "VECTORIZATION : ${VECTORIZATION_METHOD}"
echo "P_ACTIVE      : ${P_ACTIVE}"
echo "PER_TRIAL_THRESH : ${PER_TRIAL_THRESH}"
echo "CLIP_FRAMES   : ${CLIP_FRAMES}"
echo "MAX_TRIALS    : ${MAX_TRIALS}"
echo "MICE          : ${MICE}"
echo "============================================"
echo "BATCH_SIZE    : ${BATCH_SIZE_GRID}"
echo "EPOCHS_CNN3D  : ${EPOCHS_CNN3D}"
echo "LR_CNN3D      : ${LR_CNN3D}"
echo "WEIGHT_DECAY  : ${WEIGHT_DECAY}"
echo "PATIENCE      : ${EARLY_STOP_PATIENCE}"
echo "DEVICE        : ${DEVICE}"
echo "============================================"

# --- Build command -----------------------------------------------------------
CMD=(
  python3 -u "${SCRIPT}"
  --output-folder "${OUT_DIR}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --n-shuffles "${N_SHUFFLES}"
  --shuffle-type "${SHUFFLE_TYPE}"
  --seed "${SEED}"
  --different-shuffle-per-trial "${DIFFERENT_SHUFFLE_PER_TRIAL}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --vectorization-method "${VECTORIZATION_METHOD}"
  --grid-subdir "${GRID_SUBDIR}"
  --max-dim "${MAX_DIM}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --epochs-cnn3d "${EPOCHS_CNN3D}"
  --lr-cnn3d "${LR_CNN3D}"
  --weight-decay "${WEIGHT_DECAY}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --device "${DEVICE}"
  --num-workers-dl "${NUM_WORKERS_DL}"
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

mkdir -p "${OUT_DIR}"
echo "Starting within-mouse shuffle classification ..."
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "SUCCESS (exit code: ${EXIT_CODE})"
  echo "Results: ${OUT_DIR}"
else
  echo "FAILED  (exit code: ${EXIT_CODE})"
fi
echo "============================================"

exit "${EXIT_CODE}"
