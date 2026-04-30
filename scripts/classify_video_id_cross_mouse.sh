#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-cross-id-dec
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-cross-id-dec-%j.out
#SBATCH --error=logs/slurm-zz-cross-id-dec-%j.err

# ============================================================================
# Cross-mouse video ID decoding by label.
#
# For each eligible mouse pair with common repeated IDs:
#   - Evaluate both directions (A->B and B->A)
#   - Train on source mouse and test on target mouse
#   - Models: LogReg (zigzag vectorization) and 3D-CNN (grid activations)
#
# Pair eligibility uses metadata with valid_trial & valid_response == True,
# keeping IDs repeated at least MIN_ID_REPETITIONS in each mouse.
# ============================================================================

set -euo pipefail

PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_video_id_cross_mouse.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-gpu}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
VECTORIZATION_METHOD="${VECTORIZATION_METHOD:-Turnover}"
MICE="${MICE:-None}"
CLIP_FRAMES="${CLIP_FRAMES:-240}"
MAX_TRIALS="${MAX_TRIALS:-None}"
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"
CACHE_DIR="${CACHE_DIR:-}"

MIN_ID_REPETITIONS="${MIN_ID_REPETITIONS:-5}"
SEED="${SEED:-42}"

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

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/cross_mouse_id_decoding/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_method-${VECTORIZATION_METHOD}_minrep-${MIN_ID_REPETITIONS}_clip-${CLIP_FRAMES}_${RUN_TS}"
RUN_TAG_SAFE="$(echo "${RUN_TAG}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
OUT_DIR="${OUTPUT_BASE}/${RUN_TAG_SAFE}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_BASE}"

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
if [[ -n "${CACHE_DIR}" ]]; then
  echo "Cache dir override: ${CACHE_DIR}"
else
  echo "Cache dir: <data-root>/<mouse>/cache"
fi
echo "============================================"
echo "P_ACTIVE      : ${P_ACTIVE}"
echo "PER_TRIAL_THRESH : ${PER_TRIAL_THRESH}"
echo "VECTORIZATION : ${VECTORIZATION_METHOD}"
echo "MICE          : ${MICE}"
echo "CLIP_FRAMES   : ${CLIP_FRAMES}"
echo "MAX_TRIALS    : ${MAX_TRIALS}"
echo "GRID_SUBDIR   : ${GRID_SUBDIR}"
echo "============================================"
echo "MIN_ID_REPETITIONS : ${MIN_ID_REPETITIONS}"
echo "SEED          : ${SEED}"
echo "============================================"
echo "BATCH_SIZE    : ${BATCH_SIZE_GRID}"
echo "EPOCHS_CNN3D  : ${EPOCHS_CNN3D}"
echo "LR_CNN3D      : ${LR_CNN3D}"
echo "WEIGHT_DECAY  : ${WEIGHT_DECAY}"
echo "PATIENCE      : ${EARLY_STOP_PATIENCE}"
echo "DEVICE        : ${DEVICE}"
echo "============================================"

CMD=(
  python3 -u "${SCRIPT}"
  --output-folder "${OUT_DIR}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --vectorization-method "${VECTORIZATION_METHOD}"
  --grid-subdir "${GRID_SUBDIR}"
  --min-id-repetitions "${MIN_ID_REPETITIONS}"
  --seed "${SEED}"
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
echo "Starting cross-mouse ID decoding ..."
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
