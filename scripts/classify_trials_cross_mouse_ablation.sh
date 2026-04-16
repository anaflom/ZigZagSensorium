#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-cross-mouse-ablation
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-cross-mouse-ablation-%j.out
#SBATCH --error=logs/slurm-zz-cross-mouse-ablation-%j.err

# ============================================================================
# Cross-mouse leave-one-mouse-out classification using zigzag vectorizations
# + 3D-CNN on raw grids.
#
# Per held-out test mouse:
#   - Train LogReg / MLP / 1D-CNN on vectorization features pooled from all
#     other eligible mice.
#   - Train 3D-CNN on pooled raw grids from all other eligible mice.
#   - Evaluate on the held-out mouse.
#
# Eligibility:
#   - Mouse must have more than one available label after valid filtering.
#   - Per fold, label space is limited to labels shared by test mouse and
#     training pool.
#
# Saves:
#   - figures (PNG)
#   - run log
#   - metrics summary JSON + CSV
#
# Usage examples:
#   sbatch scripts/classify_trials_cross_mouse_ablation.sh
#
#   sbatch --export=METHOD=Turnover,MICE=dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce \
#          scripts/classify_trials_cross_mouse_ablation.sh
#
#   sbatch --export=METHOD=Turnover,MAX_TRIALS=120,EPOCHS_MLP=20,EPOCHS_CNN1D=20,EPOCHS_CNN3D=12 \
#          scripts/classify_trials_cross_mouse_ablation.sh
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_trials_cross_mouse_ablation.py"
VENV_DIR="${PROJECT_DIR}/.venv-gpu"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

# Core parameters
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
METHOD="${METHOD:-Turnover}"
MICE="${MICE:-None}"
CLIP_FRAMES="${CLIP_FRAMES:-None}"
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"
MAX_TRIALS="${MAX_TRIALS:-None}"

# Cache
CACHE_DIR="${CACHE_DIR:-}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

# Training parameters
BATCH_SIZE_VEC="${BATCH_SIZE_VEC:-64}"
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-16}"
EPOCHS_MLP="${EPOCHS_MLP:-60}"
EPOCHS_CNN1D="${EPOCHS_CNN1D:-60}"
EPOCHS_CNN3D="${EPOCHS_CNN3D:-40}"
LR_VEC="${LR_VEC:-0.001}"
LR_CNN3D="${LR_CNN3D:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS_DL="${NUM_WORKERS_DL:-0}"

PER_TRIAL_THRESH_NORM="$(echo "${PER_TRIAL_THRESH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PER_TRIAL_THRESH_NORM}" == "true" ]]; then
  OUTPUT_SUFFIX="per-trial"
else
  OUTPUT_SUFFIX="global"
fi

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/cross_mouse_classification_ablation/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_thr-${PER_TRIAL_THRESH}_method-${METHOD}_clip-${CLIP_FRAMES}_${RUN_TS}"
RUN_TAG_SAFE="$(echo "${RUN_TAG}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
OUT_DIR="${OUTPUT_BASE}/${RUN_TAG_SAFE}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_BASE}"

# --- Environment -------------------------------------------------------------
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Memory: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Partition: ${SLURM_PARTITION:-N/A}"
echo "Python: $(which python3)"
echo "============================================"
echo "Script: ${SCRIPT}"
echo "Data root: ${DATA_ROOT}"
echo "Meta root: ${META_ROOT}"
echo "Output dir: ${OUT_DIR}"
if [[ -n "${CACHE_DIR}" ]]; then
  echo "Cache dir override: ${CACHE_DIR}"
else
  echo "Cache dir: <data-root>/<mouse>/cache"
fi
echo "============================================"
echo "Method: ${METHOD}"
echo "P_ACTIVE: ${P_ACTIVE}"
echo "PER_TRIAL_THRESH: ${PER_TRIAL_THRESH}"
echo "MICE: ${MICE}"
echo "CLIP_FRAMES: ${CLIP_FRAMES}"
echo "GRID_SUBDIR: ${GRID_SUBDIR}"
echo "MAX_TRIALS: ${MAX_TRIALS}"
echo "FORCE_RECOMPUTE: ${FORCE_RECOMPUTE}"
echo "BATCH_SIZE_VEC: ${BATCH_SIZE_VEC}"
echo "BATCH_SIZE_GRID: ${BATCH_SIZE_GRID}"
echo "EPOCHS_MLP: ${EPOCHS_MLP}"
echo "EPOCHS_CNN1D: ${EPOCHS_CNN1D}"
echo "EPOCHS_CNN3D: ${EPOCHS_CNN3D}"
echo "LR_VEC: ${LR_VEC}"
echo "LR_CNN3D: ${LR_CNN3D}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAY}"
echo "EARLY_STOP_PATIENCE: ${EARLY_STOP_PATIENCE}"
echo "SEED: ${SEED}"
echo "DEVICE: ${DEVICE}"
echo "NUM_WORKERS_DL: ${NUM_WORKERS_DL}"
echo "============================================"

# --- Build command -----------------------------------------------------------
CMD=(
  python3 -u "${SCRIPT}"
  --output-folder "${OUT_DIR}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --method "${METHOD}"
  --grid-subdir "${GRID_SUBDIR}"
  --batch-size-vec "${BATCH_SIZE_VEC}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --epochs-mlp "${EPOCHS_MLP}"
  --epochs-cnn1d "${EPOCHS_CNN1D}"
  --epochs-cnn3d "${EPOCHS_CNN3D}"
  --lr-vec "${LR_VEC}"
  --lr-cnn3d "${LR_CNN3D}"
  --weight-decay "${WEIGHT_DECAY}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num-workers-dl "${NUM_WORKERS_DL}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  CMD+=(--cache-dir "${CACHE_DIR}")
fi

if [[ "${FORCE_RECOMPUTE}" == "true" || "${FORCE_RECOMPUTE}" == "1" ]]; then
  CMD+=(--force-recompute)
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

echo "Starting cross-mouse classification..."
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "Run output folder: ${OUT_DIR}"
if [[ -f "${OUT_DIR}/cross_mouse_metrics.json" ]]; then
  echo "  JSON: ${OUT_DIR}/cross_mouse_metrics.json"
  echo "  CSV:  ${OUT_DIR}/cross_mouse_metrics.csv"
  echo "  Figures:"
  echo "    - ${OUT_DIR}/figures/01_lomo_macro_f1_by_test_mouse.png"
  echo "    - ${OUT_DIR}/figures/02_lomo_mean_scores.png"
  echo "    - ${OUT_DIR}/figures/03_best_model_confusion_matrices.png"
  echo "  Log:  ${OUT_DIR}/logs/run.log"
fi
if [[ -n "${CACHE_DIR}" ]]; then
  echo "Vectorization cache: ${CACHE_DIR}"
else
  echo "Vectorization cache: <data-root>/<mouse>/cache"
fi
echo "============================================"

exit ${EXIT_CODE}