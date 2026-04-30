#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-ablation-shuffle
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-ablation-shuffle-%j.out
#SBATCH --error=logs/slurm-zz-ablation-shuffle-%j.err

# ============================================================================
# Unified shuffle ablation: within-mouse + cross-mouse classification
# using shuffled grid zigzag persistence.
#
# For each shuffle iteration (N total):
#   1. Load grids for all trials
#   2. Shuffle along selected dimension with seed=base_seed + shuffle_id*1000
#   3. Compute zigzag persistence on shuffled grids
#   4. Run within-mouse StratifiedKFold CV (LogReg + 3D-CNN)
#   5. Run cross-mouse LOMO CV (LogReg + 3D-CNN)
#   6. Save results with shuffle_id as column/key
#   7. Clean up shuffled data
#
# Eligibility:
#   - Mouse must have ≥2 labels after valid filtering (checked upfront).
#   - Per fold, label space may be restricted to shared labels.
#
# Saves:
#   - within_mouse_ablation_shuffle_metrics.json + CSV (if not --skip-within-mouse)
#   - cross_mouse_ablation_shuffle_metrics.json + CSV (if not --skip-cross-mouse)
#   - figures (PNG)
#   - run log
#
# Usage examples:
#   sbatch scripts/classify_labels_ablation_shuffle.sh
#
#   sbatch --export=VECTORIZATION_METHOD=Turnover,N_SHUFFLES=10,MICE=dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce \
#          scripts/classify_labels_ablation_shuffle.sh
#
#   sbatch --export=VECTORIZATION_METHOD=BettiCurve,N_SHUFFLES=20,MAX_TRIALS=100,EPOCHS_CNN3D=15 \
#          scripts/classify_labels_ablation_shuffle.sh
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_labels_ablation_shuffle.py"
VENV_DIR="${PROJECT_DIR}/.venv-gpu"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

# Core parameters
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
VECTORIZATION_METHOD="${VECTORIZATION_METHOD:-Turnover}"
MICE="${MICE:-None}"
CLIP_FRAMES="${CLIP_FRAMES:-240}"
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"
MAX_TRIALS="${MAX_TRIALS:-None}"

# Cache
CACHE_DIR="${CACHE_DIR:-}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

# Shuffle parameters
N_SHUFFLES="${N_SHUFFLES:-3}"
SHUFFLE_TYPE="${SHUFFLE_TYPE:-phase}"
MAX_DIM="${MAX_DIM:-2}"
SKIP_WITHIN_MOUSE="${SKIP_WITHIN_MOUSE:-false}"
SKIP_CROSS_MOUSE="${SKIP_CROSS_MOUSE:-false}"
SKIP_EXISTING_SHUFFLES="${SKIP_EXISTING_SHUFFLES:-false}"

# Training parameters (3D-CNN only for shuffle ablations)
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-16}"
EPOCHS_CNN3D="${EPOCHS_CNN3D:-40}"
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

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/ablation_shuffle/${SHUFFLE_TYPE}/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_method-${VECTORIZATION_METHOD}_shuffle-${SHUFFLE_TYPE}_shuffles-${N_SHUFFLES}_clip-${CLIP_FRAMES}_${RUN_TS}"
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
echo "Vectorization method: ${VECTORIZATION_METHOD}"
echo "P_ACTIVE: ${P_ACTIVE}"
echo "PER_TRIAL_THRESH: ${PER_TRIAL_THRESH}"
echo "MICE: ${MICE}"
echo "CLIP_FRAMES: ${CLIP_FRAMES}"
echo "GRID_SUBDIR: ${GRID_SUBDIR}"
echo "MAX_TRIALS: ${MAX_TRIALS}"
echo "FORCE_RECOMPUTE: ${FORCE_RECOMPUTE}"
echo "==========================================="
echo "N_SHUFFLES: ${N_SHUFFLES}"
echo "SHUFFLE_TYPE: ${SHUFFLE_TYPE}"
echo "MAX_DIM: ${MAX_DIM}"
echo "SKIP_WITHIN_MOUSE: ${SKIP_WITHIN_MOUSE}"
echo "SKIP_CROSS_MOUSE: ${SKIP_CROSS_MOUSE}"
echo "SKIP_EXISTING_SHUFFLES: ${SKIP_EXISTING_SHUFFLES}"
echo "==========================================="
echo "BATCH_SIZE_GRID: ${BATCH_SIZE_GRID}"
echo "EPOCHS_CNN3D: ${EPOCHS_CNN3D}"
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
  --vectorization-method "${VECTORIZATION_METHOD}"
  --grid-subdir "${GRID_SUBDIR}"
  --n-shuffles "${N_SHUFFLES}"
  --shuffle-type "${SHUFFLE_TYPE}"
  --max-dim "${MAX_DIM}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --epochs-cnn3d "${EPOCHS_CNN3D}"
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
  CMD+=(--force-recompute true)
fi

CMD+=(--skip-existing-shuffles "${SKIP_EXISTING_SHUFFLES}")

if [[ "${SKIP_WITHIN_MOUSE}" == "true" || "${SKIP_WITHIN_MOUSE}" == "1" ]]; then
  CMD+=(--skip-within-mouse)
fi

if [[ "${SKIP_CROSS_MOUSE}" == "true" || "${SKIP_CROSS_MOUSE}" == "1" ]]; then
  CMD+=(--skip-cross-mouse)
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

echo "Starting ${SHUFFLE_TYPE}-shuffle ablation study..."
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "SUCCESS (exit code: ${EXIT_CODE})"
  echo "Results saved to: ${OUT_DIR}"
else
  echo "FAILED (exit code: ${EXIT_CODE})"
fi
echo "============================================"

exit "${EXIT_CODE}"
