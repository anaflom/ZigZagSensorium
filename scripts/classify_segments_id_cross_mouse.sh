#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-cross-seg-id
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm-zz-cross-seg-id-%j.out
#SBATCH --error=logs/slurm-zz-cross-seg-id-%j.err

set -euo pipefail

PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_segments_id_cross_mouse.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-gpu}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
MICE="${MICE:-None}"
MODELS="${MODELS:-logreg,cnn3d_raw,cnn3d_norm}"
CLIP_FRAMES="${CLIP_FRAMES:-None}"
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"
CACHE_DIR="${CACHE_DIR:-}"
MAX_TRIALS="${MAX_TRIALS:-None}"

MIN_SEGMENT_REPETITIONS="${MIN_SEGMENT_REPETITIONS:-7}"

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

SEG_LEN_NATURALIMAGES="${SEG_LEN_NATURALIMAGES:-12}"
SEG_LEN_PINKNOISE="${SEG_LEN_PINKNOISE:-27}"
SEG_LEN_RANDOMDOTS="${SEG_LEN_RANDOMDOTS:-60}"
SEG_LEN_GABOR="${SEG_LEN_GABOR:-25}"
SEG_LEN_GAUSSIANDOT="${SEG_LEN_GAUSSIANDOT:-9}"

PER_TRIAL_THRESH_NORM="$(echo "${PER_TRIAL_THRESH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PER_TRIAL_THRESH_NORM}" == "true" ]]; then
  OUTPUT_SUFFIX="per-trial"
else
  OUTPUT_SUFFIX="global"
fi

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/cross_mouse_segment_id_decoding/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_minsegrep-${MIN_SEGMENT_REPETITIONS}_${RUN_TS}"
RUN_TAG_SAFE="$(echo "${RUN_TAG}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
OUT_DIR="${OUTPUT_BASE}/${RUN_TAG_SAFE}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_BASE}"

source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo "Script: ${SCRIPT}"
echo "Output dir: ${OUT_DIR}"
echo "MODELS: ${MODELS}"
echo "MIN_SEGMENT_REPETITIONS: ${MIN_SEGMENT_REPETITIONS}"
echo "============================================"

CMD=(
  python3 -u "${SCRIPT}"
  --output-folder "${OUT_DIR}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --models "${MODELS}"
  --grid-subdir "${GRID_SUBDIR}"
  --min-segment-repetitions "${MIN_SEGMENT_REPETITIONS}"
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
  --seg-len-naturalimages "${SEG_LEN_NATURALIMAGES}"
  --seg-len-pinknoise "${SEG_LEN_PINKNOISE}"
  --seg-len-randomdots "${SEG_LEN_RANDOMDOTS}"
  --seg-len-gabor "${SEG_LEN_GABOR}"
  --seg-len-gaussiandot "${SEG_LEN_GAUSSIANDOT}"
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

"${CMD[@]}"
EXIT_CODE=$?

echo "Finished with exit code: ${EXIT_CODE}"
echo "Output dir: ${OUT_DIR}"
exit ${EXIT_CODE}
