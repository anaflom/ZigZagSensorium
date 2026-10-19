#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-classify-grid
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm-zz-classify-grid-%A_%a.out
#SBATCH --error=logs/slurm-zz-classify-grid-%A_%a.err

# ==========================================================================
# Classify video labels / video IDs / segment IDs from grid activity (3D-CNN).
# Independent of the zigzag threshold (P_ACTIVE).
#
# Repeats retrain the CNN with a different init seed on the same CV folds,
# to quantify run-to-run variability.  Either one job running all repeats:
#   TARGET=video_label SCOPE=cross_mouse N_REPEATS=5 sbatch scripts/classify_grid.sh
# or one repeat per array task (all tasks share one run folder):
#   TARGET=video_label SCOPE=cross_mouse sbatch --array=0-4 scripts/classify_grid.sh
#
# Problems too long for one job (within-mouse segment_id) can be split into
# one job per mouse that share a run folder (see scripts/run_classify-grid.sh):
#   TARGET=segment_id SCOPE=within_mouse MICE=<mouse> PART=<mouse> RUN_NAME=<shared> \
#     sbatch --array=0-4 scripts/classify_grid.sh
#
# Results: results/grid/<MODEL>/<TARGET>_<SCOPE>/<run>/repeat-NN/
# ==========================================================================

set -euo pipefail

PROJECT_DIR="/u/area/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_grid.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-gpu}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"

# What to classify
TARGET="${TARGET:?Set TARGET=video_label|video_id|segment_id}"
SCOPE="${SCOPE:?Set SCOPE=within_mouse|cross_mouse}"
MICE="${MICE:-None}"
PART="${PART:-}"  # set when several jobs (e.g. one per mouse) share RUN_NAME
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"

# Model and repeats
MODEL="${MODEL:-cnn3d_raw}"
INIT_SEED="${INIT_SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-false}"
DEVICE="${DEVICE:-cuda}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  REPEAT_START="${SLURM_ARRAY_TASK_ID}"
  N_REPEATS=1
  RUN_NAME="${RUN_NAME:-array_${SLURM_ARRAY_JOB_ID}}"
else
  REPEAT_START="${REPEAT_START:-0}"
  N_REPEATS="${N_REPEATS:-1}"
  RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
fi

# Problem definition (empty = defaults documented in classification_data.py)
CLIP_FRAMES="${CLIP_FRAMES:-}"
CV_SCHEME="${CV_SCHEME:-}"
N_SPLITS="${N_SPLITS:-5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MIN_REPETITIONS="${MIN_REPETITIONS:-}"
MAX_TRIALS_PER_MOUSE="${MAX_TRIALS_PER_MOUSE:-}"

# Training
BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-16}"
EPOCHS_CNN3D="${EPOCHS_CNN3D:-40}"
LR_CNN3D="${LR_CNN3D:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
NUM_WORKERS_DL="${NUM_WORKERS_DL:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
if [[ "${DETERMINISTIC,,}" == "true" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
fi

CMD=(
  python3 -u "${SCRIPT}"
  --target "${TARGET}"
  --scope "${SCOPE}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --grid-subdir "${GRID_SUBDIR}"
  --model "${MODEL}"
  --n-repeats "${N_REPEATS}"
  --repeat-start "${REPEAT_START}"
  --init-seed "${INIT_SEED}"
  --deterministic "${DETERMINISTIC}"
  --device "${DEVICE}"
  --n-splits "${N_SPLITS}"
  --split-seed "${SPLIT_SEED}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --epochs-cnn3d "${EPOCHS_CNN3D}"
  --lr-cnn3d "${LR_CNN3D}"
  --weight-decay "${WEIGHT_DECAY}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --num-workers-dl "${NUM_WORKERS_DL}"
  --run-name "${RUN_NAME}"
)
[[ "${MICE}" != "None" && -n "${MICE}" ]] && CMD+=(--mice "${MICE}")
[[ -n "${PART}" ]] && CMD+=(--part "${PART}")
[[ -n "${CLIP_FRAMES}" ]] && CMD+=(--clip-frames "${CLIP_FRAMES}")
[[ -n "${CV_SCHEME}" ]] && CMD+=(--cv-scheme "${CV_SCHEME}")
[[ -n "${MIN_REPETITIONS}" ]] && CMD+=(--min-repetitions "${MIN_REPETITIONS}")
[[ -n "${MAX_TRIALS_PER_MOUSE}" ]] && CMD+=(--max-trials-per-mouse "${MAX_TRIALS_PER_MOUSE}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS}" ]] && CMD+=(${EXTRA_ARGS})

echo "Job: ${SLURM_JOB_ID:-N/A} (array task ${SLURM_ARRAY_TASK_ID:-none}) on $(hostname)"
echo "${CMD[*]}"
"${CMD[@]}"
