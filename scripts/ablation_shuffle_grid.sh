#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-ablation-shuffle-grid
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm-zz-ablation-shuffle-grid-%j.out
#SBATCH --error=logs/slurm-zz-ablation-shuffle-grid-%j.err

# ==========================================================================
# Shuffle ablation on shuffled grids (3D-CNN).
# Shuffles must exist (scripts/generate_trials_ablation_shuffle.sh); GEN_P_ACTIVE,
# GEN_PER_TRIAL_THRESH and GENERATION_CLIP_FRAMES must match the generation run,
# because they are part of the folder names of the shuffled grids.
#
#   SCOPE=within_mouse SHUFFLE_TYPE=time N_SHUFFLES=5 GEN_P_ACTIVE=30 sbatch scripts/ablation_shuffle_grid.sh
#
# Results: results/ablation_shuffle/grid/<MODEL>/<SHUFFLE_TYPE>-<mode>/<TARGET>_<SCOPE>/<run>/shuffle-NNNN/
# ==========================================================================

set -euo pipefail

PROJECT_DIR="/u/area/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/ablation_shuffle_grid.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-gpu}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"

TARGET="${TARGET:-video_label}"
SCOPE="${SCOPE:?Set SCOPE=within_mouse|cross_mouse}"
MICE="${MICE:-None}"

SHUFFLE_TYPE="${SHUFFLE_TYPE:-time}"
DIFFERENT_SHUFFLE_PER_TRIAL="${DIFFERENT_SHUFFLE_PER_TRIAL:-true}"
N_SHUFFLES="${N_SHUFFLES:-5}"
SHUFFLE_IDS="${SHUFFLE_IDS:-}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
GEN_VECTORIZATION="${GEN_VECTORIZATION:-Turnover}"
GEN_P_ACTIVE="${GEN_P_ACTIVE:-30}"
GEN_PER_TRIAL_THRESH="${GEN_PER_TRIAL_THRESH:-true}"
GENERATION_CLIP_FRAMES="${GENERATION_CLIP_FRAMES:-240}"
CACHE_DIR="${CACHE_DIR:-}"

MODEL="${MODEL:-cnn3d_raw}"
INIT_SEED="${INIT_SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-false}"
DEVICE="${DEVICE:-cuda}"
N_SPLITS="${N_SPLITS:-5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
EPOCHS_CNN3D="${EPOCHS_CNN3D:-40}"
NUM_WORKERS_DL="${NUM_WORKERS_DL:-4}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
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
  --shuffle-type "${SHUFFLE_TYPE}"
  --different-shuffle-per-trial "${DIFFERENT_SHUFFLE_PER_TRIAL}"
  --shuffle-seed "${SHUFFLE_SEED}"
  --gen-vectorization "${GEN_VECTORIZATION}"
  --gen-p-active "${GEN_P_ACTIVE}"
  --gen-per-trial-thresh "${GEN_PER_TRIAL_THRESH}"
  --generation-clip-frames "${GENERATION_CLIP_FRAMES}"
  --model "${MODEL}"
  --init-seed "${INIT_SEED}"
  --deterministic "${DETERMINISTIC}"
  --device "${DEVICE}"
  --n-splits "${N_SPLITS}"
  --split-seed "${SPLIT_SEED}"
  --epochs-cnn3d "${EPOCHS_CNN3D}"
  --num-workers-dl "${NUM_WORKERS_DL}"
  --run-name "${RUN_NAME}"
)
if [[ -n "${SHUFFLE_IDS}" ]]; then
  CMD+=(--shuffle-ids "${SHUFFLE_IDS}")
else
  CMD+=(--n-shuffles "${N_SHUFFLES}")
fi
[[ "${MICE}" != "None" && -n "${MICE}" ]] && CMD+=(--mice "${MICE}")
[[ -n "${CACHE_DIR}" ]] && CMD+=(--cache-dir "${CACHE_DIR}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS}" ]] && CMD+=(${EXTRA_ARGS})

echo "Job: ${SLURM_JOB_ID:-N/A} on $(hostname)"
echo "${CMD[*]}"
"${CMD[@]}"
