#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-classify-zigzag
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-classify-zigzag-%j.out
#SBATCH --error=logs/slurm-zz-classify-zigzag-%j.err

# ==========================================================================
# Classify video labels / video IDs / segment IDs from zigzag vectorizations.
#
#   TARGET=video_label SCOPE=within_mouse P_ACTIVE=30 sbatch scripts/classify_zigzag.sh
#   TARGET=video_id    SCOPE=cross_mouse  P_ACTIVE=70 sbatch scripts/classify_zigzag.sh
#   TARGET=segment_id  SCOPE=within_mouse MICE=<m1>,<m2> sbatch scripts/classify_zigzag.sh
#
# Neural models (MODEL=mlp|cnn1d) run faster on a GPU:
#   MODEL=mlp DEVICE=cuda sbatch --partition=GPU --gres=gpu:V100:1 scripts/classify_zigzag.sh
#
# Results: results/zigzag/<VECTORIZATION>/<MODEL>/p<P>-<per-trial|global>/<TARGET>_<SCOPE>/<run>/
# ==========================================================================

set -euo pipefail

PROJECT_DIR="/u/area/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_zigzag.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"

# What to classify
TARGET="${TARGET:?Set TARGET=video_label|video_id|segment_id}"
SCOPE="${SCOPE:?Set SCOPE=within_mouse|cross_mouse}"
MICE="${MICE:-None}"

# Zigzag features
VECTORIZATION="${VECTORIZATION:-Turnover}"
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
CACHE_DIR="${CACHE_DIR:-}"

# Model and repeats (repeats only matter for neural models)
MODEL="${MODEL:-logreg}"
N_REPEATS="${N_REPEATS:-1}"
INIT_SEED="${INIT_SEED:-42}"
DEVICE="${DEVICE:-cpu}"

# Problem definition (empty = defaults documented in classification_data.py)
CLIP_FRAMES="${CLIP_FRAMES:-}"
CV_SCHEME="${CV_SCHEME:-}"
N_SPLITS="${N_SPLITS:-5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MIN_REPETITIONS="${MIN_REPETITIONS:-}"
MAX_TRIALS_PER_MOUSE="${MAX_TRIALS_PER_MOUSE:-}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

CMD=(
  python3 -u "${SCRIPT}"
  --target "${TARGET}"
  --scope "${SCOPE}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --vectorization "${VECTORIZATION}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --model "${MODEL}"
  --n-repeats "${N_REPEATS}"
  --init-seed "${INIT_SEED}"
  --device "${DEVICE}"
  --n-splits "${N_SPLITS}"
  --split-seed "${SPLIT_SEED}"
  --run-name "${RUN_NAME}"
)
[[ "${MICE}" != "None" && -n "${MICE}" ]] && CMD+=(--mice "${MICE}")
[[ -n "${CACHE_DIR}" ]] && CMD+=(--cache-dir "${CACHE_DIR}")
[[ -n "${CLIP_FRAMES}" ]] && CMD+=(--clip-frames "${CLIP_FRAMES}")
[[ -n "${CV_SCHEME}" ]] && CMD+=(--cv-scheme "${CV_SCHEME}")
[[ -n "${MIN_REPETITIONS}" ]] && CMD+=(--min-repetitions "${MIN_REPETITIONS}")
[[ -n "${MAX_TRIALS_PER_MOUSE}" ]] && CMD+=(--max-trials-per-mouse "${MAX_TRIALS_PER_MOUSE}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS}" ]] && CMD+=(${EXTRA_ARGS})

echo "Job: ${SLURM_JOB_ID:-N/A} on $(hostname)"
echo "${CMD[*]}"
"${CMD[@]}"
