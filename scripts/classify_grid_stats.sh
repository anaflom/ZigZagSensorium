#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-classify-grid-stats
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-classify-grid-stats-%j.out
#SBATCH --error=logs/slurm-zz-classify-grid-stats-%j.err

# ==========================================================================
# Classify from structure-blind grid statistics (needs compute_grid_stats_cache.sh).
#
#   FEATURES=moments_raw     TARGET=video_label SCOPE=within_mouse sbatch scripts/classify_grid_stats.sh
#   FEATURES=moments_density TARGET=video_id    SCOPE=cross_mouse TIME_AVERAGE=true sbatch scripts/classify_grid_stats.sh
#   FEATURES=active_fraction P_ACTIVE=30 TARGET=segment_id SCOPE=within_mouse sbatch scripts/classify_grid_stats.sh
#
# Results: results/grid_stats/<features>[-trialavg]/<MODEL>/[p<P>-<...>/]<TARGET>_<SCOPE>/<run>/
# ==========================================================================

set -euo pipefail

PROJECT_DIR="/u/area/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_grid_stats.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"

FEATURES="${FEATURES:?Set FEATURES=moments_raw|moments_density|moments_std|moments_std_density|active_fraction}"
TARGET="${TARGET:?Set TARGET=video_label|video_id|segment_id}"
SCOPE="${SCOPE:?Set SCOPE=within_mouse|cross_mouse}"
TIME_AVERAGE="${TIME_AVERAGE:-false}"
P_ACTIVE="${P_ACTIVE:-}"
ABS_THRESHOLD="${ABS_THRESHOLD:-}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
MICE="${MICE:-None}"
CACHE_DIR="${CACHE_DIR:-}"

MODEL="${MODEL:-logreg}"
N_REPEATS="${N_REPEATS:-1}"
INIT_SEED="${INIT_SEED:-42}"
DEVICE="${DEVICE:-cpu}"
N_SPLITS="${N_SPLITS:-5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

CMD=(
  python3 -u "${SCRIPT}"
  --features "${FEATURES}"
  --target "${TARGET}"
  --scope "${SCOPE}"
  --time-average "${TIME_AVERAGE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --model "${MODEL}"
  --n-repeats "${N_REPEATS}"
  --init-seed "${INIT_SEED}"
  --device "${DEVICE}"
  --n-splits "${N_SPLITS}"
  --split-seed "${SPLIT_SEED}"
  --run-name "${RUN_NAME}"
)
[[ -n "${P_ACTIVE}" ]] && CMD+=(--p-active "${P_ACTIVE}")
[[ -n "${ABS_THRESHOLD}" ]] && CMD+=(--abs-threshold "${ABS_THRESHOLD}")
[[ "${MICE}" != "None" && -n "${MICE}" ]] && CMD+=(--mice "${MICE}")
[[ -n "${CACHE_DIR}" ]] && CMD+=(--cache-dir "${CACHE_DIR}")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS}" ]] && CMD+=(${EXTRA_ARGS})

echo "Job: ${SLURM_JOB_ID:-N/A} on $(hostname)"
echo "${CMD[*]}"
"${CMD[@]}"
