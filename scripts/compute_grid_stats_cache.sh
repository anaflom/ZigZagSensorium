#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-grid-stats-cache
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-grid-stats-cache-%j.out
#SBATCH --error=logs/slurm-zz-grid-stats-cache-%j.err

# ==========================================================================
# Precompute per-frame grid statistics (moments, active fraction) for
# classify_grid_stats.py.  Writes into <data-root>/<mouse>/cache/.
#
#   sbatch scripts/compute_grid_stats_cache.sh
#   MICE=<m1>,<m2> P_ACTIVE=30,50,70 sbatch scripts/compute_grid_stats_cache.sh
# ==========================================================================

set -euo pipefail

PROJECT_DIR="/u/area/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/compute_grid_stats_cache.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"

MICE="${MICE:-None}"
CACHE_DIR="${CACHE_DIR:-}"
MIN_DENSITY="${MIN_DENSITY:-0.5}"
P_ACTIVE="${P_ACTIVE:-30,50,70}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
ABS_THRESHOLDS="${ABS_THRESHOLDS:-}"  # e.g. 0 (fixed zigzag thresholds for ActiveFraction)
NUM_WORKERS="${NUM_WORKERS:-32}"
SKIP_MOMENTS="${SKIP_MOMENTS:-false}"  # true: only write ActiveFraction caches

mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

CMD=(
  python3 -u "${SCRIPT}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --min-density "${MIN_DENSITY}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --abs-thresholds "${ABS_THRESHOLDS}"
  --num-workers "${NUM_WORKERS}"
  --skip-moments "${SKIP_MOMENTS}"
)
[[ "${MICE}" != "None" && -n "${MICE}" ]] && CMD+=(--mice "${MICE}")
[[ -n "${CACHE_DIR}" ]] && CMD+=(--cache-dir "${CACHE_DIR}")

echo "Job: ${SLURM_JOB_ID:-N/A} on $(hostname)"
echo "${CMD[*]}"
"${CMD[@]}"
