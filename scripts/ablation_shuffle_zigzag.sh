#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-ablation-shuffle-zigzag
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-zz-ablation-shuffle-zigzag-%j.out
#SBATCH --error=logs/slurm-zz-ablation-shuffle-zigzag-%j.err

# ==========================================================================
# Shuffle ablation on zigzag vectorizations of shuffled grids.
# Shuffles must exist (scripts/generate_trials_ablation_shuffle.sh).
#
#   SCOPE=within_mouse SHUFFLE_TYPE=time N_SHUFFLES=5 sbatch scripts/ablation_shuffle_zigzag.sh
#
# Results: results/ablation_shuffle/zigzag/<VECTORIZATION>/<MODEL>/<SHUFFLE_TYPE>-<mode>/
#          p<P>-<per-trial|global>/<TARGET>_<SCOPE>/<run>/shuffle-NNNN/
# ==========================================================================

set -euo pipefail

PROJECT_DIR="/u/area/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/ablation_shuffle_zigzag.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"

TARGET="${TARGET:-video_label}"
SCOPE="${SCOPE:?Set SCOPE=within_mouse|cross_mouse}"
MICE="${MICE:-None}"

VECTORIZATION="${VECTORIZATION:-Turnover}"
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
CACHE_DIR="${CACHE_DIR:-}"

SHUFFLE_TYPE="${SHUFFLE_TYPE:-time}"
DIFFERENT_SHUFFLE_PER_TRIAL="${DIFFERENT_SHUFFLE_PER_TRIAL:-true}"
N_SHUFFLES="${N_SHUFFLES:-5}"
SHUFFLE_IDS="${SHUFFLE_IDS:-}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
GENERATION_CLIP_FRAMES="${GENERATION_CLIP_FRAMES:-240}"

MODEL="${MODEL:-logreg}"
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
  --target "${TARGET}"
  --scope "${SCOPE}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --vectorization "${VECTORIZATION}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --shuffle-type "${SHUFFLE_TYPE}"
  --different-shuffle-per-trial "${DIFFERENT_SHUFFLE_PER_TRIAL}"
  --shuffle-seed "${SHUFFLE_SEED}"
  --generation-clip-frames "${GENERATION_CLIP_FRAMES}"
  --model "${MODEL}"
  --device "${DEVICE}"
  --n-splits "${N_SPLITS}"
  --split-seed "${SPLIT_SEED}"
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
