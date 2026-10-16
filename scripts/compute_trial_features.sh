#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=trial-features
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-trial-features-%j.out
#SBATCH --error=logs/slurm-trial-features-%j.err

# ============================================================================
# Per-frame video (luminance ON/OFF, contrast change, optical-flow speed, structure,
# local luminance/contrast change, spatial-frequency bands) and
# behaviour (pupil, running) features for every valid trial (SLURM).
#
# Usage:
#   sbatch scripts/compute_trial_features.sh
#   sbatch --export=MICE=dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce scripts/compute_trial_features.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT="${PROJECT_DIR}/scripts/compute_trial_features.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

RAW_ROOT="${RAW_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium/data/sensorium_all_2023}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"
OUTPUT_DIR="${OUTPUT_DIR:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/trial_features}"
MICE="${MICE:-None}"
FLOW_RADIUS="${FLOW_RADIUS:-4}"
FLOW_NUM_WARP="${FLOW_NUM_WARP:-10}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
# utils.py imports zztop; fall back to the local checkout if the installed one is not importable
export PYTHONPATH="${PROJECT_DIR}/zz-top:${PYTHONPATH:-}"

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-N/A}   Node: $(hostname)"
echo "Raw root: ${RAW_ROOT}"
echo "Meta root: ${META_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "============================================"

CMD=(
  python3 -u "${SCRIPT}"
  --raw-root "${RAW_ROOT}"
  --meta-root "${META_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --flow-radius "${FLOW_RADIUS}"
  --flow-num-warp "${FLOW_NUM_WARP}"
  --num-workers "${NUM_WORKERS}"
)
if [[ "${MICE}" != "None" && "${MICE}" != "none" && "${MICE}" != "" ]]; then
  CMD+=(--mice "${MICE}")
fi

"${CMD[@]}"
