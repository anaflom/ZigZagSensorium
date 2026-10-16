#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=scene-cuts
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-scene-cuts-%j.out
#SBATCH --error=logs/slurm-scene-cuts-%j.err

# ============================================================================
# Scene-cut detection for the NaturalVideo clips (PySceneDetect AdaptiveDetector /
# ContentDetector and TransNetV2) and comparison with the metadata cuts (SLURM).
#
# Usage:
#   sbatch scripts/detect_scene_cuts.sh
#   sbatch --export=ALL,TOLERANCE=3,REVIEW_SAMPLE=100 scripts/detect_scene_cuts.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT="${PROJECT_DIR}/scripts/detect_scene_cuts.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-genoa}"

RAW_ROOT="${RAW_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium/data/sensorium_all_2023}"
META_ROOT="${META_ROOT:-/u/area/anaflom/projects_mdmc/sensorium/metadata}"
OUTPUT_DIR="${OUTPUT_DIR:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/scene_cuts}"
TOLERANCE="${TOLERANCE:-2}"
REVIEW_SAMPLE="${REVIEW_SAMPLE:-60}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"

mkdir -p "${PROJECT_DIR}/logs"
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
# utils.py imports zztop; fall back to the local checkout if the installed one is not importable
export PYTHONPATH="${PROJECT_DIR}/zz-top:${PYTHONPATH:-}"
# one torch thread per worker process
export OMP_NUM_THREADS=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-N/A}   Node: $(hostname)"
echo "Raw root: ${RAW_ROOT}"
echo "Meta root: ${META_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "============================================"

python3 -u "${SCRIPT}" \
  --raw-root "${RAW_ROOT}" \
  --meta-root "${META_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --tolerance "${TOLERANCE}" \
  --review-sample "${REVIEW_SAMPLE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-threads 1
