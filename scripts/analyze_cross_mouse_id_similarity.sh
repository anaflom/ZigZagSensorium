#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-cross-mouse-id-sim
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-zz-cross-mouse-id-sim-%j.out
#SBATCH --error=logs/slurm-zz-cross-mouse-id-sim-%j.err

# ============================================================================
# Cross-mouse ID similarity analysis using zigzag vectorizations (SLURM).
#
# For each pair of mice sharing common stimulus IDs (after eligibility +
# repetition filtering):
#   1. Vectorize zigzag barcodes
#   2. Preprocess: StandardScaler + L2 normalization + PCA (jointly per label)
#   3. Compute within-m1, within-m2, and cross-mouse distance matrices
#   4. Plot combined heatmap + violin/boxplot per label
#   5. Export distances.csv + summary.csv
#
# Usage examples:
#   sbatch scripts/analyze_cross_mouse_id_similarity.sh
#
#   sbatch --export=MICE=dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce,VECTORIZATION_METHOD=Turnover \
#          scripts/analyze_cross_mouse_id_similarity.sh
#
#   sbatch --export=MIN_ID_REPETITIONS=5,N_PCA_COMPONENTS=15 \
#          scripts/analyze_cross_mouse_id_similarity.sh
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/analyze_cross_mouse_id_similarity.py"
VENV_DIR="${PROJECT_DIR}/.venv-genoa"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

# Core parameters
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
VECTORIZATION_METHOD="${VECTORIZATION_METHOD:-Turnover}"
MICE="${MICE:-None}"
CLIP_FRAMES="${CLIP_FRAMES:-240}"
MAX_TRIALS="${MAX_TRIALS:-None}"

# Analysis parameters
MIN_ID_REPETITIONS="${MIN_ID_REPETITIONS:-7}"
N_PCA_COMPONENTS="${N_PCA_COMPONENTS:-10}"
SEED="${SEED:-42}"

# Cache
CACHE_DIR="${CACHE_DIR:-}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-false}"

PER_TRIAL_THRESH_NORM="$(echo "${PER_TRIAL_THRESH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PER_TRIAL_THRESH_NORM}" == "true" ]]; then
  OUTPUT_SUFFIX="per-trial"
else
  OUTPUT_SUFFIX="global"
fi

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/cross_mouse_id_similarity/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_method-${VECTORIZATION_METHOD}_minrep-${MIN_ID_REPETITIONS}_pca-${N_PCA_COMPONENTS}_${RUN_TS}"
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
echo "P_ACTIVE: ${P_ACTIVE}"
echo "PER_TRIAL_THRESH: ${PER_TRIAL_THRESH}"
echo "VECTORIZATION_METHOD: ${VECTORIZATION_METHOD}"
echo "MICE: ${MICE}"
echo "CLIP_FRAMES: ${CLIP_FRAMES}"
echo "MAX_TRIALS: ${MAX_TRIALS}"
echo "FORCE_RECOMPUTE: ${FORCE_RECOMPUTE}"
echo "============================================"
echo "MIN_ID_REPETITIONS: ${MIN_ID_REPETITIONS}"
echo "N_PCA_COMPONENTS: ${N_PCA_COMPONENTS}"
echo "SEED: ${SEED}"
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
  --min-id-repetitions "${MIN_ID_REPETITIONS}"
  --n-pca-components "${N_PCA_COMPONENTS}"
  --seed "${SEED}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  CMD+=(--cache-dir "${CACHE_DIR}")
fi

if [[ "${FORCE_RECOMPUTE}" == "true" || "${FORCE_RECOMPUTE}" == "1" ]]; then
  CMD+=(--force-recompute true)
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

echo "Starting cross-mouse ID similarity analysis..."
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "SUCCESS (exit code: ${EXIT_CODE})"
  echo "Results saved to: ${OUT_DIR}"
  if [[ -f "${OUT_DIR}/summary.csv" ]]; then
    echo "Summary CSV: ${OUT_DIR}/summary.csv"
  fi
else
  echo "FAILED (exit code: ${EXIT_CODE})"
fi
echo "============================================"

exit "${EXIT_CODE}"
