#!/bin/bash
#SBATCH --job-name=zz-classify-within-mouse
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-zz-classify-within-mouse-%j.out
#SBATCH --error=logs/slurm-zz-classify-within-mouse-%j.err

# ============================================================================
# Within-mouse trial classification using zigzag vectorizations (SLURM).
#
# Runs scripts/classify_trials_within_mouse.py for all available mice.
# Automatically skips mice with insufficient labels for cross-validation.
#
# This script:
#   - Loads/computes vectorizations with caching (avoids redundant computation)
#   - Performs stratified K-fold within-mouse classification
#   - Generates per-mouse accuracy/F1 metrics and confusion matrices
#   - Saves results as JSON, CSV, and PNG figures
#   - Archives all mice results in a single unified output
#
# Note: Mice with only one stimulus class are automatically skipped with a log
#       message ("Not enough samples per class for CV").
#
# Usage examples:
#   sbatch scripts/classify_trials_within_mouse.sh
#
#   sbatch --export=P_ACTIVE=30,PER_TRIAL_THRESH=true,N_SPLITS=5 \
#          scripts/classify_trials_within_mouse.sh
#
#   sbatch --export=METHOD=PersistenceImage,CLIP_FRAMES=240,N_SPLITS=3,MAX_TRIALS=100 \
#          scripts/classify_trials_within_mouse.sh
#
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_trials_within_mouse.py"
VENV_DIR="${PROJECT_DIR}/.venv"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

# Vectorization parameters
P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
METHOD="${METHOD:-Turnover}"
CLIP_FRAMES="${CLIP_FRAMES:-None}"

# Classification parameters
N_SPLITS="${N_SPLITS:-5}"
MAX_TRIALS="${MAX_TRIALS:-None}"
MICE="${MICE:-None}"

# Normalize boolean string for output folder naming
PER_TRIAL_THRESH_NORM="$(echo "${PER_TRIAL_THRESH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PER_TRIAL_THRESH_NORM}" == "true" ]]; then
  OUTPUT_SUFFIX="per-trial"
else
  OUTPUT_SUFFIX="global"
fi

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/within_mouse_classification/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"

# Optional cache directory override for vectorizations.
# If unset, Python defaults to: <data-root>/<mouse>/cache
CACHE_DIR="${CACHE_DIR:-}"

# Keep output folder unique per run
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_thr-${PER_TRIAL_THRESH}_method-${METHOD}_clip-${CLIP_FRAMES}_cv${N_SPLITS}_${RUN_TS}"
RUN_TAG_SAFE="$(echo "${RUN_TAG}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
OUT_DIR="${OUTPUT_BASE}/${RUN_TAG_SAFE}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_BASE}"

# --- Environment -------------------------------------------------------------
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}"
echo "Partition: ${SLURM_PARTITION}"
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
echo "Vectorization method: ${METHOD}"
echo "P_ACTIVE: ${P_ACTIVE}"
echo "PER_TRIAL_THRESH: ${PER_TRIAL_THRESH}"
echo "CLIP_FRAMES: ${CLIP_FRAMES}"
echo "N_SPLITS (CV folds): ${N_SPLITS}"
echo "MAX_TRIALS: ${MAX_TRIALS}"
echo "MICE: ${MICE}"
echo "============================================"

# --- Build command -----------------------------------------------------------
CMD=(
  python3 -u "${SCRIPT}"
  --output-folder "${OUT_DIR}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --method "${METHOD}"
  --n-splits "${N_SPLITS}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  CMD+=(--cache-dir "${CACHE_DIR}")
fi

# Optional parameters
if [[ "${CLIP_FRAMES}" != "None" && "${CLIP_FRAMES}" != "none" && "${CLIP_FRAMES}" != "" ]]; then
  CMD+=(--clip-frames "${CLIP_FRAMES}")
fi

if [[ "${MAX_TRIALS}" != "None" && "${MAX_TRIALS}" != "none" && "${MAX_TRIALS}" != "" ]]; then
  CMD+=(--max-trials "${MAX_TRIALS}")
fi

if [[ "${MICE}" != "None" && "${MICE}" != "none" && "${MICE}" != "" ]]; then
  CMD+=(--mice "${MICE}")
fi

echo ""
echo "Running command:"
printf '  %q' "${CMD[@]}"
echo ""
echo ""

echo "Starting within-mouse classification for all available mice..."
echo "(Mice with only one stimulus class will be automatically skipped)"
echo ""

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "Run output folder: ${OUT_DIR}"
echo "Results summary:"
if [[ -f "${OUT_DIR}/within_mouse_metrics.json" ]]; then
  echo "  JSON: ${OUT_DIR}/within_mouse_metrics.json"
  echo "  CSV:  ${OUT_DIR}/within_mouse_metrics.csv"
  echo "  Figures:"
  echo "    - ${OUT_DIR}/figures/01_within_mouse_scores.png"
  echo "    - ${OUT_DIR}/figures/02_confusion_matrices.png"
  echo "  Log:  ${OUT_DIR}/logs/run.log"
fi
if [[ -n "${CACHE_DIR}" ]]; then
  echo "Vectorization cache: ${CACHE_DIR}"
else
  echo "Vectorization cache: <data-root>/<mouse>/cache"
fi
echo "============================================"

exit ${EXIT_CODE}
