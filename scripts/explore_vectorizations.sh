#!/bin/bash
#SBATCH --job-name=zz-vectorizations
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-zz-vectorizations-%j.out
#SBATCH --error=logs/slurm-zz-vectorizations-%j.err

# ============================================================================
# Explore vectorizations from zigzag persistence barcodes (headless, SLURM).
#
# Runs scripts/explore_vectorizations.py and saves:
#   - figures (PNG)
#   - run log (captured stdout/stderr)
#   - metrics summary JSON
#
# Usage examples:
#   sbatch scripts/explore_vectorizations.sh
#
#   sbatch --export=P_ACTIVE=30,PER_TRIAL_THRESH=true \
#          scripts/explore_vectorizations.sh
#
#   sbatch --export=REF_MOUSE=dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce,\
#MOUSE_2=dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce,\
#CLIP_FRAMES=None,SKIP_SECTIONS=6,MAX_TRIALS=None \
#          scripts/explore_vectorizations.sh
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/explore_vectorizations.py"
VENV_DIR="${PROJECT_DIR}/.venv"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-false}"

PER_TRIAL_THRESH_NORM="$(echo "${PER_TRIAL_THRESH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PER_TRIAL_THRESH_NORM}" == "true" ]]; then
  OUTPUT_SUFFIX="per-trial"
else
  OUTPUT_SUFFIX="global"
fi

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/vectorizations/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"

# Optional inputs accepted by the Python script.
# Use literal "None" to trigger auto behavior.
REF_MOUSE="${REF_MOUSE:-None}"
MOUSE_2="${MOUSE_2:-None}"
CLIP_FRAMES="${CLIP_FRAMES:-None}"

# Comma-separated subset of: 3,4,5,6,7,7b,8 (section 2 always runs)
SKIP_SECTIONS="${SKIP_SECTIONS:-}"

# Optional cap for quick tests; "None" disables cap.
MAX_TRIALS="${MAX_TRIALS:-None}"

# Keep output folder unique per run.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_thr-${PER_TRIAL_THRESH}_ref-${REF_MOUSE}_m2-${MOUSE_2}_clip-${CLIP_FRAMES}_${RUN_TS}"
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
echo "CPUs: ${SLURM_CPUS_PER_TASK:-32}"
echo "Memory: ${SLURM_MEM_PER_NODE:-200G}"
echo "Python: $(which python3)"
echo "============================================"
echo "Script: ${SCRIPT}"
echo "Data root: ${DATA_ROOT}"
echo "Meta root: ${META_ROOT}"
echo "Output dir: ${OUT_DIR}"
echo "P_ACTIVE: ${P_ACTIVE}"
echo "PER_TRIAL_THRESH: ${PER_TRIAL_THRESH}"
echo "REF_MOUSE: ${REF_MOUSE}"
echo "MOUSE_2: ${MOUSE_2}"
echo "CLIP_FRAMES: ${CLIP_FRAMES}"
echo "SKIP_SECTIONS: ${SKIP_SECTIONS}"
echo "MAX_TRIALS: ${MAX_TRIALS}"
echo "============================================"

# --- Build command -----------------------------------------------------------
CMD=(
  python3 -u "${SCRIPT}"
  --output-folder "${OUT_DIR}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --ref-mouse "${REF_MOUSE}"
  --mouse-2 "${MOUSE_2}"
  --clip-frames "${CLIP_FRAMES}"
)

if [[ -n "${SKIP_SECTIONS}" ]]; then
  CMD+=(--skip-sections "${SKIP_SECTIONS}")
fi

if [[ "${MAX_TRIALS}" != "None" && "${MAX_TRIALS}" != "none" && "${MAX_TRIALS}" != "" ]]; then
  CMD+=(--max-trials "${MAX_TRIALS}")
fi

echo "Running command:"
printf '  %q' "${CMD[@]}"
echo

echo ""
echo "Starting vectorization exploration..."
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "Run output folder: ${OUT_DIR}"
echo "============================================"

exit ${EXIT_CODE}
