#!/bin/bash
#SBATCH --job-name=zz-neural
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/slurm-zz-neural-%A_%a.out
#SBATCH --error=logs/slurm-zz-neural-%A_%a.err

# ============================================================================
# Zigzag persistence on 3D neural grids — Slurm array job
#
# Runs one mouse per array task (10 mice total, array indices 0–9).
# Each task uses multiprocessing across 32 CPUs on a GENOA node.
#
# Usage:
#   sbatch run_zigzag.sbatch                   # default p_active=30
#   sbatch --export=P_ACTIVE=40 run_zigzag.sbatch  # override p_active
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
DATA_ROOT="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax"
VENV_DIR="${PROJECT_DIR}/.venv"
SCRIPT="${PROJECT_DIR}/scripts/compute_zigzag.py"

# Threshold percentile (can be overridden via --export=P_ACTIVE=...)
P_ACTIVE="${P_ACTIVE:-30}"

# Set p active per trial to True to compute separate thresholds for each trial (overrides p_active)
P_ACTIVE_PER_TRIAL="${P_ACTIVE_PER_TRIAL:-True}"

# Number of parallel workers (match cpus-per-task)
N_WORKERS="${SLURM_CPUS_PER_TASK:-32}"

# --- Mouse list (sorted, must match array size) ------------------------------
mapfile -t MICE < <(ls -1d "${DATA_ROOT}"/dynamic* | sort)

N_MICE=${#MICE[@]}
echo "Found ${N_MICE} mice in ${DATA_ROOT}"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_MICE}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= N_MICE=${N_MICE}"
    exit 1
fi

MOUSE_DIR="${MICE[${SLURM_ARRAY_TASK_ID}]}"
MOUSE_NAME=$(basename "${MOUSE_DIR}")
OUT_DIR=None

echo "============================================"
echo "Job: ${SLURM_JOB_ID}, Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Mouse: ${MOUSE_NAME}"
echo "p_active: ${P_ACTIVE}"
echo "Workers: ${N_WORKERS}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE:-100G}"
echo "============================================"

# --- Activate virtual environment --------------------------------------------
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
echo "Python: $(which python3)"
echo "zztop version: $(python3 -c 'import zztop; print(zztop.__version__)' 2>/dev/null || echo 'unknown')"

# --- Run zigzag persistence --------------------------------------------------
echo ""
echo "Starting zigzag persistence computation..."
echo "  Mouse dir: ${MOUSE_DIR}"
echo "  Output dir: ${OUT_DIR}"
echo "  p_active: ${P_ACTIVE}"
echo "  n_workers: ${N_WORKERS}"
echo ""

python3 -u "${SCRIPT}" \
    --mouse-dir "${MOUSE_DIR}" \
    --p-active "${P_ACTIVE}" \
    --p-active-per-trial "${P_ACTIVE_PER_TRIAL}" \
    --n-workers "${N_WORKERS}" \
    --n-threshold-samples 20 \
    --max-dim 2 \
    --skip-existing

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}