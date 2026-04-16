#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=grid-activation
#SBATCH --partition=GENOA
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-grid-activation-%j.out
#SBATCH --error=logs/slurm-grid-activation-%j.err

# ============================================================================
# Compute 3D grid activations from Sensorium responses (SLURM).
#
# Runs scripts/compute_grid_activation.py with optional per-run overrides.
#
# Usage examples:
#   sbatch scripts/compute_grid_activation.sh
#
#   sbatch --export=MICE=dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce \
#          scripts/compute_grid_activation.sh
#
#   sbatch --export=ROOT_OUTPUT=/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax_new,SKIP_EXISTING=true \
#          scripts/compute_grid_activation.sh
#
#   sbatch --export=COMPARE_REFERENCE_ROOT=/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax \
#          scripts/compute_grid_activation.sh
# ============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/compute_grid_activation.py"
VENV_DIR="${PROJECT_DIR}/.venv"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium/data/sensorium_all_2023}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"
ROOT_OUTPUT="${ROOT_OUTPUT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"

# Optional controls
MICE="${MICE:-}"
NORMALIZATION="${NORMALIZATION:-by_minmax}"
NUM_GRID="${NUM_GRID:-15 15 10}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
COMPARE_REFERENCE_ROOT="${COMPARE_REFERENCE_ROOT:-}"
FORTRAN_SOURCE="${FORTRAN_SOURCE:-}"

mkdir -p "${PROJECT_DIR}/logs"

# --- Environment -------------------------------------------------------------
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE:-unknown}"
echo "Partition: ${SLURM_PARTITION:-unknown}"
echo "Python: $(which python3)"
echo "============================================"
echo "Script: ${SCRIPT}"
echo "Data root: ${DATA_ROOT}"
echo "Meta root: ${META_ROOT}"
echo "Root output: ${ROOT_OUTPUT}"
echo "Mice: ${MICE:-<all discovered>}"
echo "Normalization: ${NORMALIZATION}"
echo "Num grid: ${NUM_GRID}"
echo "N workers: ${N_WORKERS}"
echo "Skip existing: ${SKIP_EXISTING}"
if [[ -n "${COMPARE_REFERENCE_ROOT}" ]]; then
  echo "Compare reference root: ${COMPARE_REFERENCE_ROOT}"
else
  echo "Compare reference root: <disabled>"
fi
if [[ -n "${FORTRAN_SOURCE}" ]]; then
  echo "Fortran source override: ${FORTRAN_SOURCE}"
else
  echo "Fortran source: <python script default>"
fi
echo "============================================"

# --- Build command -----------------------------------------------------------
CMD=(
  python3 -u "${SCRIPT}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --root-output "${ROOT_OUTPUT}"
  --normalization "${NORMALIZATION}"
  --num-grid ${NUM_GRID}
  --n-workers "${N_WORKERS}"
)

if [[ -n "${MICE}" ]]; then
  CMD+=(--mice "${MICE}")
fi

if [[ -n "${COMPARE_REFERENCE_ROOT}" ]]; then
  CMD+=(--compare-reference-root "${COMPARE_REFERENCE_ROOT}")
fi

if [[ -n "${FORTRAN_SOURCE}" ]]; then
  CMD+=(--fortran-source "${FORTRAN_SOURCE}")
fi

SKIP_EXISTING_NORM="$(echo "${SKIP_EXISTING}" | tr '[:upper:]' '[:lower:]')"
if [[ "${SKIP_EXISTING_NORM}" == "true" || "${SKIP_EXISTING_NORM}" == "1" || "${SKIP_EXISTING_NORM}" == "yes" ]]; then
  CMD+=(--skip-existing)
fi

echo ""
echo "Running command:"
printf '  %q' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "Root output: ${ROOT_OUTPUT}"
echo "Summary file: ${ROOT_OUTPUT}/compute_grid_activation_summary.json"
echo "============================================"

exit ${EXIT_CODE}
