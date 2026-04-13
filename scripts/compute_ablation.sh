#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium/logs/ablation_%j.out
#SBATCH --error=/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium/logs/ablation_%j.err

set -euo pipefail

WORKDIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
VENV_DIR="${WORKDIR}/.venv"

CONDA_DIR="/u/mdmc/anaflom/miniconda3"
CONDA_ENV="topofisher_gpu"

cd "$WORKDIR"
mkdir -p logs results

# Activate virtual environment (venv needs Python 3.13)
source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Python:       $(which python3)"
echo "============================================"


# Run ablation on all mice (pass --mouse 0 1 2 to select specific mice)
python3 -u scripts/compute_ablation.py "$@"

echo "============================================"
echo "Finished:     $(date)"
echo "============================================"
