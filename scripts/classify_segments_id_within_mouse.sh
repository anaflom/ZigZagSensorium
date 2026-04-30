#!/bin/bash

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=zz-within-mouse-seg-id
#SBATCH --partition=GPU
#SBATCH --account=MDMC
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm-zz-within-mouse-seg-id-%j.out
#SBATCH --error=logs/slurm-zz-within-mouse-seg-id-%j.err

# ============================================================================
# Within-mouse segment-ID decoding from Turnover sub-vectors + grid segments.
#
# Labels:
#   NaturalImages, PinkNoise, RandomDots, Gabor, GaussianDot
#
# Segment classes:
#   video JSON segment IDs
#
# CV:
#   Trial-level for both branches by default (Leave-One-Group-Out on trial_id).
#
#   Available options (passed through to classify_segments_within_mouse_id.py):
#   - loso: one held-out segment sample per fold (slowest, most exhaustive)
#   - logo: one held-out trial per fold (trial-level, default here)
#   - groupkfold: k-fold split by trial_id (fast, trial-level)
#   - kfold: k-fold split by segment samples (fastest, not trial-level)
#
#   Note:
#   --cv-n-splits-* is used only by groupkfold and kfold.
# ============================================================================

set -euo pipefail

PROJECT_DIR="/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium"
SCRIPT="${PROJECT_DIR}/scripts/classify_segments_id_within_mouse.py"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-gpu}"

DATA_ROOT="${DATA_ROOT:-/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_minmax}"
META_ROOT="${META_ROOT:-/u/mdmc/anaflom/projects_mdmc/sensorium/metadata}"

# Mouse selection
#DEFAULT_MICE="dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce,dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce"
#MICE="${MICE:-${DEFAULT_MICE}}"
MICE="${MICE:-dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce}"

P_ACTIVE="${P_ACTIVE:-30}"
PER_TRIAL_THRESH="${PER_TRIAL_THRESH:-true}"
CLIP_FRAMES="${CLIP_FRAMES:-None}"
GRID_SUBDIR="${GRID_SUBDIR:-trials_grid}"
MAX_TRIALS="${MAX_TRIALS:-None}"

CACHE_DIR="${CACHE_DIR:-}"

BATCH_SIZE_GRID="${BATCH_SIZE_GRID:-16}"
EPOCHS_CNN3D="${EPOCHS_CNN3D:-40}"
LR_CNN3D="${LR_CNN3D:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS_DL="${NUM_WORKERS_DL:-0}"

# CV controls per model branch.
# Defaults below are trial-level for both LogReg and 3D-CNN.
CV_SCHEME_LOGREG="${CV_SCHEME_LOGREG:-logo}"
CV_SCHEME_CNN3D="${CV_SCHEME_CNN3D:-logo}"
CV_N_SPLITS_LOGREG="${CV_N_SPLITS_LOGREG:-5}"
CV_N_SPLITS_CNN3D="${CV_N_SPLITS_CNN3D:-5}"

SEG_LEN_NATURALIMAGES="${SEG_LEN_NATURALIMAGES:-15}"
SEG_LEN_PINKNOISE="${SEG_LEN_PINKNOISE:-27}"
SEG_LEN_RANDOMDOTS="${SEG_LEN_RANDOMDOTS:-60}"
SEG_LEN_GABOR="${SEG_LEN_GABOR:-25}"
SEG_LEN_GAUSSIANDOT="${SEG_LEN_GAUSSIANDOT:-9}"

PER_TRIAL_THRESH_NORM="$(echo "${PER_TRIAL_THRESH}" | tr '[:upper:]' '[:lower:]')"
if [[ "${PER_TRIAL_THRESH_NORM}" == "true" ]]; then
  OUTPUT_SUFFIX="per-trial"
else
  OUTPUT_SUFFIX="global"
fi

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/results/within_mouse_segment_id_decoding/p${P_ACTIVE}-${OUTPUT_SUFFIX}}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="p${P_ACTIVE}_thr-${PER_TRIAL_THRESH}_method-Turnover_segment-id_logreg-${CV_SCHEME_LOGREG}_cnn-${CV_SCHEME_CNN3D}_${RUN_TS}"
RUN_TAG_SAFE="$(echo "${RUN_TAG}" | sed 's/[^a-zA-Z0-9._-]/_/g')"
OUT_DIR="${OUTPUT_BASE}/${RUN_TAG_SAFE}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_BASE}"

source "${VENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo "Venv: ${VENV_DIR}"
echo "Script: ${SCRIPT}"
echo "Data root: ${DATA_ROOT}"
echo "Meta root: ${META_ROOT}"
echo "Output dir: ${OUT_DIR}"
echo "CV LogReg: ${CV_SCHEME_LOGREG} (n_splits=${CV_N_SPLITS_LOGREG})"
echo "CV 3D-CNN: ${CV_SCHEME_CNN3D} (n_splits=${CV_N_SPLITS_CNN3D})"
if [[ -n "${CACHE_DIR}" ]]; then
  echo "Cache dir override: ${CACHE_DIR}"
else
  echo "Cache dir: <data-root>/<mouse>/cache"
fi
echo "============================================"

CMD=(
  python3 -u "${SCRIPT}"
  --output-folder "${OUT_DIR}"
  --data-root "${DATA_ROOT}"
  --meta-root "${META_ROOT}"
  --p-active "${P_ACTIVE}"
  --per-trial-thresh "${PER_TRIAL_THRESH}"
  --grid-subdir "${GRID_SUBDIR}"
  --batch-size-grid "${BATCH_SIZE_GRID}"
  --epochs-cnn3d "${EPOCHS_CNN3D}"
  --lr-cnn3d "${LR_CNN3D}"
  --weight-decay "${WEIGHT_DECAY}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num-workers-dl "${NUM_WORKERS_DL}"
  --cv-scheme-logreg "${CV_SCHEME_LOGREG}"
  --cv-scheme-cnn3d "${CV_SCHEME_CNN3D}"
  --cv-n-splits-logreg "${CV_N_SPLITS_LOGREG}"
  --cv-n-splits-cnn3d "${CV_N_SPLITS_CNN3D}"
  --seg-len-naturalimages "${SEG_LEN_NATURALIMAGES}"
  --seg-len-pinknoise "${SEG_LEN_PINKNOISE}"
  --seg-len-randomdots "${SEG_LEN_RANDOMDOTS}"
  --seg-len-gabor "${SEG_LEN_GABOR}"
  --seg-len-gaussiandot "${SEG_LEN_GAUSSIANDOT}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  CMD+=(--cache-dir "${CACHE_DIR}")
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

"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished with exit code: ${EXIT_CODE}"
echo "Run output folder: ${OUT_DIR}"
echo "  JSON: ${OUT_DIR}/within_mouse_segment_id_metrics.json"
echo "  CSV:  ${OUT_DIR}/within_mouse_segment_id_metrics.csv"
echo "  Skip: ${OUT_DIR}/within_mouse_segment_id_skips.csv"
echo "  Log:  ${OUT_DIR}/logs/run.log"
echo "============================================"

exit ${EXIT_CODE}
