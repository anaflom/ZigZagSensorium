#!/bin/bash
# Relaunch the trial-mean grid jobs cancelled on 2026-10-21 (CephFS read hangs).
# Each job depends only on what it needs:
#   - Turnover cache at threshold 0 for the remaining mice (existing caches are skipped)
#   - Turnover threshold-0 classification + incremental analysis at threshold 0: after that cache
#   - incremental analysis at P_ACTIVE 30/50/70: now (caches exist)
#   - 3D-CNN within-mouse video ID (one job per mouse), within-mouse segment ID for the
#     4 missing mice, cross-mouse segment ID: now; same run name as the original analysis
#
# Usage: bash scripts/run_relaunch_trialmean.sh   (from the project root)

set -euo pipefail

export DATA_ROOT="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_trialmean"
PIPELINE_RUN="20261020_111052"  # run name of run_trialmean_pipeline.sh
SEGMENT_MICE_TODO=(
  dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce
)

# 1. Turnover cache, threshold 0 (remaining mice)
VEC=$(ABS_THRESHOLD=0 CLIP_FRAMES=None sbatch --parsable scripts/compute_vectorization_cache.sh)
echo "Turnover cache abs0: ${VEC}"

# 2. Turnover threshold-0 classification
for TARGET in video_label video_id segment_id; do
  for SCOPE in within_mouse cross_mouse; do
    TARGET=$TARGET SCOPE=$SCOPE ABS_THRESHOLD=0 RUN_NAME=$PIPELINE_RUN \
      sbatch --parsable --dependency="afterok:${VEC}" scripts/classify_zigzag.sh > /dev/null
  done
done
echo "Turnover abs0 classification: 6 jobs after ${VEC}"

# 3. Incremental analysis
GRID=trialmean THRESHOLDS="p30 p50 p70" bash scripts/run_incremental.sh
GRID=trialmean THRESHOLDS="abs0" DEPENDENCY="${VEC}" bash scripts/run_incremental.sh

# 4. 3D-CNN on the grid (one repeat)
for d in "${DATA_ROOT}"/dynamic*; do
  m=$(basename "$d")
  TARGET=video_id SCOPE=within_mouse MICE=$m PART="${m%%-Video*}" N_REPEATS=1 RUN_NAME=$PIPELINE_RUN \
    sbatch --parsable scripts/classify_grid.sh > /dev/null
done
for m in "${SEGMENT_MICE_TODO[@]}"; do
  TARGET=segment_id SCOPE=within_mouse MICE=$m PART="${m%%-Video*}" N_REPEATS=1 RUN_NAME=$PIPELINE_RUN \
    sbatch --parsable scripts/classify_grid.sh > /dev/null
done
TARGET=segment_id SCOPE=cross_mouse N_REPEATS=1 RUN_NAME=$PIPELINE_RUN sbatch --parsable scripts/classify_grid.sh > /dev/null
echo "3D-CNN: 10 + 4 + 1 GPU jobs"
