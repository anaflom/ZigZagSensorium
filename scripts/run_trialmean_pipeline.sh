#!/bin/bash
# Submit the full analysis on the trial-mean-normalized grid, chained with Slurm
# dependencies (each step starts when its inputs exist):
#
#   1. grid      compute_grid_activation  NORMALIZATION=by_trialmean (one job, ~2 h)
#   2. zigzag    compute_zigzag           threshold 0 and P_ACTIVE 30/50/70 (array over mice)
#   3. caches    compute_vectorization_cache (Turnover, clipnone) per threshold
#                compute_grid_stats_cache (moments + active fraction)
#   4. classify  zigzag (logreg), grid stats (logreg), 3D-CNN (one repeat)
#
# Usage: bash scripts/run_trialmean_pipeline.sh   (from the project root)
#        STEPS="classify" bash scripts/run_trialmean_pipeline.sh   (skip steps whose outputs exist)

set -euo pipefail

GRID_ROOT="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/grid-15x15x10_norm-by_trialmean"
export DATA_ROOT="${GRID_ROOT}"  # data root of every step except the grid computation
RAW_DATA_ROOT="/orfeo/scratch/area/ygardinazzi/sensorium/data/sensorium_all_2023"
STEPS="${STEPS:-grid zigzag caches classify}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
P_LIST=(30 50 70)
TARGETS=(video_label video_id segment_id)
SCOPES=(within_mouse cross_mouse)
SEGMENT_MICE=(
  dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce
)

has() { [[ " ${STEPS} " == *" $1 "* ]]; }
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || true; }
submit() { sbatch --parsable "$@"; }

# 1. Grid -------------------------------------------------------------------
GRID_JOB=""
if has grid; then
  # compute_grid_activation.sh reads the raw Sensorium data from its own DATA_ROOT
  GRID_JOB=$(DATA_ROOT="${RAW_DATA_ROOT}" NORMALIZATION=by_trialmean ROOT_OUTPUT="${GRID_ROOT}" \
    submit scripts/compute_grid_activation.sh)
  echo "grid: ${GRID_JOB}"
fi

# 2. Zigzag (array over the 10 mice), one per threshold ---------------------
declare -A ZZ_JOB
if has zigzag; then
  ZZ_JOB[abs0]=$(THRESHOLD=0 submit $(dep "${GRID_JOB}") scripts/compute_zigzag.sh)
  for p in "${P_LIST[@]}"; do
    ZZ_JOB[p$p]=$(P_ACTIVE=$p THRESHOLD= submit $(dep "${GRID_JOB}") scripts/compute_zigzag.sh)
  done
  echo "zigzag: ${ZZ_JOB[*]}"
fi

# 3. Caches -----------------------------------------------------------------
declare -A VEC_JOB
STATS_JOB=""
if has caches; then
  VEC_JOB[abs0]=$(ABS_THRESHOLD=0 CLIP_FRAMES=None submit $(dep "${ZZ_JOB[abs0]:-}") scripts/compute_vectorization_cache.sh)
  for p in "${P_LIST[@]}"; do
    VEC_JOB[p$p]=$(P_ACTIVE=$p ABS_THRESHOLD= CLIP_FRAMES=None submit $(dep "${ZZ_JOB[p$p]:-}") scripts/compute_vectorization_cache.sh)
  done
  all_zz=$(IFS=:; echo "${ZZ_JOB[*]:-}")
  STATS_JOB=$(P_ACTIVE=30,50,70 ABS_THRESHOLDS=0 submit $(dep "${all_zz}") scripts/compute_grid_stats_cache.sh)
  echo "vectorization caches: ${VEC_JOB[*]}  grid stats cache: ${STATS_JOB}"
fi

# 4. Classification ---------------------------------------------------------
if has classify; then
  n=0
  for TARGET in "${TARGETS[@]}"; do
    for SCOPE in "${SCOPES[@]}"; do
      # Zigzag Turnover, logreg
      TARGET=$TARGET SCOPE=$SCOPE ABS_THRESHOLD=0 RUN_NAME=$RUN_NAME \
        submit $(dep "${VEC_JOB[abs0]:-}") scripts/classify_zigzag.sh > /dev/null; n=$((n + 1))
      for p in "${P_LIST[@]}"; do
        TARGET=$TARGET SCOPE=$SCOPE P_ACTIVE=$p ABS_THRESHOLD= RUN_NAME=$RUN_NAME \
          submit $(dep "${VEC_JOB[p$p]:-}") scripts/classify_zigzag.sh > /dev/null; n=$((n + 1))
      done

      # Grid statistics, logreg (per frame and trial-averaged)
      for TIME_AVERAGE in false true; do
        for FEATURES in moments_std moments_std_density; do
          FEATURES=$FEATURES TARGET=$TARGET SCOPE=$SCOPE TIME_AVERAGE=$TIME_AVERAGE RUN_NAME=$RUN_NAME \
            submit $(dep "${STATS_JOB}") scripts/classify_grid_stats.sh > /dev/null; n=$((n + 1))
        done
        FEATURES=active_fraction ABS_THRESHOLD=0 TARGET=$TARGET SCOPE=$SCOPE TIME_AVERAGE=$TIME_AVERAGE RUN_NAME=$RUN_NAME \
          submit $(dep "${STATS_JOB}") scripts/classify_grid_stats.sh > /dev/null; n=$((n + 1))
        for p in "${P_LIST[@]}"; do
          FEATURES=active_fraction P_ACTIVE=$p ABS_THRESHOLD= TARGET=$TARGET SCOPE=$SCOPE TIME_AVERAGE=$TIME_AVERAGE \
            RUN_NAME=$RUN_NAME submit $(dep "${STATS_JOB}") scripts/classify_grid_stats.sh > /dev/null; n=$((n + 1))
        done
      done

      # 3D-CNN on the grid, one repeat (within-mouse segment_id: one job per mouse)
      if [[ "$TARGET" == "segment_id" && "$SCOPE" == "within_mouse" ]]; then
        for MOUSE in "${SEGMENT_MICE[@]}"; do
          TARGET=$TARGET SCOPE=$SCOPE MICE=$MOUSE PART="${MOUSE%%-Video*}" N_REPEATS=1 RUN_NAME=$RUN_NAME \
            submit $(dep "${GRID_JOB}") scripts/classify_grid.sh > /dev/null; n=$((n + 1))
        done
      else
        TARGET=$TARGET SCOPE=$SCOPE N_REPEATS=1 RUN_NAME=$RUN_NAME \
          submit $(dep "${GRID_JOB}") scripts/classify_grid.sh > /dev/null; n=$((n + 1))
      fi
    done
  done
  echo "classification jobs submitted: ${n} (run name ${RUN_NAME})"
fi
