#!/bin/bash
# Submit, chained with Slurm dependencies:
#   A. zigzag: P_ACTIVE 90 per trial (min-max and trial-mean grids);
#              per-frame P_ACTIVE 50/70/90 (trial-mean grid)
#   B. Turnover caches (full length) for each new zigzag
#   C. active-fraction caches for P_ACTIVE 90 (both grids; moment caches untouched)
#   D. logreg (fixed C) for the overview: Turnover at the new thresholds;
#      active fraction p90 per frame and trial-averaged
#   E. incremental analysis with logreg_cv (tuned C) at every threshold:
#      reruns of the existing thresholds plus the new ones (scripts/run_incremental.sh)
#
# Usage (from the project root):
#   DRY_RUN=true bash scripts/run_perframe_p90_pipeline.sh   # list the jobs, submit nothing
#   bash scripts/run_perframe_p90_pipeline.sh

set -euo pipefail

DERIV="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives"
MINMAX="${DERIV}/grid-15x15x10_norm-by_minmax"
TRIALMEAN="${DERIV}/grid-15x15x10_norm-by_trialmean"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
# Zigzag arrays over the 10 mice, at most 3 tasks (32 CPUs each) running per array at a time
ZZ_ARRAY="${ZZ_ARRAY:-0-9%3}"
export DRY_RUN="${DRY_RUN:-false}"
N=0
FAKE_COUNTER=$(mktemp)  # dry-run job ids (a file, because submit runs inside $(...))
echo 100000 > "${FAKE_COUNTER}"
trap 'rm -f "${FAKE_COUNTER}"' EXIT

submit() {  # submit [sbatch options...] <script>; prints the job id
  N=$((N + 1))
  if [[ "${DRY_RUN}" == "true" ]]; then
    FAKE_ID=$(( $(cat "${FAKE_COUNTER}") + 1 )); echo "${FAKE_ID}" > "${FAKE_COUNTER}"
    echo "  [dry-run] id=${FAKE_ID} ${DATA_ROOT##*/} P=${P_ACTIVE:-} PF=${PER_FRAME:-}${PER_FRAME_THRESH:-} T=${TARGET:-}/${SCOPE:-} F=${FEATURES:-} TA=${TIME_AVERAGE:-} $*" >&2
    echo "${FAKE_ID}"
  else
    sbatch --parsable "$@"
  fi
}
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || true; }

declare -A ZZ VEC
# ---- A + B: zigzag and Turnover caches -------------------------------------
for spec in "minmax p90" "trialmean p90" "trialmean pf50" "trialmean pf70" "trialmean pf90"; do
  read -r grid thr <<< "${spec}"
  root=$([[ "${grid}" == minmax ]] && echo "${MINMAX}" || echo "${TRIALMEAN}")
  if [[ "${thr}" == pf* ]]; then p="${thr#pf}"; pf=true; else p="${thr#p}"; pf=false; fi
  ZZ[${grid}_${thr}]=$(DATA_ROOT="${root}" P_ACTIVE="${p}" PER_FRAME="${pf}" THRESHOLD= submit --array="${ZZ_ARRAY}" scripts/compute_zigzag.sh)
  VEC[${grid}_${thr}]=$(DATA_ROOT="${root}" P_ACTIVE="${p}" PER_FRAME_THRESH="${pf}" ABS_THRESHOLD= CLIP_FRAMES=None \
    submit $(dep "${ZZ[${grid}_${thr}]}") scripts/compute_vectorization_cache.sh)
done

# ---- C: active-fraction caches for P_ACTIVE 90 ------------------------------
declare -A STATS
STATS[minmax]=$(DATA_ROOT="${MINMAX}" P_ACTIVE=90 ABS_THRESHOLDS= SKIP_MOMENTS=true \
  submit $(dep "${ZZ[minmax_p90]}") scripts/compute_grid_stats_cache.sh)
STATS[trialmean]=$(DATA_ROOT="${TRIALMEAN}" P_ACTIVE=90 ABS_THRESHOLDS= SKIP_MOMENTS=true \
  submit $(dep "${ZZ[trialmean_p90]}") scripts/compute_grid_stats_cache.sh)

# ---- D: overview classifications (logreg, fixed C) ---------------------------
for TARGET in video_label video_id segment_id; do
  for SCOPE in within_mouse cross_mouse; do
    for key in minmax_p90 trialmean_p90 trialmean_pf50 trialmean_pf70 trialmean_pf90; do
      grid="${key%%_*}"; thr="${key#*_}"
      root=$([[ "${grid}" == minmax ]] && echo "${MINMAX}" || echo "${TRIALMEAN}")
      if [[ "${thr}" == pf* ]]; then p="${thr#pf}"; pf=true; else p="${thr#p}"; pf=false; fi
      DATA_ROOT="${root}" TARGET=$TARGET SCOPE=$SCOPE P_ACTIVE="${p}" PER_FRAME_THRESH="${pf}" ABS_THRESHOLD= \
        RUN_NAME="${RUN_NAME}" MODEL=logreg submit $(dep "${VEC[${key}]}") scripts/classify_zigzag.sh > /dev/null
    done
    for grid in minmax trialmean; do
      root=$([[ "${grid}" == minmax ]] && echo "${MINMAX}" || echo "${TRIALMEAN}")
      for TA in false true; do
        DATA_ROOT="${root}" FEATURES=active_fraction P_ACTIVE=90 ABS_THRESHOLD= TARGET=$TARGET SCOPE=$SCOPE \
          TIME_AVERAGE=$TA RUN_NAME="${RUN_NAME}" MODEL=logreg submit $(dep "${STATS[${grid}]}") scripts/classify_grid_stats.sh > /dev/null
      done
    done
  done
done
echo "steps A-D: ${N} jobs" >&2

# ---- E: incremental analysis, logreg_cv -------------------------------------
export MODEL=logreg_cv RUN_NAME
inc() { GRID="$1" THRESHOLDS="$2" DEPENDENCY="$3" bash scripts/run_incremental.sh; }
inc minmax "p30 p50 p70" ""
inc minmax "p90" "${VEC[minmax_p90]}:${STATS[minmax]}"
inc trialmean "abs0 p30 p50 p70" ""
inc trialmean "p90" "${VEC[trialmean_p90]}:${STATS[trialmean]}"
# one call so the shared moments-alone control is submitted once (waits for the three caches)
inc trialmean "pf50 pf70 pf90" "${VEC[trialmean_pf50]}:${VEC[trialmean_pf70]}:${VEC[trialmean_pf90]}"
