#!/bin/bash
# Incremental-information analysis: what the zigzag topology adds beyond the
# activity level.  For each zigzag threshold, target and scope it submits, with
# the same model (default logreg_cv: regularization tuned by inner CV):
#   control alone                          (classify_grid_stats.sh)
#   Turnover, Turnover-H1, Turnover-H2     (classify_zigzag.sh)
#   each of the three + control            (classify_zigzag.sh --add-features)
# Control: active fraction at the same threshold (per-trial / fixed thresholds);
# moments for per-frame thresholds, where the active fraction is constant.
#
# Thresholds: p<P> (per-trial P_ACTIVE), abs<T> (fixed), pf<P> (per-frame P_ACTIVE).
#
# Usage (from the project root):
#   GRID=minmax    THRESHOLDS="p30 p50 p70 p90" bash scripts/run_incremental.sh
#   GRID=trialmean THRESHOLDS="pf50 pf70" DEPENDENCY=<job ids, colon-separated> bash scripts/run_incremental.sh
#   DRY_RUN=true ... prints the jobs instead of submitting them

set -euo pipefail

GRID="${GRID:?Set GRID=minmax|trialmean}"
DERIV="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives"
case "${GRID}" in
  minmax)    export DATA_ROOT="${DERIV}/grid-15x15x10_norm-by_minmax";    MOMENTS=moments_raw ;;
  trialmean) export DATA_ROOT="${DERIV}/grid-15x15x10_norm-by_trialmean"; MOMENTS=moments_std ;;
  *) echo "Unknown GRID=${GRID}"; exit 1 ;;
esac
THRESHOLDS="${THRESHOLDS:?Set THRESHOLDS, e.g. \"p30 p50 abs0 pf50\"}"
export MODEL="${MODEL:-logreg_cv}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
DEP_ARGS=()
[[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS=(--dependency="afterok:${DEPENDENCY}")

n=0
submit() {  # submit <script> with the current environment
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  [dry-run] ${GRID} THR=${THR} FEATURES=${FEATURES:-} EXTRA=[${EXTRA_ARGS:-}] T=${TARGET}/${SCOPE} MODEL=${MODEL} $*"
  else
    sbatch --parsable "${DEP_ARGS[@]}" "$@" > /dev/null
  fi
  n=$((n + 1))
}

moments_done=""
for THR in ${THRESHOLDS}; do
  case "${THR}" in
    abs*) export ABS_THRESHOLD="${THR#abs}" P_ACTIVE= PER_FRAME_THRESH=false; CONTROL=active_fraction ;;
    pf*)  export P_ACTIVE="${THR#pf}" ABS_THRESHOLD= PER_FRAME_THRESH=true;  CONTROL="${MOMENTS}" ;;
    p*)   export P_ACTIVE="${THR#p}" ABS_THRESHOLD= PER_FRAME_THRESH=false; CONTROL=active_fraction ;;
    *) echo "Unknown threshold ${THR}"; exit 1 ;;
  esac
  for TARGET in video_label video_id segment_id; do
    for SCOPE in within_mouse cross_mouse; do
      export TARGET SCOPE RUN_NAME
      # control alone (moments do not depend on the threshold: once per scheme)
      if [[ "${CONTROL}" == "active_fraction" ]]; then
        FEATURES=active_fraction EXTRA_ARGS= submit scripts/classify_grid_stats.sh
      elif [[ " ${moments_done} " != *" ${TARGET}/${SCOPE} "* ]]; then
        (export P_ACTIVE= ABS_THRESHOLD=; FEATURES="${MOMENTS}" EXTRA_ARGS= submit scripts/classify_grid_stats.sh)
        n=$((n + 1)); moments_done+=" ${TARGET}/${SCOPE}"
      fi
      # topology alone and topology + control
      for BLOCKS in "" 1 2; do
        for ADD in "" "${CONTROL}"; do
          extra=""
          [[ -n "${BLOCKS}" ]] && extra+="--blocks ${BLOCKS} "
          [[ -n "${ADD}" ]] && extra+="--add-features ${ADD}"
          FEATURES= EXTRA_ARGS="${extra}" submit scripts/classify_zigzag.sh
        done
      done
    done
  done
done
echo "${DRY_RUN/true/[dry-run] }submitted ${n} jobs for GRID=${GRID} (thresholds: ${THRESHOLDS}; model ${MODEL}; run name ${RUN_NAME})"
