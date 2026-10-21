#!/bin/bash
# Incremental-information analysis: what the zigzag topology adds beyond the
# activity level.  For each zigzag threshold, target and scope it submits
# classify_zigzag.sh (logreg) with
#   Turnover-H1, Turnover-H2                         (one homology dimension)
#   Turnover+active-fraction, Turnover-H1+..., -H2+... (topology + level control)
# Compare with the active-fraction control alone (classify_grid_stats.py) and
# plain Turnover, which other scripts already run with the same folds.
#
# Usage (from the project root):
#   GRID=minmax    bash scripts/run_incremental.sh
#   GRID=trialmean DEPENDENCY=<job ids, colon-separated> bash scripts/run_incremental.sh

set -euo pipefail

GRID="${GRID:?Set GRID=minmax|trialmean}"
DERIV="/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives"
case "${GRID}" in
  minmax)    export DATA_ROOT="${DERIV}/grid-15x15x10_norm-by_minmax";    THRESHOLDS="${THRESHOLDS:-p30 p50 p70}" ;;
  trialmean) export DATA_ROOT="${DERIV}/grid-15x15x10_norm-by_trialmean"; THRESHOLDS="${THRESHOLDS:-abs0 p30 p50 p70}" ;;
  *) echo "Unknown GRID=${GRID}"; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
DEP_ARGS=()
[[ -n "${DEPENDENCY:-}" ]] && DEP_ARGS=(--dependency="afterok:${DEPENDENCY}")

# blocks|added features  (plain Turnover is already run by the main analyses)
VARIANTS=("1|" "2|" "|active_fraction" "1|active_fraction" "2|active_fraction")

n=0
for THR in ${THRESHOLDS}; do
  if [[ "${THR}" == abs* ]]; then thr_env=(ABS_THRESHOLD="${THR#abs}" P_ACTIVE=); else thr_env=(P_ACTIVE="${THR#p}" ABS_THRESHOLD=); fi
  for TARGET in video_label video_id segment_id; do
    for SCOPE in within_mouse cross_mouse; do
      for V in "${VARIANTS[@]}"; do
        BLOCKS="${V%%|*}"; ADD="${V#*|}"
        extra=""
        [[ -n "${BLOCKS}" ]] && extra+="--blocks ${BLOCKS} "
        [[ -n "${ADD}" ]] && extra+="--add-features ${ADD}"
        env "${thr_env[@]}" TARGET="${TARGET}" SCOPE="${SCOPE}" RUN_NAME="${RUN_NAME}" EXTRA_ARGS="${extra}" \
          sbatch --parsable "${DEP_ARGS[@]}" scripts/classify_zigzag.sh > /dev/null
        n=$((n + 1))
      done
    done
  done
done
echo "submitted ${n} jobs for GRID=${GRID} (thresholds: ${THRESHOLDS}; run name ${RUN_NAME})"
