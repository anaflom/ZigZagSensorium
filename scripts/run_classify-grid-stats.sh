#!/bin/bash
# Submit all grid-statistics classifications (logreg, CPU):
#   moments_raw / moments_density, per frame and trial-averaged,
#   active_fraction for each P_ACTIVE, per frame and trial-averaged,
# for every target and scope.  Run compute_grid_stats_cache.sh first.
#
# Usage: bash scripts/run_classify-grid-stats.sh   (from the project root)

set -euo pipefail

RUN_NAME="$(date +%Y%m%d_%H%M%S)"

for TARGET in video_label video_id segment_id; do
  for SCOPE in within_mouse cross_mouse; do
    for TIME_AVERAGE in false true; do
      for FEATURES in moments_raw moments_density; do
        FEATURES=$FEATURES TARGET=$TARGET SCOPE=$SCOPE TIME_AVERAGE=$TIME_AVERAGE RUN_NAME=$RUN_NAME \
          sbatch scripts/classify_grid_stats.sh
      done
      for P_ACTIVE in 30 50 70; do
        FEATURES=active_fraction P_ACTIVE=$P_ACTIVE TARGET=$TARGET SCOPE=$SCOPE TIME_AVERAGE=$TIME_AVERAGE \
          RUN_NAME=$RUN_NAME sbatch scripts/classify_grid_stats.sh
      done
    done
  done
done
