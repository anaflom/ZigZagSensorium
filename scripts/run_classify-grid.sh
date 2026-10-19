#!/bin/bash
# Submit all grid (3D-CNN) classifications, N_REPEATS repeats each as a Slurm array.
#
# Each array task runs one repeat and must fit in the 18 h GPU limit.  All
# schemes fit in one job per repeat except within-mouse segment_id (~4-5 h per
# mouse), which is split into one job per mouse; those jobs share RUN_NAME and
# write repeat-NN_<mouse> folders into the same run folder.
#
# Usage: bash scripts/run_classify-grid.sh   (from the project root)

set -euo pipefail

N_REPEATS="${N_REPEATS:-5}"
ARRAY="0-$((N_REPEATS - 1))"
RUN_NAME="$(date +%Y%m%d_%H%M%S)"

# Mice with segment metadata (the other mice only have NaturalVideo trials).
SEGMENT_MICE=(
  dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce
  dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce
)

for TARGET in video_label video_id segment_id; do
  for SCOPE in within_mouse cross_mouse; do
    if [[ "${TARGET}" == "segment_id" && "${SCOPE}" == "within_mouse" ]]; then
      for MOUSE in "${SEGMENT_MICE[@]}"; do
        TARGET=$TARGET SCOPE=$SCOPE MICE=$MOUSE PART="${MOUSE%%-Video*}" RUN_NAME=$RUN_NAME \
          sbatch --array="${ARRAY}" scripts/classify_grid.sh
      done
    else
      TARGET=$TARGET SCOPE=$SCOPE RUN_NAME=$RUN_NAME sbatch --array="${ARRAY}" scripts/classify_grid.sh
    fi
  done
done
