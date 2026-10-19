for P_ACTIVE in 30 50 70; do
  for TARGET in video_label video_id segment_id; do
    for SCOPE in within_mouse cross_mouse; do
      TARGET=$TARGET SCOPE=$SCOPE P_ACTIVE=$P_ACTIVE sbatch scripts/classify_zigzag.sh
    done
  done
done