# How to continue...

## To check

### Imbalance in the video-label distribution
- Check whether the classification metric is the best choice given the imbalanced distribution of trials across labels (for example, macro F1).
- Check whether another design should be applied, such as resampling.

### Classification schema for the cross mice
- In the current implementation only the labels common to training and testing sets are kept (4 labels). COnsider if keeping all 6 labels could make sense


## Next steps

### Other ablation experiments to investigate what the zigzag persistence is catching

#### 1. Destroying temporal structure: time shuffle

Destroy temporal structure and then apply the video-type classification again:
- within mouse on the zigzag vectorization using LogReg
- within mouse on the grid data using 3D-CNN
- cross mouse on the zigzag vectorization using LogReg
- cross mouse on the grid data using 3D-CNN

There could be two approaches for destroying temporal structure that should give similar results:
1. Shuffle time samples coherently in space (same shuffle for all cubes in the grid)
2. Apply an FFT and add a random phase coherently in space (same phase added for all cubes in the grid), then transform back to the temporal domain. This should preserve spectral properties of the activity, not only the mean and standard deviation.

Hypothesis:
- Time shuffling should not significantly affect either within-mouse or cross-mouse classification performance for the 3D-CNN on grid activity.
- Time shuffling should significantly affect both within-mouse and cross-mouse classification performance for LogReg on zigzag vectorizations.


Notes:
- How many shuffles to apply?

#### 2. Destroying spatial structure: grid shuffle

Destroy spatial structure by permuting the cubes in the grid coherently in time.
Then apply the classification schemes as with time shuffling.

Hypothesis:
- Space shuffling should not significantly affect either within-mouse or cross-mouse classification performance for the 3D-CNN on grid activity.
- Space shuffling should significantly affect both within-mouse and cross-mouse classification performance for LogReg on zigzag vectorizations (?)


### Similarity between representations for different instances of the same video

Differences between video classes are fairly obvious and might be due to detecting clear patterns in segment switches or features that actually represent each segment.
3D-CNN classification on grid activity probably relies on very different patterns of activity for each video label, rather than on temporal structure or other "high-level" structure in the activation patterns.
Investigating whether the zigzag vectorizations carry information about the videos might tell us something about what they are catching.
Classification might not be feasible due to the reduced number of trials; however, some things could still be done by looking at similarity matrices.

- Look at similarity matrices for the zigzag vectorizations for the different repeated videos within mouse. Check if differences exist for videos belonging to different classes. 
We have 
    - 18 NaturalVideos x ~10
    -

- Analogous analysis across mice. This is reduced to the NaturalVideo category since we do not have the responses available for the second set of mice for the parametric stimuli, as well as for part of the NaturalVideos. I need to check how many responses are available across mice.

Significance could be addressed by permuting ID labels.