<!--
SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>

SPDX-License-Identifier: BSD-3-Clause
-->

# ZigZag Persistence for the Sensorium Dataset

This repository applies cubical zigzag persistence to neural responses from the Sensorium dataset. 
It builds per-trial 3D activation grids, computes zigzag persistence barcodes, computes vectorizations, and runs multiple decoding analyses within and across mice.
Decoding is done on zigzag persistence vectorizations using Logistic Regression (MLP or 1D-CNN are also supported) and for comparison on the grid activity using a 3D-CNN. 

Mouse trial metadata (labels, stimulus identifiers, sessions) is expected from the companion Sensorium metadata repository:
https://github.com/anaflom/sensorium/tree/main

## Requirements

- Python ≥ 3.9
- `numpy >= 1.23`
- `tqdm >= 4.60`
- `pandas >= 1.5`
- `scikit-learn >= 1.2`
- `torch >= 2.0`
- `matplotlib >= 3.7`
- `scipy >= 1.9`
- `jupyter >= 1.0`
- **`zz-top`** — private package providing zigzag persistence and vectorization utilities; access must be granted separately
- **`assign_grid.f90`** — private code necessary for estimating grid activation; access must be granted separately

Install:

```bash
pip install -r requirements.txt
```

## Data Inputs

Neural response data is part of the **Sensorium dataset**, available at https://sensorium-competition.net/.

Mouse session metadata (trial labels, stimulus information, etc.) is available from the companion repository: https://github.com/anaflom/sensorium/tree/main.

In the scripts: 

- Sensorium-derived neural/grid data under `--data-root`
- Companion metadata under `--meta-root`


## Current Repository Layout

```text
ZigZagSensorium/
├── scripts/
│   ├── assign_grid.f90
│   ├── utils.py
│   ├── classification_models.py      # networks, training loops
│   ├── classification_data.py        # what is classified: samples, classes, frame windows, folds
│   ├── classification_features.py    # zigzag features from caches, grid datasets, shuffle caches
│   ├── classification_engine.py      # shared CLI, seeding, evaluation, output files
│   ├── results_index.py              # load results for plotting
│   ├── shuffle_utils.py
│   ├── compute_grid_activation.py/.sh
│   ├── compute_zigzag.py/.sh
│   ├── compute_vectorization_cache.py/.sh
│   ├── explore_vectorizations.py/.sh
│   ├── classify_zigzag.py/.sh
│   ├── classify_grid.py/.sh
│   ├── generate_trials_ablation_shuffle.py/.sh
│   ├── ablation_shuffle_zigzag.py/.sh
│   ├── ablation_shuffle_grid.py/.sh
│   ├── similarity_id_within_mouse.py/.sh
│   └── similarity_id_cross_mouse.py/.sh
├── notebooks/
└── logs/
```

## Main Pipelines

### 1) Data Preparation and Vectorization

- `scripts/compute_grid_activation.py`
	Builds trial-level 3D activation grids (uses `assign_grid.f90`).

- `scripts/compute_zigzag.py`
	Computes zigzag persistence outputs from the activation grids.

- `scripts/compute_vectorization_cache.py`
	Precomputes vectorization caches for downstream reuse. For Turnover, the
	unclipped cache (`CLIP_FRAMES=None`) is enough: shorter windows are cut from it
	(see `FRAME_ALIGNED_BLOCKS` in `utils.py`). Other vectorizations need a cache
	for each clip length they are used with.

- `scripts/explore_vectorizations.py`
	One-mouse vectorization exploration and diagnostics.

### 2) Classification

Two entry points share the same problem definitions (`classification_data.py`),
so zigzag and grid results are computed on identical samples, frame windows and folds:

- `scripts/classify_zigzag.py`: zigzag vectorizations (default model: `logreg`; also `mlp`, `cnn1d`).
- `scripts/classify_grid.py`: grid activity with a 3D-CNN (`cnn3d_raw`, `cnn3d_norm`).
	Does not depend on `P_ACTIVE`. `--n-repeats` (or a Slurm array) retrains with
	different initialisation seeds on fixed folds to measure run-to-run variability.

Both take `--target` and `--scope`:

| `--target` | classes | frame window |
|---|---|---|
| `video_label` | stimulus label | first N frames, N = shortest trial over the mice used (240) |
| `video_id` | video ID within each label | first N frames, N = shortest trial in each problem (e.g. 300 for NaturalVideo) |
| `segment_id` | segment ID within each label | the segment's own frames |

| `--scope` | evaluation |
|---|---|
| `within_mouse` | CV within each mouse (`stratified` 5-fold for video_label, leave-one-trial-out for video_id / segment_id) |
| `cross_mouse` | leave-one-mouse-out (video_label, segment_id); directed mouse pairs (video_id) |

### 3) Shuffle Ablation

1. `scripts/generate_trials_ablation_shuffle.py`: generate shuffled grids and their vectorization caches.
2. `scripts/ablation_shuffle_zigzag.py` / `scripts/ablation_shuffle_grid.py`: run the same
	classification problems on the shuffled data, once per shuffle.

### 4) Similarity Analyses

- `scripts/similarity_id_within_mouse.py`
	Within-mouse trial/ID similarity analyses.

- `scripts/similarity_id_cross_mouse.py`
	Cross-mouse trial/ID similarity analyses.

## Running Jobs

For cluster runs, use the `.sh` wrappers in `scripts/` (they set Slurm resources
and default arguments; settings are passed as environment variables):

```bash
# Data preparation
sbatch scripts/compute_grid_activation.sh
sbatch scripts/compute_zigzag.sh
sbatch scripts/compute_vectorization_cache.sh

# Classification on zigzag vectorizations (CPU)
TARGET=video_label SCOPE=within_mouse P_ACTIVE=30 sbatch scripts/classify_zigzag.sh
TARGET=video_id    SCOPE=cross_mouse  P_ACTIVE=70 sbatch scripts/classify_zigzag.sh

# Classification on grids (GPU), 5 repeats as a Slurm array
TARGET=segment_id SCOPE=cross_mouse sbatch --array=0-4 scripts/classify_grid.sh

# Shuffle ablation (2-step)
sbatch scripts/generate_trials_ablation_shuffle.sh
SCOPE=within_mouse SHUFFLE_TYPE=time sbatch scripts/ablation_shuffle_zigzag.sh
SCOPE=within_mouse SHUFFLE_TYPE=time GEN_P_ACTIVE=30 sbatch scripts/ablation_shuffle_grid.sh

# Similarity analyses
sbatch scripts/similarity_id_within_mouse.sh
sbatch scripts/similarity_id_cross_mouse.sh
```

## Outputs

```text
results/
├── zigzag/<vectorization>/<model>/p<P>-<per-trial|global>/<target>_<scope>/<run>/repeat-NN/
├── grid/<model>/<target>_<scope>/<run>/repeat-NN/
└── ablation_shuffle/
    ├── zigzag/<vectorization>/<model>/<shuffle>-<mode>/p<P>-<...>/<target>_<scope>/<run>/shuffle-NNNN/
    └── grid/<model>/<shuffle>-<mode>/<target>_<scope>/<run>/shuffle-NNNN/
```

Each `repeat-NN` / `shuffle-NNNN` folder contains:

- `metrics.csv`: one row per classification problem (pooled accuracy, balanced accuracy, macro-F1, chance, fold mean/std)
- `fold_metrics.csv`: one row per fold
- `predictions.csv`: one row per test sample (true and predicted class)
- `skips.csv`: problems that could not be built, with the reason
- `run_config.json`: all arguments, seeds, caches used and git commit
- `figures/summary.png`, `run.log`

The first columns of every CSV describe the run (target, scope, input, vectorization,
model, p_active, thresh_mode, repeat/shuffle_id, seeds), so results can be combined
without parsing paths:

```python
from results_index import load_metrics, summarize_repeats
df = load_metrics(target="video_label", scope="cross_mouse")         # zigzag + grid
summarize_repeats(load_metrics(input="grid", model="cnn3d_raw"))     # mean/std over repeats
```

Results produced by the previous scripts (before this layout) are in `results_legacy/`.

## Notes

- Shared reusable code lives in:
	- `scripts/utils.py`
	- `scripts/classification_*.py`
	- `scripts/shuffle_utils.py`

## License

This project is licensed under the BSD 3-Clause License.
See `LICENSES/` for details.