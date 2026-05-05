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
│   ├── classification_models.py
│   ├── shuffle_utils.py
│   ├── compute_grid_activation.py/.sh
│   ├── compute_zigzag.py/.sh
│   ├── compute_vectorization_cache.py/.sh
│   ├── explore_vectorizations.py/.sh
│   ├── classify_labels_within_mouse_ablation.py/.sh
│   ├── classify_labels_cross_mouse_ablation.py/.sh
│   ├── classify_labels_within_mouse_ablation_shuffle.py/.sh
│   ├── classify_labels_cross_mouse_ablation_shuffle.py/.sh
│   ├── classify_segments_id_within_mouse.py/.sh
│   ├── classify_segments_id_cross_mouse.py/.sh
│   ├── classify_video_id_within_mouse.py/.sh
│   ├── classify_video_id_cross_mouse.py/.sh
│   ├── similarity_id_within_mouse.py/.sh
│   ├── similarity_id_cross_mouse.py/.sh
│   └── generate_trials_ablation_shuffle.py/.sh
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
	Precomputes vectorization caches for downstream reuse.

- `scripts/explore_vectorizations.py`
	One-mouse vectorization exploration and diagnostics.

### 2) Label Classification

- `scripts/classify_labels_within_mouse_ablation.py`
	Within-mouse label classification with vector and grid models.

- `scripts/classify_labels_cross_mouse_ablation.py`
	Cross-mouse leave-one-mouse-out label classification.

### 3) Segment-ID Decoding

- `scripts/classify_segments_id_within_mouse.py`
	Within-mouse segment-ID decoding by label.

- `scripts/classify_segments_id_cross_mouse.py`
	Cross-mouse segment-ID decoding by label.

### 4) Video-ID Decoding

- `scripts/classify_video_id_within_mouse.py`
	Within-mouse video-ID decoding by label.

- `scripts/classify_video_id_cross_mouse.py`
	Cross-mouse video-ID decoding.

### 5) Similarity Analyses

- `scripts/similarity_id_within_mouse.py`
	Within-mouse trial/ID similarity analyses.

- `scripts/similarity_id_cross_mouse.py`
	Cross-mouse trial/ID similarity analyses.

### 6) Shuffle Ablation Workflow

Two-step workflow:

1. `scripts/generate_trials_ablation_shuffle.py`
	 Generate shuffled vectorization caches.
2. `scripts/classify_labels_within_mouse_ablation_shuffle.py` or
	 `scripts/classify_labels_cross_mouse_ablation_shuffle.py`
	 Run classifiers on shuffled caches.

## Running Jobs

For cluster runs, use the `.sh` wrappers in `scripts/` (they set Slurm resources and default arguments).

Examples:

```bash
sbatch scripts/classify_labels_within_mouse_ablation.sh
sbatch scripts/classify_segments_id_cross_mouse.sh
sbatch scripts/classify_video_id_within_mouse.sh
```

## Recommended Commands by Pipeline

Use these as the default entrypoints for each analysis family:

```bash
# Data preparation
sbatch scripts/compute_grid_activation.sh
sbatch scripts/compute_zigzag.sh

# Vectorizations cache precomputation
sbatch scripts/compute_vectorization_cache.sh

# Label classification
sbatch scripts/classify_labels_within_mouse_ablation.sh
sbatch scripts/classify_labels_cross_mouse_ablation.sh

# Segment-ID decoding
sbatch scripts/classify_segments_id_within_mouse.sh
sbatch scripts/classify_segments_id_cross_mouse.sh

# Video-ID decoding
sbatch scripts/classify_video_id_within_mouse.sh
sbatch scripts/classify_video_id_cross_mouse.sh

# Similarity analyses
sbatch scripts/similarity_id_within_mouse.sh
sbatch scripts/similarity_id_cross_mouse.sh

# Shuffle ablation (2-step)
sbatch scripts/generate_trials_ablation_shuffle.sh
sbatch scripts/classify_labels_within_mouse_ablation_shuffle.sh
sbatch scripts/classify_labels_cross_mouse_ablation_shuffle.sh
```

## Outputs

Each run writes into a timestamped folder under the corresponding `results/*` subtree and typically includes:

- metrics CSV
- metrics JSON
- confusion matrices JSON
- prediction outputs JSON
- figures/
- logs/run.log

## Notes

- Shared reusable code lives in:
	- `scripts/utils.py`
	- `scripts/classification_models.py`
	- `scripts/shuffle_utils.py`

## License

This project is licensed under the BSD 3-Clause License.
See `LICENSES/` for details.