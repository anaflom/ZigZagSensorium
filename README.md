# ZigZag Persistence for the Sensorium Dataset

This project applies **cubical zigzag persistence** to neural population responses from the [Sensorium dataset](https://sensorium-competition.net/). Topological features extracted from 3D activation grids are used to classify visual stimuli within and across mice.

Mouse metadata (session information, trial labels, stimulus identifiers) can be obtained from the companion repository at https://github.com/anaflom/sensorium/tree/main.

---

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
- **`zz-top`** — private package providing zigzag persistence and vectorization utilities (included as a local editable install under `zz-top/`; access must be granted separately)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Data

Neural response data is part of the **Sensorium dataset**, available at https://sensorium-competition.net/.

Mouse session metadata (trial labels, stimulus information, etc.) is available from the companion repository: https://github.com/anaflom/sensorium/tree/main.

---

## Project Structure

```
ZigZagSensorium/
├── requirements.txt
├── scripts/                            # Production CLI scripts (see below)
│   ├── assign_grid.f90                 # Fortran kernel for grid assignment
│   ├── utils.py                        # Shared data-loading & vectorization utilities
│   ├── classification_models.py        # Shared classification models and training utilities
│   ├── shuffle_utils.py                # Shared shuffle utilities for ablation studies
│   ├── compute_grid_activation.py / .sh
│   ├── compute_zigzag.py / .sh
│   ├── compute_vectorization_cache.py / .sh
│   ├── explore_vectorizations.py / .sh
│   ├── classify_labels_within_mouse_ablation.py / .sh
│   ├── classify_labels_cross_mouse_ablation.py / .sh
│   ├── classify_segments_id_within_mouse.py / .sh
│   ├── classify_video_id_cross_mouse.py / .sh
│   ├── generate_trials_ablation_shuffle.py / .sh
│   ├── classify_labels_within_mouse_ablation_shuffle.py / .sh
│   ├── classify_labels_cross_mouse_ablation_shuffle.py / .sh
│   ├── analyze_within_mouse_id_similarity.py / .sh
│   ├── analyze_cross_mouse_id_similarity.py / .sh
│   └── analyze_repeated_trial_clustering.py / .sh
├── results/
│   ├── vectorizations/                 # Exploration of zigzag vectorizations for one mouse
│   ├── within_mouse_classification_ablation/   # Within-mouse video-label classification outputs
│   ├── cross_mouse_classification_ablation/    # Cross-mouse video-label classification outputs
│   ├── within_mouse_segment_id_decoding/       # Within-mouse segment-ID decoding outputs
│   ├── cross_mouse_id_decoding/                # Cross-mouse video-ID decoding outputs
│   ├── within_mouse_id_similarity/             # Within-mouse trial similarity analysis outputs
│   ├── cross_mouse_id_similarity/              # Cross-mouse trial similarity analysis outputs
│   └── ablation_shuffle/                       # Shuffle ablation outputs (time / spatial)
└── logs/                               # Slurm job logs
```

---

## Scripts

### Data preparation

| Script | Description |
|---|---|
| `compute_grid_activation.py` | Bins raw Sensorium neural responses into 3D spatial grids (15×15×10 voxels) per trial. Uses a Fortran kernel (`assign_grid.f90`) for efficient grid assignment. |
| `compute_zigzag.py` | Computes cubical zigzag persistence on the 3D activation grids for all trials of a given mouse. Designed to be run as a Slurm array job (one task per mouse). |
| `compute_vectorization_cache.py` | Precomputes and caches one vectorization file per mouse for a given configuration (vectorization method, persistence threshold, frame clipping). Used to decouple vectorization from classification and speed up repeated runs. |
| `explore_vectorizations.py` | Extracts zigzag vectorizations for one mouse and produces figures (feature matrices, PCA, inter-trial similarity, classification tests). |

### Classification

| Script | Description |
|---|---|
| `classify_labels_within_mouse_ablation.py` | Within-mouse video-label classification. Compares zigzag vectorization models (LogReg, MLP, 1D-CNN) against a 3D-CNN baseline on raw activation grids. |
| `classify_labels_cross_mouse_ablation.py` | Leave-one-mouse-out cross-mouse video-label classification. Trains on all eligible mice and evaluates on the held-out mouse; restricts labels to those shared between train and test sets. Compares the same model families as the within-mouse script. |
| `classify_segments_id_within_mouse.py` | Within-mouse, within-label segment-ID decoding. Builds segment-level samples from video metadata, runs Leave-One-Segment-Out CV with LogReg (zigzag) and 3D-CNN (grid). |
| `classify_video_id_cross_mouse.py` | Cross-mouse video-ID decoding. Runs directed pair experiments (train on mouse A, test on mouse B) with LogReg and 3D-CNN; pairs are selected based on shared IDs repeated across mice. |

### Shuffle ablation

Shuffle ablation follows a two-step workflow: first generate cached shuffles, then classify on them.

| Script | Description |
|---|---|
| `generate_trials_ablation_shuffle.py` | Generates and caches shuffled zigzag vectorizations. Supports `--shuffle-type time` (per-voxel time-axis shuffle), `spatial` (coherent spatial permutation), and `phase` (FFT phase randomisation). Ensures a target number of shuffles exist without overwriting existing caches. |
| `classify_labels_within_mouse_ablation_shuffle.py` | Within-mouse video-label classification on pre-cached shuffled vectorizations. Requires shuffles from `generate_trials_ablation_shuffle.py`. |
| `classify_labels_cross_mouse_ablation_shuffle.py` | Cross-mouse (LOMO) video-label classification on pre-cached shuffled vectorizations. Requires shuffles from `generate_trials_ablation_shuffle.py`. |

Shuffle outputs are stored in separate sub-directories to avoid collisions:

- `results/ablation_shuffle/time/...`
- `results/ablation_shuffle/spatial/...`
- `results/ablation_shuffle/phase/...`

### Similarity analysis

| Script | Description |
|---|---|
| `similarity_id_within_mouse.py` | Computes within-mouse trial-level similarity matrices based on repeated stimulus IDs. Normalises and optionally PCA-reduces vectorizations per label, then aggregates pairwise distances into ID-to-ID similarity matrices and exports figures and summary CSVs. |
| `similarity_id_cross_mouse.py` | Cross-mouse similarity analysis for trials sharing the same stimulus ID. Compares within-mouse vs cross-mouse distance distributions for shared IDs; exports combined heatmaps, boxplots, and tidy distance CSVs. |

### Shared modules

| Module | Description |
|---|---|
| `utils.py` | Shared utilities for loading zigzag persistence data and computing vectorizations, used by all classification and exploration scripts. |
| `classification_models.py` | Shared classification model definitions (MLP, 1D-CNN, 3D-CNN) and training loop utilities. |
| `shuffle_utils.py` | Shared shuffle utilities (time, spatial, phase) used by the shuffle generation script. |

Shell scripts (`.sh`) are the corresponding Slurm batch submission wrappers for each Python script.

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/threshold_distribution.ipynb` | Loads per-trial activation thresholds computed with `p-active-per-trial=True`, joins them with trial metadata, and visualises the threshold distributions split by video-type label for each mouse. Includes overlapping histograms, boxplots with Holm-corrected pairwise significance brackets (OLS + multiple comparisons), and aggregated per-mouse statistical summaries. |

---

## License

The software of this project is licensed under the BSD 3-Clause "New" or "Revised" License.

See LICENSES/ for details.