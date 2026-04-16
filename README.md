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
│   ├── compute_grid_activation.py / .sh
│   ├── compute_zigzag.py / .sh
│   ├── explore_vectorizations.py / .sh
│   ├── classify_trials_within_mouse.py / .sh
│   ├── classify_trials_within_mouse_ablation.py / .sh
│   └── classify_trials_cross_mouse_ablation.py / .sh
├── results/
│   ├── vectorizations/                 # Exploration of the zigzag vectorizations for one mouse
│   ├── within_mouse_classification_ablation/  # Within mouse classification outputs
│   └── between_mouse_classification_ablation/  # Between mouse classification outputs
└── logs/                               # Slurm job logs
```

---

## Scripts

| Script | Description |
|---|---|
| `compute_grid_activation.py` | Bins raw Sensorium neural responses into 3D spatial grids (15×15×10 voxels) per trial. Uses a Fortran kernel (`assign_grid.f90`) for efficient grid assignment. |
| `compute_zigzag.py` | Computes cubical zigzag persistence on the 3D activation grids for all trials of a given mouse. Designed to be run as a Slurm array job (one task per mouse). |
| `explore_vectorizations.py` | Extracts zigzag vectorizations for one mouse and produces figures (feature matrices, PCA, inter-trial similarity, classification tests). |
| `classify_trials_within_mouse.py` | Within-mouse trial classification using configurable zigzag vectorizations. Trains and evaluates a classifier on trials from a single mouse. |
| `classify_trials_within_mouse_ablation.py` | Within-mouse trial classification using configurable zigzag vectorizations, and ablation study comparing zigzag vectorization models against a 3D-CNN baseline operating on raw activation grids. |
| `classify_trials_cross_mouse_ablation.py` | Leave-one-mouse-out cross-mouse classification. Trains on all eligible mice and evaluates on the held-out mouse; restricts labels to those shared between train and test sets. Compares zigzag vectorization models against a 3D-CNN baseline operating on raw activation grids. |
| `utils.py` | Shared utilities for loading zigzag persistence data and computing vectorizations, used by all classification and exploration scripts. |

Shell scripts (`.sh`) are the corresponding Slurm batch submission wrappers for each Python script.

---

## License

The software of this project is licensed under the BSD 3-Clause "New" or "Revised" License.

See LICENSES/ for details.