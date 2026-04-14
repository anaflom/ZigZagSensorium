#!/usr/bin/env python3
"""Explore zigzag vectorizations from the command line.

This script ports the workflow from notebooks/02_explore_vectorizations.ipynb
into a headless CLI pipeline that:
1) saves all generated figures to an output folder, and
2) captures printed output in a log file.

Section 2 (vectorization extraction) always runs. Sections 3-8 are optional and
can be skipped with --skip-sections.
3) Visualize feature matrices
4) Inter-trial similarity (distance matrices)
5) PCA exploration
6) Correlation between vectorizations (representational similarity)
7) Preliminary classification test (cross-mouse)
7b) Within-mouse video-stimulus classification (requires metadata and labels)
8) Summary and recommendations
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

# Force a non-interactive backend for cluster/headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from zztop.vectorizations import (
    BettiCurve,
    BettiProfile,
    BirthFrequency,
    CumulativePersistence,
    EffectivePersistenceImage,
    PersistenceEntropy,
    PersistenceImage,
    PersistenceLandscape,
    PersistenceProfile,
    PersistenceStatistics,
    Silhouette,
    TurnoverRate,
)
from zztop.vectorizations._diagram import normalize_diagram


@dataclass
class RunState:
    data_root: Path
    meta_root: Path
    p_active: int
    per_trial_thresh: bool
    zz_folder: str
    ref_mouse: Optional[str]
    mouse_2: Optional[str]
    clip_frames_arg: Optional[int]
    skip_sections: Set[str]
    max_trials: Optional[int]


class FigureSaver:
    """Save figures with stable numbered filenames."""

    def __init__(self, figures_dir: Path) -> None:
        self.figures_dir = figures_dir
        self.counter = 1

    def save(self, fig: plt.Figure, name: str) -> Path:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
        out_path = self.figures_dir / f"{self.counter:02d}_{safe_name}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        self.counter += 1
        return out_path


def _str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _opt_int(value: str) -> Optional[int]:
    if value is None:
        return None
    if value.strip().lower() in {"none", "null", ""}:
        return None
    return int(value)


def _opt_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = value.strip()
    if s.lower() in {"none", "null", ""}:
        return None
    return s


def _parse_skip_sections(text: str) -> Set[str]:
    if not text.strip():
        return set()
    allowed = {"3", "4", "5", "6", "7", "7b", "8"}
    parts = {p.strip().lower() for p in text.split(",") if p.strip()}
    invalid = parts - allowed
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid --skip-sections entries: {sorted(invalid)}. Allowed: {sorted(allowed)}"
        )
    return parts


def _safe_name(name: str, n: int = 50) -> str:
    return f"{name[:n]}..." if len(name) > n else name


def _build_zz_folder(p_active: int, per_trial_thresh: bool) -> str:
    if per_trial_thresh:
        return f"trials_zz-thresh-{p_active}-per-trial"
    return f"trials_zz-thresh-{p_active}"


def _discover_mice(data_root: Path) -> List[str]:
    return sorted(
        [d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("dynamic")]
    )


def load_zigzag_barcodes(
    data_root: Path,
    mouse_name: str,
    zz_folder: str,
    max_trials: Optional[int] = None,
) -> Tuple[List[List[Tuple[float, float, float]]], List[str]]:
    """Load zigzag barcode files for one mouse."""
    zz_dir = data_root / mouse_name / zz_folder
    files = sorted(zz_dir.glob("zz-thresh-*.npy"))
    files = [f for f in files if "info" not in f.name]

    if max_trials is not None and files:
        indices = np.linspace(0, len(files) - 1, min(max_trials, len(files)), dtype=int)
        files = [files[i] for i in indices]

    barcodes: List[List[Tuple[float, float, float]]] = []
    trial_names: List[str] = []
    for fpath in files:
        raw = np.load(fpath, allow_pickle=True)
        bars = [tuple(row) for row in raw]
        barcodes.append(bars)
        trial_names.append(fpath.stem)

    return barcodes, trial_names


def find_ref_mouse(
    mice: Sequence[str], data_root: Path, zz_folder: str, provided: Optional[str]
) -> Optional[str]:
    if provided is not None:
        return provided
    for m in mice:
        zz_dir = data_root / m / zz_folder
        if zz_dir.is_dir() and len(list(zz_dir.glob("*.npy"))) > 10:
            return m
    return None


def find_second_mouse(
    mice: Sequence[str],
    data_root: Path,
    zz_folder: str,
    ref_mouse: str,
    provided: Optional[str],
) -> Optional[str]:
    if provided is not None:
        return provided
    for m in mice:
        if m == ref_mouse:
            continue
        zz_dir = data_root / m / zz_folder
        if zz_dir.is_dir() and len(list(zz_dir.glob("*.npy"))) > 10:
            return m
    return None


def load_trial_metadata(meta_root: Path, mouse_name: str) -> pd.DataFrame:
    csv_path = meta_root / mouse_name / "trials" / f"meta-trials_{mouse_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_labelled_barcodes(
    data_root: Path,
    meta_root: Path,
    mouse_name: str,
    zz_folder: str,
) -> Tuple[List[List[Tuple[float, float, float]]], np.ndarray, List[int], np.ndarray]:
    """Load barcodes and attach stimulus labels from metadata."""
    df = load_trial_metadata(meta_root, mouse_name)
    trial_to_label = dict(zip(df["trial"].astype(int), df["label"]))
    trial_to_frames = dict(zip(df["trial"].astype(int), df["valid_frames"].astype(int)))

    zz_dir = data_root / mouse_name / zz_folder
    files = sorted(zz_dir.glob("zz-thresh-*.npy"))
    files = [f for f in files if "info" not in f.name]

    barcodes: List[List[Tuple[float, float, float]]] = []
    labels_list: List[str] = []
    trial_ids: List[int] = []
    frames_list: List[int] = []

    for fpath in files:
        match = re.search(r"trial-(\d+)", fpath.stem)
        if match is None:
            continue
        trial_num = int(match.group(1))
        if trial_num not in trial_to_label:
            continue

        raw = np.load(fpath, allow_pickle=True)
        bars = [tuple(row) for row in raw]
        barcodes.append(bars)
        labels_list.append(trial_to_label[trial_num])
        trial_ids.append(trial_num)
        frames_list.append(trial_to_frames[trial_num])

    return barcodes, np.array(labels_list), trial_ids, np.array(frames_list)


def clip_barcodes(
    barcodes: Iterable[Iterable[Tuple[float, float, float]]], n_frames: int
) -> List[List[Tuple[float, float, float]]]:
    clipped: List[List[Tuple[float, float, float]]] = []
    for bars in barcodes:
        new_bars: List[Tuple[float, float, float]] = []
        for dim, b, d in bars:
            if b >= n_frames:
                continue
            new_bars.append((dim, b, min(d, n_frames)))
        clipped.append(new_bars)
    return clipped


def make_vectorizers() -> Dict[str, object]:
    return {
        "PI_20x20_s1": PersistenceImage(resolution=(20, 20), sigma=1.0),
        "PI_20x20_s2": PersistenceImage(resolution=(20, 20), sigma=2.0),
        "PI_30x30_s1": PersistenceImage(resolution=(30, 30), sigma=1.0),
        "Landscape_5": PersistenceLandscape(n_landscapes=5, resolution=100),
        "Landscape_10": PersistenceLandscape(n_landscapes=10, resolution=100),
        "Silhouette_p1": Silhouette(resolution=100, power=1.0),
        "Silhouette_p2": Silhouette(resolution=100, power=2.0),
        "BettiCurve_100": BettiCurve(resolution=100),
        "BettiCurve_200": BettiCurve(resolution=200),
        "Entropy": PersistenceEntropy(normalize=True),
        "Statistics": PersistenceStatistics(),
        "BettiProfile": BettiProfile(),
        "BirthFreq": BirthFrequency(),
        "PersProfile": PersistenceProfile(),
        "Turnover": TurnoverRate(),
        "EffPI": EffectivePersistenceImage(),
        "CumPers": CumulativePersistence(),
    }


def make_stim_vectorizers(clip_frames: int) -> Dict[str, object]:
    return {
        "PersImage": PersistenceImage(resolution=(20, 20)),
        "Landscape": PersistenceLandscape(resolution=100, n_landscapes=5),
        "Silhouette": Silhouette(resolution=100),
        "BettiCurve": BettiCurve(resolution=100),
        "PersEntropy": PersistenceEntropy(),
        "PersStats": PersistenceStatistics(),
        "BettiProfile": BettiProfile(n_frames=clip_frames),
        "BirthFreq": BirthFrequency(n_frames=clip_frames),
        "PersProfile": PersistenceProfile(n_frames=clip_frames),
        "Turnover": TurnoverRate(n_frames=clip_frames),
        "EffPI": EffectivePersistenceImage(n_frames=clip_frames),
        "CumPers": CumulativePersistence(n_frames=clip_frames),
    }


def compute_distance_matrices(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    distance_matrices: Dict[str, np.ndarray] = {}
    for name, xmat in features.items():
        x_clean = np.nan_to_num(xmat)
        scaler = StandardScaler()
        x_scaled = np.nan_to_num(scaler.fit_transform(x_clean))

        norms = np.linalg.norm(x_scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x_normed = x_scaled / norms

        try:
            dist = squareform(pdist(x_normed, metric="cosine"))
            distance_matrices[name] = np.nan_to_num(dist)
        except Exception as exc:
            print(f"  {name}: distance computation failed - {exc}")
    return distance_matrices


def run_pipeline(state: RunState, output_dir: Path) -> Dict[str, object]:
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as log_fp, redirect_stdout(log_fp), redirect_stderr(log_fp):
        print("=" * 90)
        print("Explore Zigzag Vectorizations (CLI)")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)

        print("\nResolved arguments:")
        print(f"  output_dir:      {output_dir}")
        print(f"  data_root:       {state.data_root}")
        print(f"  meta_root:       {state.meta_root}")
        print(f"  p_active:        {state.p_active}")
        print(f"  per_trial_thresh:{state.per_trial_thresh}")
        print(f"  zz_folder:       {state.zz_folder}")
        print(f"  ref_mouse:       {state.ref_mouse}")
        print(f"  mouse_2:         {state.mouse_2}")
        print(f"  clip_frames:     {state.clip_frames_arg}")
        print(f"  skip_sections:   {sorted(state.skip_sections)}")
        print(f"  max_trials:      {state.max_trials}")

        saver = FigureSaver(figures_dir)

        artifacts: Dict[str, object] = {
            "log_path": str(log_path),
            "figures": [],
            "status": {},
        }

        # Section 1: Load barcodes and pick reference mouse.
        print("\n## 1. Load computed zigzag barcodes")
        mice = _discover_mice(state.data_root)
        print(f"Found {len(mice)} mice")

        ref_mouse = find_ref_mouse(mice, state.data_root, state.zz_folder, state.ref_mouse)
        if ref_mouse is None:
            raise RuntimeError(
                f"No mouse has zigzag results in {state.zz_folder}. Check data_root/parameters."
            )
        print(f"Using mouse: {ref_mouse}")

        barcodes, trial_names = load_zigzag_barcodes(
            state.data_root, ref_mouse, state.zz_folder, max_trials=state.max_trials
        )
        print(f"Loaded {len(barcodes)} barcodes")
        for i in range(min(3, len(barcodes))):
            bars = barcodes[i]
            btype = type(bars[0]) if bars else "empty"
            print(f"  {trial_names[i]}: {len(bars)} bars, type={btype}")

        # Section 2 always runs.
        print("\n## 2. Standard + zigzag-specific vectorizations")
        vectorizers = make_vectorizers()
        features: Dict[str, np.ndarray] = {}
        for name, vec in vectorizers.items():
            try:
                xmat = vec.fit_transform(barcodes)
                if xmat.ndim == 1:
                    xmat = xmat.reshape(-1, 1)
                features[name] = xmat
                n_finite = np.isfinite(xmat).all(axis=1).sum()
                print(
                    f"  {name:>20s}: shape={str(xmat.shape):>15s}, "
                    f"finite={n_finite}/{len(xmat)}, "
                    f"range=[{np.nanmin(xmat):.4f}, {np.nanmax(xmat):.4f}]"
                )
            except Exception as exc:
                print(f"  {name:>20s}: FAILED - {exc}")

        artifacts["status"]["2"] = "done"

        # Section 3
        if "3" not in state.skip_sections:
            print("\n## 3. Visualize feature matrices")
            try:
                n_vecs = len(features)
                n_cols = 4
                n_rows = (n_vecs + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
                axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

                for ax, (name, xmat) in zip(axes_flat, features.items()):
                    x_disp = np.nan_to_num(xmat.copy())
                    if x_disp.shape[1] > 200:
                        step = max(1, x_disp.shape[1] // 200)
                        x_disp = x_disp[:, ::step]
                    ax.imshow(x_disp, aspect="auto", cmap="viridis", interpolation="nearest")
                    ax.set_title(name, fontsize=9)
                    ax.set_xlabel("Feature")
                    ax.set_ylabel("Trial")

                for ax in list(axes_flat)[len(features):]:
                    ax.set_visible(False)

                fig.suptitle("Feature matrices from all vectorizations", fontsize=14)
                fig.tight_layout()
                path = saver.save(fig, "03_feature_matrices")
                print(f"Saved figure: {path}")
                artifacts["figures"].append(str(path))

                selected_vecs = [
                    "BettiCurve_100",
                    "Landscape_5",
                    "Silhouette_p1",
                    "BettiProfile",
                    "PersProfile",
                    "Turnover",
                ]
                n_show = min(5, len(barcodes))
                fig, axes = plt.subplots(2, 3, figsize=(16, 8))

                for ax, vname in zip(axes.flat, selected_vecs):
                    if vname not in features:
                        ax.set_visible(False)
                        continue
                    xmat = features[vname]
                    for i in range(n_show):
                        ax.plot(xmat[i], alpha=0.6, linewidth=0.8, label=f"trial {i}")
                    ax.set_title(vname, fontsize=10)
                    ax.set_xlabel("Feature index")
                    ax.grid(alpha=0.3)
                    if vname == selected_vecs[0]:
                        ax.legend(fontsize=7, loc="upper right")

                fig.suptitle("Sample feature vectors across trials", fontsize=14)
                fig.tight_layout()
                path = saver.save(fig, "03_sample_feature_vectors")
                print(f"Saved figure: {path}")
                artifacts["figures"].append(str(path))
                artifacts["status"]["3"] = "done"
            except Exception:
                print("Section 3 failed:")
                traceback.print_exc()
                artifacts["status"]["3"] = "failed"
        else:
            print("\n## 3. Skipped by CLI")
            artifacts["status"]["3"] = "skipped"

        # Section 4
        distance_matrices: Dict[str, np.ndarray] = {}
        if "4" not in state.skip_sections:
            print("\n## 4. Inter-trial similarity")
            try:
                distance_matrices = compute_distance_matrices(features)

                n_dm = len(distance_matrices)
                n_cols = 4
                n_rows = (n_dm + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
                axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

                for ax, (name, dist) in zip(axes_flat, distance_matrices.items()):
                    im = ax.imshow(dist, cmap="plasma", vmin=0, interpolation="nearest")
                    ax.set_title(name, fontsize=8)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                for ax in list(axes_flat)[len(distance_matrices):]:
                    ax.set_visible(False)

                fig.suptitle("Pairwise cosine distance matrices", fontsize=14)
                fig.tight_layout()
                path = saver.save(fig, "04_pairwise_cosine_distances")
                print(f"Saved figure: {path}")
                artifacts["figures"].append(str(path))
                artifacts["status"]["4"] = "done"
            except Exception:
                print("Section 4 failed:")
                traceback.print_exc()
                artifacts["status"]["4"] = "failed"
        else:
            print("\n## 4. Skipped by CLI")
            artifacts["status"]["4"] = "skipped"

        # Section 5
        if "5" not in state.skip_sections:
            print("\n## 5. PCA exploration")
            try:
                n_vecs = len(features)
                n_cols = 4
                n_rows = (n_vecs + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
                axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

                for ax, (name, xmat) in zip(axes_flat, features.items()):
                    x_clean = np.nan_to_num(xmat)
                    if x_clean.shape[1] < 2:
                        ax.text(
                            0.5,
                            0.5,
                            f"{name}\\n(1D, skip PCA)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=9,
                        )
                        continue

                    scaler = StandardScaler()
                    x_scaled = np.nan_to_num(scaler.fit_transform(x_clean))
                    pca = PCA(n_components=2)
                    x_pca = pca.fit_transform(x_scaled)

                    ax.scatter(x_pca[:, 0], x_pca[:, 1], s=8, alpha=0.5)
                    ax.set_title(
                        f"{name}\\nVar: {pca.explained_variance_ratio_.sum():.1%}", fontsize=9
                    )
                    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

                for ax in list(axes_flat)[n_vecs:]:
                    ax.set_visible(False)

                fig.suptitle("PCA projections of vectorized barcodes", fontsize=14)
                fig.tight_layout()
                path = saver.save(fig, "05_pca_projections")
                print(f"Saved figure: {path}")
                artifacts["figures"].append(str(path))
                artifacts["status"]["5"] = "done"
            except Exception:
                print("Section 5 failed:")
                traceback.print_exc()
                artifacts["status"]["5"] = "failed"
        else:
            print("\n## 5. Skipped by CLI")
            artifacts["status"]["5"] = "skipped"

        # Section 6
        if "6" not in state.skip_sections:
            print("\n## 6. Correlation between vectorizations")
            try:
                if not distance_matrices:
                    print("Distance matrices not available yet; computing now for section 6.")
                    distance_matrices = compute_distance_matrices(features)

                vec_names = list(distance_matrices.keys())
                n = len(vec_names)
                corr_matrix = np.zeros((n, n))

                for i in range(n):
                    for j in range(n):
                        d_i = squareform(distance_matrices[vec_names[i]])
                        d_j = squareform(distance_matrices[vec_names[j]])
                        corr_matrix[i, j] = np.corrcoef(d_i, d_j)[0, 1]

                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                ax.set_xticklabels(vec_names, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(vec_names, fontsize=8)
                plt.colorbar(im, ax=ax, label="Correlation of distance matrices")
                ax.set_title("Representational similarity between vectorizations", fontsize=13)
                fig.tight_layout()
                path = saver.save(fig, "06_representational_similarity")
                print(f"Saved figure: {path}")
                artifacts["figures"].append(str(path))
                artifacts["status"]["6"] = "done"
            except Exception:
                print("Section 6 failed:")
                traceback.print_exc()
                artifacts["status"]["6"] = "failed"
        else:
            print("\n## 6. Skipped by CLI")
            artifacts["status"]["6"] = "skipped"

        # Section 7
        results: Dict[str, Dict[str, float]] = {}
        mouse_2 = state.mouse_2
        if "7" not in state.skip_sections:
            print("\n## 7. Preliminary classification test (cross-mouse)")
            try:
                mouse_2 = find_second_mouse(
                    mice, state.data_root, state.zz_folder, ref_mouse, state.mouse_2
                )

                if mouse_2 is None:
                    print("Only one mouse has zigzag results - skipping classification test.")
                else:
                    print(f"Mouse 1: {_safe_name(ref_mouse, 40)}")
                    print(f"Mouse 2: {_safe_name(mouse_2, 40)}")

                    barcodes_2, _ = load_zigzag_barcodes(
                        state.data_root, mouse_2, state.zz_folder, max_trials=state.max_trials
                    )

                    all_barcodes = barcodes + barcodes_2
                    labels = np.array([0] * len(barcodes) + [1] * len(barcodes_2))
                    print(
                        f"Combined: {len(all_barcodes)} barcodes, "
                        f"class 0: {(labels == 0).sum()}, class 1: {(labels == 1).sum()}"
                    )

                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    for vname, vec in vectorizers.items():
                        try:
                            xmat = vec.fit_transform(all_barcodes)
                            if xmat.ndim == 1:
                                xmat = xmat.reshape(-1, 1)
                            xmat = np.nan_to_num(xmat)

                            pipe = Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    (
                                        "clf",
                                        LogisticRegression(max_iter=1000, random_state=42),
                                    ),
                                ]
                            )
                            scores = cross_val_score(
                                pipe, xmat, labels, cv=cv, scoring="accuracy"
                            )
                            results[vname] = {
                                "mean_acc": float(scores.mean()),
                                "std_acc": float(scores.std()),
                                "n_features": int(xmat.shape[1]),
                            }
                            print(
                                f"  {vname:>20s}: acc={scores.mean():.3f} +/- {scores.std():.3f} "
                                f"(n_feat={xmat.shape[1]})"
                            )
                        except Exception as exc:
                            print(f"  {vname:>20s}: FAILED - {exc}")

                    if results:
                        names = list(results.keys())
                        accs = [results[n]["mean_acc"] for n in names]
                        stds = [results[n]["std_acc"] for n in names]

                        zz_specific = {
                            "BettiProfile",
                            "BirthFreq",
                            "PersProfile",
                            "Turnover",
                            "EffPI",
                            "CumPers",
                        }
                        colors = ["tab:orange" if n in zz_specific else "tab:blue" for n in names]

                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.bar(range(len(names)), accs, yerr=stds, color=colors, capsize=3, alpha=0.8)
                        ax.set_xticks(range(len(names)))
                        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
                        ax.set_ylabel("5-fold CV Accuracy")
                        ax.set_title(
                            "Mouse classification accuracy by vectorization\n"
                            "(blue=standard, orange=zigzag-specific)",
                            fontsize=12,
                        )
                        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
                        ax.legend()
                        ax.set_ylim(0, 1.05)
                        ax.grid(axis="y", alpha=0.3)
                        fig.tight_layout()
                        path = saver.save(fig, "07_cross_mouse_classification")
                        print(f"Saved figure: {path}")
                        artifacts["figures"].append(str(path))

                artifacts["status"]["7"] = "done"
            except Exception:
                print("Section 7 failed:")
                traceback.print_exc()
                artifacts["status"]["7"] = "failed"
        else:
            print("\n## 7. Skipped by CLI")
            artifacts["status"]["7"] = "skipped"

        # Section 7b
        stim_results: Dict[str, Dict[str, float]] = {}
        clip_frames_used: Optional[int] = None
        if "7b" not in state.skip_sections:
            print("\n## 7b. Within-mouse video-stimulus classification")
            try:
                stim_barcodes, stim_labels, stim_trial_ids, stim_valid_frames = load_labelled_barcodes(
                    state.data_root, state.meta_root, ref_mouse, state.zz_folder
                )
                print(f"Mouse: {_safe_name(ref_mouse)}")
                print(f"Total matched trials: {len(stim_barcodes)}")
                unique, counts = np.unique(stim_labels, return_counts=True)
                for u, c in zip(unique, counts):
                    print(f"  {u:>15s}: {c:>4d} trials")
                print(
                    f"Valid-frames range: {stim_valid_frames.min()} - {stim_valid_frames.max()}"
                )

                if state.clip_frames_arg is None:
                    clip_frames_used = int(stim_valid_frames.min())
                    print(f"clip_frames is None; using min(valid_frames)={clip_frames_used}")
                else:
                    clip_frames_used = int(state.clip_frames_arg)
                    print(f"Using provided clip_frames={clip_frames_used}")

                stim_barcodes_clipped = clip_barcodes(stim_barcodes, clip_frames_used)
                for label in sorted(set(stim_labels)):
                    mask = stim_labels == label
                    orig = sum(len(b) for b, m in zip(stim_barcodes, mask) if m)
                    clp = sum(len(b) for b, m in zip(stim_barcodes_clipped, mask) if m)
                    pct = 100.0 * clp / orig if orig > 0 else float("nan")
                    print(
                        f"  {label:>15s}: {orig:>6d} bars -> {clp:>6d} after clipping "
                        f"({pct:.1f}%)"
                    )

                print("Pre-normalising clipped barcodes to DiagramDict ...")
                stim_diagrams = [normalize_diagram(b, drop_inf=True) for b in stim_barcodes_clipped]
                print(f"Done - {len(stim_diagrams)} diagrams ready.")

                stim_vectorizers = make_stim_vectorizers(clip_frames_used)
                cv_stim = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scoring = {"acc": "accuracy", "f1": "f1_macro"}

                for vname, vec in stim_vectorizers.items():
                    try:
                        xmat = vec.fit_transform(stim_diagrams)
                        if xmat.ndim == 1:
                            xmat = xmat.reshape(-1, 1)
                        xmat = np.nan_to_num(xmat)

                        pipe = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                (
                                    "clf",
                                    LogisticRegression(
                                        max_iter=2000,
                                        random_state=42,
                                        class_weight="balanced",
                                    ),
                                ),
                            ]
                        )
                        cv_res = cross_validate(
                            pipe, xmat, stim_labels, cv=cv_stim, scoring=scoring
                        )
                        acc_scores = cv_res["test_acc"]
                        f1_scores = cv_res["test_f1"]
                        stim_results[vname] = {
                            "mean_acc": float(acc_scores.mean()),
                            "std_acc": float(acc_scores.std()),
                            "mean_f1": float(f1_scores.mean()),
                            "std_f1": float(f1_scores.std()),
                            "n_features": int(xmat.shape[1]),
                        }
                        print(
                            f"  {vname:>15s}: acc={acc_scores.mean():.3f}+/-{acc_scores.std():.3f}  "
                            f"F1={f1_scores.mean():.3f}+/-{f1_scores.std():.3f}  "
                            f"(n_feat={xmat.shape[1]})"
                        )
                    except Exception as exc:
                        print(f"  {vname:>15s}: FAILED - {exc}")

                if stim_results:
                    names = list(stim_results.keys())
                    accs = [stim_results[n]["mean_acc"] for n in names]
                    a_err = [stim_results[n]["std_acc"] for n in names]
                    f1s = [stim_results[n]["mean_f1"] for n in names]
                    f_err = [stim_results[n]["std_f1"] for n in names]

                    zz_specific = {
                        "BettiProfile",
                        "BirthFreq",
                        "PersProfile",
                        "Turnover",
                        "EffPI",
                        "CumPers",
                    }
                    colors = ["tab:orange" if n in zz_specific else "tab:blue" for n in names]

                    x = np.arange(len(names))
                    width = 0.35
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.bar(
                        x - width / 2,
                        accs,
                        width,
                        yerr=a_err,
                        color=colors,
                        capsize=3,
                        alpha=0.8,
                        label="Accuracy",
                    )
                    ax.bar(
                        x + width / 2,
                        f1s,
                        width,
                        yerr=f_err,
                        color=colors,
                        capsize=3,
                        alpha=0.4,
                        hatch="//",
                        label="Macro F1",
                    )

                    n_classes = len(np.unique(stim_labels))
                    ax.axhline(
                        1.0 / n_classes,
                        color="gray",
                        ls="--",
                        alpha=0.5,
                        label=f"Chance (1/{n_classes})",
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
                    ax.set_ylabel("Score")
                    ax.set_title(
                        f"Within-mouse stimulus classification ({n_classes} classes, "
                        f"{len(stim_barcodes)} trials)\\n"
                        "blue = standard, orange = zigzag-specific  |  "
                        "solid = accuracy, hatched = macro-F1",
                        fontsize=11,
                    )
                    ax.legend(loc="upper right")
                    ax.set_ylim(0, 1.05)
                    ax.grid(axis="y", alpha=0.3)
                    fig.tight_layout()
                    path = saver.save(fig, "07b_stimulus_classification")
                    print(f"Saved figure: {path}")
                    artifacts["figures"].append(str(path))

                    best_name = max(stim_results, key=lambda k: stim_results[k]["mean_f1"])
                    best_vec = stim_vectorizers[best_name]

                    x_best = best_vec.fit_transform(stim_diagrams)
                    if x_best.ndim == 1:
                        x_best = x_best.reshape(-1, 1)
                    x_best = np.nan_to_num(x_best)

                    pipe_best = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "clf",
                                LogisticRegression(
                                    max_iter=2000,
                                    random_state=42,
                                    class_weight="balanced",
                                ),
                            ),
                        ]
                    )
                    y_pred = cross_val_predict(pipe_best, x_best, stim_labels, cv=cv_stim)
                    class_order = sorted(np.unique(stim_labels))
                    cm = confusion_matrix(stim_labels, y_pred, labels=class_order)

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    ConfusionMatrixDisplay(cm, display_labels=class_order).plot(
                        ax=axes[0], cmap="Blues", colorbar=False
                    )
                    axes[0].set_title(f"Confusion matrix (counts)\\n{best_name}")

                    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                    cm_norm = np.nan_to_num(cm_norm)
                    ConfusionMatrixDisplay(cm_norm, display_labels=class_order).plot(
                        ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f"
                    )
                    axes[1].set_title(f"Confusion matrix (recall)\\n{best_name}")
                    fig.tight_layout()
                    path = saver.save(fig, "07b_best_vectorizer_confusion_matrix")
                    print(f"Saved figure: {path}")
                    artifacts["figures"].append(str(path))

                    print(
                        f"Best vectorizer by macro-F1: {best_name} "
                        f"(F1={stim_results[best_name]['mean_f1']:.3f})"
                    )

                artifacts["status"]["7b"] = "done"
            except Exception:
                print("Section 7b failed:")
                traceback.print_exc()
                artifacts["status"]["7b"] = "failed"
        else:
            print("\n## 7b. Skipped by CLI")
            artifacts["status"]["7b"] = "skipped"

        # Section 8
        if "8" not in state.skip_sections:
            print("\n## 8. Summary and recommendations")
            try:
                if not distance_matrices:
                    print("Distance matrices not available yet; computing now for section 8.")
                    distance_matrices = compute_distance_matrices(features)

                print(
                    f"{'Vectorization':>20s} | {'Dims':>6s} | {'Dist spread':>12s} | "
                    f"{'Mouse acc':>10s} | {'Stim F1':>10s} | {'Type':>10s}"
                )
                print("-" * 90)

                zz_specific = {
                    "BettiProfile",
                    "BirthFreq",
                    "PersProfile",
                    "Turnover",
                    "EffPI",
                    "CumPers",
                }
                for name in features:
                    n_feat = int(features[name].shape[1])

                    if name in distance_matrices:
                        d_spread = f"{np.std(squareform(distance_matrices[name])):.4f}"
                    else:
                        d_spread = "N/A"

                    if results and name in results:
                        acc = f"{results[name]['mean_acc']:.3f}"
                    else:
                        acc = "N/A"

                    if stim_results and name in stim_results:
                        stim_f1 = f"{stim_results[name]['mean_f1']:.3f}"
                    else:
                        stim_f1 = "N/A"

                    vtype = "zigzag" if name in zz_specific else "standard"
                    print(
                        f"{name:>20s} | {n_feat:>6d} | {d_spread:>12s} | "
                        f"{acc:>10s} | {stim_f1:>10s} | {vtype:>10s}"
                    )

                print("\n" + "=" * 90)
                print("RECOMMENDATION: Choose vectorization(s) with:")
                print("  - High stimulus-classification F1 (within-mouse discriminative power)")
                print("  - High cross-mouse accuracy (captures mouse-specific signatures)")
                print("  - Moderate feature dimensionality")
                print("  - Low redundancy with other chosen vectorizations")
                print("  - Consider combining one standard + one zigzag-specific vectorization")
                artifacts["status"]["8"] = "done"
            except Exception:
                print("Section 8 failed:")
                traceback.print_exc()
                artifacts["status"]["8"] = "failed"
        else:
            print("\n## 8. Skipped by CLI")
            artifacts["status"]["8"] = "skipped"

        artifacts["resolved"] = {
            "ref_mouse": ref_mouse,
            "mouse_2": mouse_2,
            "clip_frames": clip_frames_used,
        }

        metrics_path = output_dir / "metrics_summary.json"
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "status": artifacts["status"],
                    "resolved": artifacts["resolved"],
                    "n_figures": len(artifacts["figures"]),
                    "figures": artifacts["figures"],
                    "cross_mouse_results": results,
                    "stim_results": stim_results,
                },
                fp,
                indent=2,
            )
        print(f"\nWrote metrics summary: {metrics_path}")
        artifacts["metrics_path"] = str(metrics_path)

    return artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert notebook vectorization exploration to a CLI run that saves figures and logs."
        )
    )
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)
    parser.add_argument("--p-active", required=True, type=int)
    parser.add_argument("--per-trial-thresh", required=True, type=_str2bool)
    parser.add_argument("--ref-mouse", default=None, type=_opt_str)
    parser.add_argument("--mouse-2", default=None, type=_opt_str)
    parser.add_argument("--clip-frames", default=None, type=_opt_int)
    parser.add_argument(
        "--skip-sections",
        default="",
        type=_parse_skip_sections,
        help="Comma-separated list among: 3,4,5,6,7,7b,8",
    )
    parser.add_argument(
        "--max-trials",
        default=None,
        type=_opt_int,
        help="Optional cap for number of barcode files sampled per mouse.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = args.output_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    state = RunState(
        data_root=args.data_root,
        meta_root=args.meta_root,
        p_active=args.p_active,
        per_trial_thresh=args.per_trial_thresh,
        zz_folder=_build_zz_folder(args.p_active, args.per_trial_thresh),
        ref_mouse=args.ref_mouse,
        mouse_2=args.mouse_2,
        clip_frames_arg=args.clip_frames,
        skip_sections=args.skip_sections,
        max_trials=args.max_trials,
    )

    try:
        artifacts = run_pipeline(state, output_dir)
    except Exception as exc:
        # If the crash happened before log redirection, show details on stderr.
        print(f"Fatal error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # Keep terminal output concise; full details are in run.log.
    print("Run completed.")
    print(f"Log file: {artifacts['log_path']}")
    print(f"Figures saved: {len(artifacts['figures'])}")
    print(f"Metrics summary: {artifacts['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
