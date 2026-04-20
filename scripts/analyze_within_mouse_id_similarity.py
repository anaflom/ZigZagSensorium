#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Analyze within-mouse trial similarity based on repeated stimulus IDs.

For each mouse and label independently:
1. Filter trials to only those with repeated IDs (ID appearing >= min_id_repetitions)
2. Normalize vectors and optionally reduce dimensionality with PCA (per label)
3. Compute trial-level distance matrices (pairwise Euclidean distances)
4. Aggregate distances by stimulus ID to create ID-to-ID similarity matrix
5. Generate visualizations (per-trial heatmap, ID-to-ID heatmap, distance distributions)
6. Export results (distances, plots, summary)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import (
    _build_zz_folder,
    _opt_csv_list,
    _opt_int,
    _resolve_mouse_cache_dir,
    _str2bool,
    load_labelled_barcodes,
    load_or_compute_vectorization_features,
    load_trial_metadata,
    _discover_mice,
    _eligible_trials,
)


@dataclass
class RunState:
    output_folder: Path
    data_root: Path
    meta_root: Path
    p_active: int
    per_trial_thresh: bool
    zz_folder: str
    method: str
    mice: Optional[List[str]]
    clip_frames: Optional[int]
    cache_dir: Optional[Path]
    force_recompute: bool
    max_trials: Optional[int]
    min_id_repetitions: int
    n_pca_components: Optional[int]
    seed: int


def _opt_none_or_int(value: str) -> Optional[int]:
    """Convert string to None or int.
    
    Accepts:
    - "None", "none", "null", "NULL" -> None
    - Any integer string -> int(value)
    """
    if isinstance(value, str) and value.lower() in ("none", "null"):
        return None
    if value is None:
        return None
    return int(value)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Output and input
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Data root folder")
    parser.add_argument("--meta-root", type=Path, required=True, help="Metadata root folder")

    # Zarr parameters
    parser.add_argument("--p-active", type=int, default=30, help="Persistence threshold (default 30)")
    parser.add_argument(
        "--per-trial-thresh",
        type=_str2bool,
        default=False,
        help="Use per-trial thresholds (default False)",
    )

    # Vectorization parameters
    parser.add_argument(
        "--vectorization-method",
        type=str,
        default="BettiCurve",
        help="Vectorization method (default BettiCurve)",
    )
    parser.add_argument("--clip-frames", type=_opt_int, default=None, help="Clip frames")
    parser.add_argument("--cache-dir", type=lambda x: Path(x) if x else None, default=None, help="Cache folder")
    parser.add_argument("--force-recompute", type=_str2bool, default=False, help="Force recompute cache")
    parser.add_argument("--max-trials", type=_opt_int, default=None, help="Max trials per mouse")

    # Mouse selection
    parser.add_argument(
        "--mice",
        type=_opt_csv_list,
        default=None,
        help="Comma-separated mouse names (default: all discovered)",
    )

    # Analysis parameters
    parser.add_argument(
        "--min-id-repetitions",
        type=int,
        default=7,
        help="Minimum number of times an ID must appear to be included (default 7)",
    )
    parser.add_argument(
        "--n-pca-components",
        type=_opt_none_or_int,
        default=10,
        help="Number of PCA components per label, or 'None' to skip PCA (default 10)",
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")

    return parser.parse_args()


def filter_to_repeated_trials(
    df_meta: pd.DataFrame,
    min_id_repetitions: int,
) -> pd.DataFrame:
    """Filter metadata to trials with repeated IDs and eligible status.
    
    Args:
        df_meta: Metadata DataFrame with columns including 'ID', 'trial', 'valid_trial', 'valid_response'
        min_id_repetitions: Minimum number of repetitions for an ID to be kept
        
    Returns:
        Filtered DataFrame containing only eligible trials with repeated IDs
    """
    # Check if required columns exist
    if "ID" not in df_meta.columns:
        raise ValueError(f"Metadata must contain 'ID' column. Found columns: {df_meta.columns.tolist()}")
    
    # Apply eligibility filter first (same as in load_labelled_barcodes)
    df_eligible = _eligible_trials(df_meta)
    
    # Group by ID and count occurrences (on eligible trials only)
    id_counts = df_eligible["ID"].value_counts()
    repeated_ids = id_counts[id_counts >= min_id_repetitions].index.tolist()
    
    # Filter to repeated IDs
    df_filtered = df_eligible[df_eligible["ID"].isin(repeated_ids)].copy()
    
    return df_filtered


def prepare_label_data(
    xmat: np.ndarray,
    labels: np.ndarray,
    trial_ids: List[int],
    df_meta_filtered: pd.DataFrame,
    label: str,
    n_pca_components: Optional[int],
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Prepare data for a single label: filter, normalize, optional PCA.
    
    Args:
        xmat: Feature matrix (n_samples x n_features)
        labels: Label array (n_samples,)
        trial_ids: List of trial IDs corresponding to xmat rows
        df_meta_filtered: Filtered metadata with ID column
        label: Label value to filter to
        n_pca_components: Number of PCA components, or None to skip PCA
        seed: Random seed for PCA
        
    Returns:
        Dict with 'X_pca' (or 'X_normalized' if no PCA), 'id_labels', 'trial_ids_label' or None if insufficient data
    """
    # Filter to this label
    mask_label = labels == label
    trial_ids_label_all = [trial_ids[i] for i in range(len(trial_ids)) if mask_label[i]]
    
    if len(trial_ids_label_all) == 0:
        return None
    
    # Build trial_to_id mapping from filtered metadata
    df_meta_label_trials = df_meta_filtered[df_meta_filtered["trial"].isin(trial_ids_label_all)].copy()
    trial_to_id = dict(zip(df_meta_label_trials["trial"].astype(int), df_meta_label_trials["ID"]))
    
    # Keep only trials that are in the filtered metadata (have repeated IDs)
    indices_to_keep = []
    trial_ids_label = []
    id_labels_list = []
    
    for i, tid in enumerate(trial_ids_label_all):
        if tid in trial_to_id:
            indices_to_keep.append(i)
            trial_ids_label.append(tid)
            id_labels_list.append(trial_to_id[tid])
    
    if len(trial_ids_label) == 0:
        return None
    
    X_label = xmat[mask_label][indices_to_keep]
    id_labels = np.array(id_labels_list)
    
    # Normalize: StandardScaler
    X_label_clean = np.nan_to_num(X_label)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_label_clean)
    X_scaled = np.nan_to_num(X_scaled)
    
    # L2 normalization
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X_scaled / norms
    
    # PCA (per label) - optional
    if n_pca_components is not None:
        n_components = min(n_pca_components, X_normed.shape[0], X_normed.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        X_reduced = pca.fit_transform(X_normed)
    else:
        X_reduced = X_normed
        pca = None
    
    return {
        "X_reduced": X_reduced,
        "X_normalized": X_normed,
        "id_labels": id_labels,
        "trial_ids_label": trial_ids_label,
        "pca_model": pca,
        "was_pca_applied": n_pca_components is not None,
    }


def compute_distance_matrix(
    X_reduced: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Euclidean distances.
    
    Args:
        X_reduced: Feature matrix (n_samples x n_features), either PCA-reduced or normalized
        
    Returns:
        Distance matrix (n_samples x n_samples)
    """
    dist_condensed = pdist(X_reduced, metric="euclidean")
    dist_matrix = squareform(dist_condensed)
    return dist_matrix


def compute_id_aggregated_distance_matrix(
    dist_matrix: np.ndarray,
    id_labels: np.ndarray,
    trial_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate trial-level distances to ID-level distance matrix.
    
    For each pair of unique IDs, compute the mean distance between all trials
    with those IDs.
    
    Args:
        dist_matrix: Trial-level distance matrix (n_trials x n_trials)
        id_labels: ID label for each trial (n_trials,)
        trial_ids: Trial IDs (n_trials,)
        
    Returns:
        Tuple of (ID_distance_matrix, unique_ids) where:
        - ID_distance_matrix: (n_unique_ids x n_unique_ids) symmetric distance matrix
        - unique_ids: sorted array of unique ID values
    """
    unique_ids = np.sort(np.unique(id_labels))
    n_ids = len(unique_ids)
    
    # Map ID values to indices
    id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
    
    # Initialize ID-level distance matrix
    id_dist_matrix = np.zeros((n_ids, n_ids))
    
    # For each pair of IDs, compute mean distance
    for i, id_i in enumerate(unique_ids):
        for j, id_j in enumerate(unique_ids):
            # Get indices of trials with these IDs
            indices_i = np.where(id_labels == id_i)[0]
            indices_j = np.where(id_labels == id_j)[0]
            
            # Extract distances between these trial pairs
            distances = dist_matrix[np.ix_(indices_i, indices_j)].flatten()
            
            # Store mean distance
            id_dist_matrix[i, j] = np.mean(distances)
    
    return id_dist_matrix, unique_ids


def plot_trial_heatmap_with_clustering(
    dist_matrix: np.ndarray,
    id_labels: np.ndarray,
    trial_ids: List[int],
    output_path: Path,
) -> None:
    """Plot trial-level distance matrix heatmap with trials ordered by ID.
    
    Args:
        dist_matrix: Distance matrix (n_samples x n_samples)
        id_labels: ID labels for each trial
        trial_ids: Trial IDs
        output_path: Path to save figure
    """
    # Sort by ID
    sort_indices = np.argsort(id_labels)
    dist_sorted = dist_matrix[sort_indices][:, sort_indices]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dist_sorted, cmap="viridis", aspect="auto")
    ax.set_title("Trial-Level Distance Matrix (ordered by ID)")
    ax.set_xlabel("Trial (sorted by ID)")
    ax.set_ylabel("Trial (sorted by ID)")
    plt.colorbar(im, ax=ax, label="Euclidean Distance")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_id_distance_heatmap(
    id_dist_matrix: np.ndarray,
    unique_ids: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ID-aggregated distance matrix heatmap.
    
    Args:
        id_dist_matrix: ID-level distance matrix (n_unique_ids x n_unique_ids)
        unique_ids: Unique ID values (n_unique_ids,)
        output_path: Path to save figure
    """
    n_ids = len(unique_ids)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(id_dist_matrix, cmap="viridis", aspect="auto")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_ids))
    ax.set_yticks(np.arange(n_ids))
    ax.set_xticklabels(unique_ids, rotation=45, ha="right")
    ax.set_yticklabels(unique_ids)
    
    ax.set_title("ID-Aggregated Distance Matrix (mean distances between ID pairs)")
    ax.set_xlabel("Stimulus ID")
    ax.set_ylabel("Stimulus ID")
    plt.colorbar(im, ax=ax, label="Mean Euclidean Distance")
    
    # Add grid
    ax.set_xticks(np.arange(n_ids) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_ids) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_boxplot_distances_by_id(
    dist_matrix: np.ndarray,
    id_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Plot boxplot of distances stratified by ID pairs.
    
    Args:
        dist_matrix: Distance matrix
        id_labels: ID labels for each trial
        output_path: Path to save figure
    """
    # Extract distances within same ID vs between different IDs
    distances_within = []
    distances_between = []
    
    for i in range(len(id_labels)):
        for j in range(i + 1, len(id_labels)):
            d = dist_matrix[i, j]
            if id_labels[i] == id_labels[j]:
                distances_within.append(d)
            else:
                distances_between.append(d)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [distances_within, distances_between]
    bp = ax.boxplot(data_to_plot, tick_labels=["Same ID", "Different ID"], patch_artist=True)
    
    for patch, color in zip(bp["boxes"], ["lightgreen", "lightcoral"]):
        patch.set_facecolor(color)
    
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Distribution of Distances by ID Relationship")
    ax.grid(axis="y", alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def export_trial_distance_matrix_csv(
    dist_matrix: np.ndarray,
    trial_ids: List[int],
    id_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Export trial-level distance matrix to tidy CSV format.
    
    Args:
        dist_matrix: Distance matrix
        trial_ids: Trial IDs
        id_labels: ID labels
        output_path: Path to save CSV
    """
    rows = []
    for i in range(len(trial_ids)):
        for j in range(i + 1, len(trial_ids)):
            rows.append({
                "trial_id_1": trial_ids[i],
                "trial_id_2": trial_ids[j],
                "id_label_1": id_labels[i],
                "id_label_2": id_labels[j],
                "distance": dist_matrix[i, j],
            })
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def export_id_distance_matrix_csv(
    id_dist_matrix: np.ndarray,
    unique_ids: np.ndarray,
    output_path: Path,
) -> None:
    """Export ID-aggregated distance matrix to CSV format.
    
    Args:
        id_dist_matrix: ID-level distance matrix
        unique_ids: Unique ID values
        output_path: Path to save CSV
    """
    df = pd.DataFrame(id_dist_matrix, index=unique_ids, columns=unique_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


def run_pipeline(state: RunState) -> Dict[str, Any]:
    """Main analysis pipeline: per-mouse, per-label similarity analysis."""
    
    output_folder = state.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Analysis: Within-Mouse Trial Similarity by Stimulus ID")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output folder: {output_folder}")
    print(f"Data root: {state.data_root}")
    print(f"Meta root: {state.meta_root}")
    print(f"P-active: {state.p_active}")
    print(f"Vectorization method: {state.method}")
    print(f"Min ID repetitions: {state.min_id_repetitions}")
    print(f"N PCA components: {state.n_pca_components if state.n_pca_components is not None else 'None (PCA disabled)'}")
    print(f"Seed: {state.seed}")
    
    # Discover mice
    discovered_mice = _discover_mice(state.data_root)
    selected_mice = state.mice if state.mice is not None else discovered_mice
    selected_mice = [m for m in selected_mice if m in discovered_mice]
    
    if not selected_mice:
        raise RuntimeError("No valid mice selected for analysis.")
    
    print(f"\nDiscovered mice: {len(discovered_mice)}")
    print(f"Selected mice: {len(selected_mice)}")
    print(f"Selected: {selected_mice}")
    
    # Results storage
    summary_rows = []
    
    # Per-mouse loop
    for mouse_name in selected_mice:
        print(f"\n## Mouse: {mouse_name}")
        mouse_out_folder = output_folder / mouse_name
        
        try:
            # Load metadata
            df_meta = load_trial_metadata(state.meta_root, mouse_name)
            print(f"  Metadata: {len(df_meta)} rows")
            
            # Check for ID column
            if "ID" not in df_meta.columns:
                print(f"  ERROR: Metadata missing 'ID' column. Skipping.")
                continue
            
            # Filter to repeated trials
            df_meta_filtered = filter_to_repeated_trials(df_meta, state.min_id_repetitions)
            print(f"  Repeated trials (ID >= {state.min_id_repetitions}): {len(df_meta_filtered)} rows")
            
            if len(df_meta_filtered) == 0:
                print(f"  No repeated trials found. Skipping mouse.")
                continue
            
            # Load barcodes
            barcodes, labels, trial_ids, valid_frames = load_labelled_barcodes(
                state.data_root,
                state.meta_root,
                mouse_name,
                state.zz_folder,
                max_trials=state.max_trials,
            )
            
            if len(barcodes) == 0:
                print(f"  No labelled barcodes found. Skipping.")
                continue
            
            print(f"  Loaded {len(barcodes)} labelled barcodes")
            print(f"  Unique labels: {len(np.unique(labels))}")
            
            # Clip frames
            clip_used = state.clip_frames
            if clip_used is None:
                clip_used = int(valid_frames.min())
                print(f"  Clip frames: {clip_used} (from min valid_frames)")
            
            # Load or compute vectorization
            xmat, vec_source, _cache_path = load_or_compute_vectorization_features(
                data_root=state.data_root,
                mouse_name=mouse_name,
                method=state.method,
                p_active=state.p_active,
                per_trial_thresh=state.per_trial_thresh,
                clip_frames=int(clip_used),
                barcodes=barcodes,
                labels=labels,
                trial_ids=trial_ids,
                valid_frames=valid_frames,
                cache_dir=_resolve_mouse_cache_dir(state, mouse_name),
                force_recompute=state.force_recompute,
            )
            if vec_source == "cache":
                print(f"  Using cached vectorization")
            else:
                print(f"  Computed vectorization")
            
            print(f"  Feature matrix shape: {xmat.shape}")
            
            # Per-label analysis
            unique_labels = np.unique(labels)
            print(f"  Analyzing {len(unique_labels)} labels...")
            
            for label in unique_labels:
                print(f"    Label: {label}")
                label_out_folder = mouse_out_folder / label
                
                try:
                    # Prepare label-specific data
                    label_data = prepare_label_data(
                        xmat, labels, trial_ids, df_meta_filtered, label,
                        state.n_pca_components, state.seed
                    )
                    
                    if label_data is None:
                        print(f"      Insufficient data for label. Skipping.")
                        continue
                    
                    X_reduced = label_data["X_reduced"]
                    id_labels = label_data["id_labels"]
                    trial_ids_label = label_data["trial_ids_label"]
                    was_pca_applied = label_data["was_pca_applied"]
                    
                    n_unique_ids = len(np.unique(id_labels))
                    print(f"      Trials: {len(trial_ids_label)}, Unique IDs: {n_unique_ids}")
                    print(f"      Feature dimensionality: {X_reduced.shape[1]} ({('PCA-reduced' if was_pca_applied else 'normalized')})")
                    
                    if n_unique_ids < 2:
                        print(f"      Only one unique ID. Skipping.")
                        continue
                    
                    # Compute trial-level distance matrix
                    dist_matrix = compute_distance_matrix(X_reduced)
                    
                    # Compute ID-aggregated distance matrix
                    id_dist_matrix, unique_ids = compute_id_aggregated_distance_matrix(
                        dist_matrix, id_labels, trial_ids_label
                    )
                    
                    # Create output folder
                    label_out_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Export trial distances
                    trial_dist_csv_path = label_out_folder / "trial_distances.csv"
                    export_trial_distance_matrix_csv(dist_matrix, trial_ids_label, id_labels, trial_dist_csv_path)
                    print(f"      Saved trial distances: {trial_dist_csv_path}")
                    
                    # Export ID distances
                    id_dist_csv_path = label_out_folder / "id_distances.csv"
                    export_id_distance_matrix_csv(id_dist_matrix, unique_ids, id_dist_csv_path)
                    print(f"      Saved ID distances: {id_dist_csv_path}")
                    
                    # Generate plots
                    trial_heatmap_path = label_out_folder / "trial_heatmap.png"
                    plot_trial_heatmap_with_clustering(dist_matrix, id_labels, trial_ids_label, trial_heatmap_path)
                    print(f"      Saved trial heatmap: {trial_heatmap_path}")
                    
                    id_heatmap_path = label_out_folder / "id_distance_heatmap.png"
                    plot_id_distance_heatmap(id_dist_matrix, unique_ids, id_heatmap_path)
                    print(f"      Saved ID distance heatmap: {id_heatmap_path}")
                    
                    boxplot_path = label_out_folder / "boxplot_distances.png"
                    plot_boxplot_distances_by_id(dist_matrix, id_labels, boxplot_path)
                    print(f"      Saved boxplot: {boxplot_path}")
                    
                    # Create summary row
                    summary_rows.append({
                        "mouse": mouse_name,
                        "label": label,
                        "n_trials": len(trial_ids_label),
                        "n_unique_ids": n_unique_ids,
                        "dimensionality": X_reduced.shape[1],
                        "pca_applied": was_pca_applied,
                    })
                    
                except Exception as exc:
                    print(f"      FAILED: {exc}")
                    traceback.print_exc()
        
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
    
    # Export summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_folder / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
        print(summary_df.to_string())
    else:
        print("\nNo results to summarize.")
    
    print(f"\nCompleted: {datetime.now().isoformat()}")
    
    return {
        "status": "completed",
        "n_mice": len(selected_mice),
        "n_labels_analyzed": len(summary_rows),
        "summary_path": str(output_folder / "summary.csv") if summary_rows else None,
    }


def main() -> int:
    """Entry point."""
    try:
        args = parse_arguments()
        
        state = RunState(
            output_folder=args.output_folder,
            data_root=args.data_root,
            meta_root=args.meta_root,
            p_active=args.p_active,
            per_trial_thresh=args.per_trial_thresh,
            zz_folder=_build_zz_folder(args.p_active, args.per_trial_thresh),
            method=args.vectorization_method,
            mice=args.mice,
            clip_frames=args.clip_frames,
            cache_dir=args.cache_dir,
            force_recompute=args.force_recompute,
            max_trials=args.max_trials,
            min_id_repetitions=args.min_id_repetitions,
            n_pca_components=args.n_pca_components,
            seed=args.seed,
        )
        
        result = run_pipeline(state)
        
        print(f"\nResult summary:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
        return 0
    
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
