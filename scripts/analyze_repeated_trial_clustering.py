#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Analyze ID-based clustering of repeated trials using zigzag vectorizations.

For each mouse and label independently:
1. Filter trials to only those with repeated IDs (ID appearing >= min_id_repetitions)
2. Normalize vectors and reduce dimensionality with PCA (per label)
3. Cluster with hierarchical clustering (Ward linkage, Euclidean distance)
4. Evaluate clustering via Adjusted Rand Index (ARI) with bootstrap resampling
5. Compute distance matrices and generate visualizations
6. Export results (ARI, distances, plots)
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
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
    max_trials: Optional[int]
    min_id_repetitions: int
    n_pca_components: int
    n_resamplings: int
    seed: int


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
    parser.add_argument("--max-trials", type=_opt_int, default=None, help="Max trials per mouse")

    # Mouse selection
    parser.add_argument(
        "--mice",
        type=_opt_csv_list,
        default=None,
        help="Comma-separated mouse names (default: all discovered)",
    )

    # Clustering parameters
    parser.add_argument(
        "--min-id-repetitions",
        type=int,
        default=7,
        help="Minimum number of times an ID must appear to be included (default 7)",
    )
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=10,
        help="Number of PCA components per label (default 10)",
    )
    parser.add_argument(
        "--n-resamplings",
        type=int,
        default=50,
        help="Number of bootstrap resamplings for ARI evaluation (default 50)",
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
        df_meta: Metadata DataFrame with columns including 'video_ID', 'trial', 'valid_trial', 'valid_response'
        min_id_repetitions: Minimum number of repetitions for an ID to be kept
        
    Returns:
        Filtered DataFrame containing only eligible trials with repeated IDs
    """
    # Check if required columns exist
    if "video_ID" not in df_meta.columns:
        raise ValueError(f"Metadata must contain 'video_ID' column. Found columns: {df_meta.columns.tolist()}")
    
    # Apply eligibility filter first (same as in load_labelled_barcodes)
    df_eligible = _eligible_trials(df_meta)
    
    # Group by ID and count occurrences (on eligible trials only)
    id_counts = df_eligible["video_ID"].value_counts()
    repeated_ids = id_counts[id_counts >= min_id_repetitions].index.tolist()
    
    # Filter to repeated IDs
    df_filtered = df_eligible[df_eligible["video_ID"].isin(repeated_ids)].copy()
    
    return df_filtered


def prepare_label_data(
    xmat: np.ndarray,
    labels: np.ndarray,
    trial_ids: List[int],
    df_meta_filtered: pd.DataFrame,
    label: str,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """Prepare data for a single label: filter, normalize, PCA.
    
    Args:
        xmat: Feature matrix (n_samples x n_features)
        labels: Label array (n_samples,)
        trial_ids: List of trial IDs corresponding to xmat rows
        df_meta_filtered: Filtered metadata with ID column
        label: Label value to filter to
        seed: Random seed for PCA
        
    Returns:
        Dict with 'X_pca', 'X_normalized', 'id_labels', 'trial_ids_label' or None if insufficient data
    """
    # Filter to this label
    mask_label = labels == label
    trial_ids_label_all = [trial_ids[i] for i in range(len(trial_ids)) if mask_label[i]]
    
    if len(trial_ids_label_all) == 0:
        return None
    
    # Build trial_to_id mapping from filtered metadata
    df_meta_label_trials = df_meta_filtered[df_meta_filtered["trial"].isin(trial_ids_label_all)].copy()
    trial_to_id = dict(zip(df_meta_label_trials["trial"].astype(int), df_meta_label_trials["video_ID"]))
    
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
    
    # PCA (per label)
    n_components = min(10, X_normed.shape[0], X_normed.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_normed)
    
    return {
        "X_pca": X_pca,
        "X_normalized": X_normed,
        "id_labels": id_labels,
        "trial_ids_label": trial_ids_label,
        "pca_model": pca,
    }


def evaluate_ari_with_resampling(
    X_pca: np.ndarray,
    id_labels: np.ndarray,
    n_clusters: int,
    n_resamplings: int,
    seed: int,
) -> np.ndarray:
    """Evaluate ARI via bootstrap resampling and clustering with varying random_state.
    
    Args:
        X_pca: PCA-reduced feature matrix (n_samples x n_components)
        id_labels: Ground-truth ID labels (n_samples,)
        n_clusters: Number of clusters (= number of unique IDs)
        n_resamplings: Number of bootstrap resamplings
        seed: Base random seed
        
    Returns:
        Array of ARI values (length n_resamplings)
    """
    ari_values = []
    rng = np.random.default_rng(seed)
    
    for resample_idx in range(n_resamplings):
        # Bootstrap: resample trials with replacement
        indices = rng.choice(len(X_pca), size=len(X_pca), replace=True)
        X_boot = X_pca[indices]
        id_boot = id_labels[indices]
        
        # Cluster with varying random_state
        cluster_seed = seed + resample_idx
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            metric="euclidean",
        )
        # Note: AgglomerativeClustering doesn't have random_state, only fit is deterministic
        # We use the seed to control initial state if needed, but linkage is deterministic
        predicted_clusters = clusterer.fit_predict(X_boot)
        
        # Compute ARI
        ari = adjusted_rand_score(id_boot, predicted_clusters)
        ari_values.append(ari)
    
    return np.array(ari_values)


def compute_distance_matrix(
    X_pca: np.ndarray,
) -> np.ndarray:
    """Compute pairwise Euclidean distances in PCA space.
    
    Args:
        X_pca: PCA-reduced feature matrix
        
    Returns:
        Distance matrix (n_samples x n_samples)
    """
    dist_condensed = pdist(X_pca, metric="euclidean")
    dist_matrix = squareform(dist_condensed)
    return dist_matrix


def plot_heatmap_with_clustering(
    dist_matrix: np.ndarray,
    id_labels: np.ndarray,
    trial_ids: List[int],
    output_path: Path,
) -> None:
    """Plot distance matrix heatmap with trials ordered by ID.
    
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
    ax.set_title("Distance Matrix (ordered by ID)")
    ax.set_xlabel("Trial (sorted by ID)")
    ax.set_ylabel("Trial (sorted by ID)")
    plt.colorbar(im, ax=ax, label="Euclidean Distance")
    
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
    
    ax.set_ylabel("Euclidean Distance (PCA space)")
    ax.set_title("Distribution of Distances by ID Relationship")
    ax.grid(axis="y", alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_ari_distribution(
    ari_values: np.ndarray,
    output_path: Path,
) -> None:
    """Plot distribution of ARI values across resamplings.
    
    Args:
        ari_values: Array of ARI values
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ari_values, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(ari_values.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {ari_values.mean():.3f}")
    ax.set_xlabel("Adjusted Rand Index")
    ax.set_ylabel("Frequency")
    ax.set_title(f"ARI Distribution (N={len(ari_values)} resamplings)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def export_distance_matrix_csv(
    dist_matrix: np.ndarray,
    trial_ids: List[int],
    id_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Export distance matrix to tidy CSV format.
    
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


def run_pipeline(state: RunState) -> Dict[str, Any]:
    """Main analysis pipeline: per-mouse, per-label clustering and distance analysis."""
    
    output_folder = state.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Analysis: ID-based Trial Clustering")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Output folder: {output_folder}")
    print(f"Data root: {state.data_root}")
    print(f"Meta root: {state.meta_root}")
    print(f"P-active: {state.p_active}")
    print(f"Vectorization method: {state.method}")
    print(f"Min ID repetitions: {state.min_id_repetitions}")
    print(f"N PCA components: {state.n_pca_components}")
    print(f"N resamplings: {state.n_resamplings}")
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
            if "video_ID" not in df_meta.columns:
                print(f"  ERROR: Metadata missing 'video_ID' column. Skipping.")
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
                        xmat, labels, trial_ids, df_meta_filtered, label, state.seed
                    )
                    
                    if label_data is None:
                        print(f"      Insufficient data for label. Skipping.")
                        continue
                    
                    X_pca = label_data["X_pca"]
                    id_labels = label_data["id_labels"]
                    trial_ids_label = label_data["trial_ids_label"]
                    
                    n_unique_ids = len(np.unique(id_labels))
                    print(f"      Trials: {len(trial_ids_label)}, Unique IDs: {n_unique_ids}")
                    
                    if n_unique_ids < 2:
                        print(f"      Only one unique ID. Skipping.")
                        continue
                    
                    if len(X_pca) < n_unique_ids:
                        print(f"      Insufficient trials ({len(X_pca)}) for clusters ({n_unique_ids}). Skipping.")
                        continue
                    
                    # Evaluate ARI
                    print(f"      Evaluating ARI with {state.n_resamplings} resamplings...")
                    ari_values = evaluate_ari_with_resampling(
                        X_pca, id_labels, n_unique_ids, state.n_resamplings, state.seed
                    )
                    
                    print(f"      ARI: mean={ari_values.mean():.3f}, std={ari_values.std():.3f}")
                    
                    # Compute distance matrix
                    dist_matrix = compute_distance_matrix(X_pca)
                    
                    # Create output folder
                    label_out_folder.mkdir(parents=True, exist_ok=True)
                    
                    # Export ARI results
                    ari_csv_path = label_out_folder / "ari_resamples.csv"
                    df_ari = pd.DataFrame({"ari": ari_values})
                    df_ari.to_csv(ari_csv_path, index=False)
                    print(f"      Saved ARI results: {ari_csv_path}")
                    
                    # Export distance matrix
                    dist_csv_path = label_out_folder / "distances.csv"
                    export_distance_matrix_csv(dist_matrix, trial_ids_label, id_labels, dist_csv_path)
                    print(f"      Saved distance matrix: {dist_csv_path}")
                    
                    # Generate plots
                    heatmap_path = label_out_folder / "heatmap.png"
                    plot_heatmap_with_clustering(dist_matrix, id_labels, trial_ids_label, heatmap_path)
                    print(f"      Saved heatmap: {heatmap_path}")
                    
                    boxplot_path = label_out_folder / "boxplot_distances.png"
                    plot_boxplot_distances_by_id(dist_matrix, id_labels, boxplot_path)
                    print(f"      Saved boxplot: {boxplot_path}")
                    
                    ari_dist_path = label_out_folder / "ari_distribution.png"
                    plot_ari_distribution(ari_values, ari_dist_path)
                    print(f"      Saved ARI distribution: {ari_dist_path}")
                    
                    # Create summary row
                    summary_rows.append({
                        "mouse": mouse_name,
                        "label": label,
                        "n_trials": len(trial_ids_label),
                        "n_unique_ids": n_unique_ids,
                        "n_clusters": n_unique_ids,
                        "mean_ari": ari_values.mean(),
                        "std_ari": ari_values.std(),
                        "min_ari": ari_values.min(),
                        "max_ari": ari_values.max(),
                    })
                    
                except Exception as exc:
                    print(f"      FAILED: {exc}")
                    traceback.print_exc()
        
        except RuntimeError:
            raise
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
            max_trials=args.max_trials,
            min_id_repetitions=args.min_id_repetitions,
            n_pca_components=args.n_pca_components,
            n_resamplings=args.n_resamplings,
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
