#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cross-mouse similarity analysis for trials sharing the same stimulus ID.

For each pair of mice that share at least one common stimulus ID (after
eligibility + repetition filtering):
1. Normalize + PCA-reduce vectorizations (fit jointly on pooled data, per label)
2. Compute within-mouse-1, within-mouse-2, and cross-mouse distance matrices
3. Plot a combined distance heatmap with trials ordered by label then ID
4. Plot boxplots separating same-ID vs different-ID distances within/cross mice
5. Plot ID-aggregated distance heatmap (one entry per unique ID)
6. Export tidy distance CSV and per-pair summary CSV
"""

from __future__ import annotations

import argparse
import itertools
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
from scipy.spatial.distance import cdist, pdist, squareform
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
    max_trials: Optional[int]
    min_id_repetitions: int
    n_pca_components: Optional[int]
    seed: int


def _opt_none_or_int(value: str) -> Optional[int]:
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
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--p-active", type=int, default=30)
    parser.add_argument("--per-trial-thresh", type=_str2bool, default=False)
    parser.add_argument(
        "--vectorization-method", type=str, default="BettiCurve",
        help="Vectorization method (default BettiCurve)",
    )
    parser.add_argument("--clip-frames", type=_opt_int, default=None)
    parser.add_argument("--cache-dir", type=lambda x: Path(x) if x else None, default=None)
    parser.add_argument("--max-trials", type=_opt_int, default=None)
    parser.add_argument(
        "--mice", type=_opt_csv_list, default=None,
        help="Comma-separated mouse names (default: all discovered)",
    )
    parser.add_argument("--min-id-repetitions", type=int, default=7)
    parser.add_argument(
        "--n-pca-components",
        type=_opt_none_or_int,
        default=10,
        help="Number of PCA components; set to None to disable PCA",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _filter_to_repeated_eligible(
    df_meta: pd.DataFrame,
    min_id_repetitions: int,
) -> pd.DataFrame:
    if "video_ID" not in df_meta.columns:
        raise ValueError(
            f"Metadata missing 'video_ID' column. Found: {df_meta.columns.tolist()}"
        )
    df_elig = _eligible_trials(df_meta)
    id_counts = df_elig["video_ID"].value_counts()
    repeated_ids = id_counts[id_counts >= min_id_repetitions].index
    return df_elig[df_elig["video_ID"].isin(repeated_ids)].copy()


def _load_mouse(state: RunState, mouse_name: str) -> Optional[Dict[str, Any]]:
    try:
        df_meta = load_trial_metadata(state.meta_root, mouse_name)
    except FileNotFoundError as exc:
        print(f"  [{mouse_name}] Metadata not found: {exc}")
        return None

    if "video_ID" not in df_meta.columns:
        print(f"  [{mouse_name}] No 'video_ID' column — skipping.")
        return None

    df_filtered = _filter_to_repeated_eligible(df_meta, state.min_id_repetitions)
    if len(df_filtered) == 0:
        print(f"  [{mouse_name}] No repeated eligible trials — skipping.")
        return None

    barcodes, labels, trial_ids, valid_frames = load_labelled_barcodes(
        state.data_root, state.meta_root, mouse_name,
        state.zz_folder, max_trials=state.max_trials,
    )
    if len(barcodes) == 0:
        print(f"  [{mouse_name}] No labelled barcodes — skipping.")
        return None

    clip_used = state.clip_frames if state.clip_frames is not None else int(valid_frames.min())

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
        print(f"  [{mouse_name}] Using cached vectorization.")
    else:
        print(f"  [{mouse_name}] Computed vectorization.")

    return {
        "xmat": xmat,
        "labels": labels,
        "trial_ids": [int(t) for t in trial_ids],
        "df_filtered": df_filtered,
        "id_set": set(df_filtered["video_ID"].unique()),
        "clip_frames": clip_used,
    }


def _select_trials_for_label(
    mouse_data: Dict[str, Any],
    label: str,
) -> Optional[Dict[str, Any]]:
    """Return feature rows, trial IDs, and ID array for a single label."""
    labels = mouse_data["labels"]
    trial_ids = mouse_data["trial_ids"]
    xmat = mouse_data["xmat"]
    df_filt = mouse_data["df_filtered"]

    mask = labels == label
    tids_all = [trial_ids[i] for i in range(len(trial_ids)) if mask[i]]
    if not tids_all:
        return None

    trial_to_id = dict(zip(df_filt["trial"].astype(int), df_filt["video_ID"]))

    keep_idx, keep_tids, keep_ids = [], [], []
    for i, tid in enumerate(tids_all):
        if tid in trial_to_id:
            keep_idx.append(i)
            keep_tids.append(tid)
            keep_ids.append(trial_to_id[tid])

    if not keep_tids:
        return None

    return {
        "xmat_raw": xmat[mask][keep_idx],
        "trial_ids": keep_tids,
        "id_array": np.array(keep_ids),
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class _Tee:
    """Simple stream tee to duplicate stdout/stderr into a file."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

def _normalize_and_pca(
    X: np.ndarray,
    n_components: Optional[int],
    seed: int,
    pca_model: Optional[PCA] = None,
    scaler_model: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, Optional[PCA], StandardScaler]:
    """StandardScaler → L2 norm → optional PCA with optional pre-fitted models."""
    X_clean = np.nan_to_num(X)
    if scaler_model is None:
        scaler_model = StandardScaler()
        X_scaled = scaler_model.fit_transform(X_clean)
    else:
        X_scaled = scaler_model.transform(X_clean)
    X_scaled = np.nan_to_num(X_scaled)

    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X_scaled / norms

    if n_components is None:
        return X_normed, None, scaler_model

    n_comp = max(1, min(n_components, X_normed.shape[0] - 1, X_normed.shape[1]))
    if pca_model is None:
        pca_model = PCA(n_components=n_comp, random_state=seed)
        X_pca = pca_model.fit_transform(X_normed)
    else:
        X_pca = pca_model.transform(X_normed)

    return X_pca, pca_model, scaler_model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_combined_heatmap(
    dist_full: np.ndarray,
    mouse_tags: np.ndarray,
    id_arrays: np.ndarray,
    label_arrays: np.ndarray,
    mouse1: str,
    mouse2: str,
    output_path: Path,
) -> None:
    """(N1+N2)×(N1+N2) heatmap sorted by (label, ID, mouse); white dashed separator."""
    sort_keys = list(zip(label_arrays.tolist(), id_arrays.tolist(), mouse_tags.tolist()))
    sort_idx = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

    d_sorted = dist_full[np.ix_(sort_idx, sort_idx)]
    tags_sorted = mouse_tags[sort_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(d_sorted, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Euclidean Distance (PCA space)")

    n1_sorted = int((tags_sorted == "m1").sum())
    if 0 < n1_sorted < len(tags_sorted):
        ax.axhline(n1_sorted - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(n1_sorted - 0.5, color="white", linewidth=1.5, linestyle="--")

    m1_short = mouse1.split("-")[0]
    m2_short = mouse2.split("-")[0]
    ax.set_title(
        f"Distance matrix: {m1_short} vs {m2_short}\n"
        "(ordered by label → ID; dashed line separates mice)"
    )
    ax.set_xlabel("Trials")
    ax.set_ylabel("Trials")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_boxplot_by_comparison(
    records: List[Dict[str, Any]],
    mouse1: str,
    mouse2: str,
    output_path: Path,
) -> None:
    """Per-label boxplot with 6 groups: within/cross × same-ID/different-ID."""
    df = pd.DataFrame(records)
    if df.empty:
        return

    labels_present = sorted(df["label"].unique())
    m1_short = mouse1.split("-")[0]
    m2_short = mouse2.split("-")[0]
    # Required order:
    # 1) mouse1 within ID, 2) mouse1 between ID,
    # 3) mouse2 within ID, 4) mouse2 between ID,
    # 5) cross-mouse within ID, 6) cross-mouse between ID.
    groups = [
        ("within_m1", "same", f"Within {m1_short}\nSame ID"),
        ("within_m1", "different", f"Within {m1_short}\nDiff ID"),
        ("within_m2", "same", f"Within {m2_short}\nSame ID"),
        ("within_m2", "different", f"Within {m2_short}\nDiff ID"),
        ("cross", "same", "Cross mice\nSame ID"),
        ("cross", "different", "Cross mice\nDiff ID"),
    ]
    group_colors = [
        "steelblue", "cornflowerblue",
        "darkorange", "lightsalmon",
        "seagreen", "mediumseagreen",
    ]
    # Add visible gaps between mouse1, mouse2, and cross-mouse groups.
    group_positions = [1.0, 2.0, 4.0, 5.0, 7.0, 8.0]

    n_labels = len(labels_present)
    fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 6), squeeze=False)

    for col, lbl in enumerate(labels_present):
        ax = axes[0, col]
        data_by_group: List[np.ndarray] = []
        for comp_type, id_relation, _label in groups:
            vals = df[
                (df["label"] == lbl)
                & (df["comparison_type"] == comp_type)
                & (df["id_relation"] == id_relation)
            ]["distance"].values
            data_by_group.append(vals)

        non_empty = [
            (p, d, idx)
            for idx, (p, d) in enumerate(zip(group_positions, data_by_group))
            if len(d) > 0
        ]

        if not non_empty:
            ax.set_title(f"Label: {lbl}\n(no data)")
            continue

        bp = ax.boxplot(
            [d for _, d, _ in non_empty],
            positions=[p for p, _, _ in non_empty],
            patch_artist=True,
            widths=0.55,
        )
        for patch, (_pos, _vals, idx) in zip(bp["boxes"], non_empty):
            patch.set_facecolor(group_colors[idx])
            patch.set_alpha(0.8)

        ax.set_xticks(group_positions)
        ax.set_xticklabels([entry[2] for entry in groups], fontsize=8, rotation=25, ha="right")
        ax.set_xlim(0.4, 8.6)
        # Visual separators for the larger inter-group gaps.
        ax.axvline(3.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.axvline(6.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_title(f"Label: {lbl}")
        ax.set_ylabel("Euclidean Distance (PCA space)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Distances: {m1_short} vs {m2_short}", fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def _build_id_aggregated_outputs(
    records: List[Dict[str, Any]],
) -> Tuple[List[Tuple[str, Any]], List[str], int, int, np.ndarray, pd.DataFrame]:
    df = pd.DataFrame(records)
    if df.empty:
        return [], [], 0, 0, np.empty((0, 0), dtype=float), pd.DataFrame()

    ids_m1 = set()
    ids_m2 = set()

    for row in df.itertuples(index=False):
        comp_type = getattr(row, "comparison_type")
        if comp_type == "within_m1":
            ids_m1.add(getattr(row, "id_label_1"))
            ids_m1.add(getattr(row, "id_label_2"))
        elif comp_type == "within_m2":
            ids_m2.add(getattr(row, "id_label_1"))
            ids_m2.add(getattr(row, "id_label_2"))
        elif comp_type == "cross":
            ids_m1.add(getattr(row, "id_label_1"))
            ids_m2.add(getattr(row, "id_label_2"))

    ids_m1_sorted = sorted(ids_m1)
    ids_m2_sorted = sorted(ids_m2)
    n1 = len(ids_m1_sorted)
    n2 = len(ids_m2_sorted)

    axis_keys = [("m1", id_val) for id_val in ids_m1_sorted] + [
        ("m2", id_val) for id_val in ids_m2_sorted
    ]
    axis_labels = [str(id_val) for _, id_val in axis_keys]
    key_to_idx = {axis_key: i for i, axis_key in enumerate(axis_keys)}

    pair_values: Dict[Tuple[Tuple[str, Any], Tuple[str, Any]], List[float]] = {}
    for row in df.itertuples(index=False):
        comp_type = getattr(row, "comparison_type")
        id1 = getattr(row, "id_label_1")
        id2 = getattr(row, "id_label_2")
        distance = float(getattr(row, "distance"))

        if comp_type == "within_m1":
            key1 = ("m1", id1)
            key2 = ("m1", id2)
        elif comp_type == "within_m2":
            key1 = ("m2", id1)
            key2 = ("m2", id2)
        else:
            key1 = ("m1", id1)
            key2 = ("m2", id2)

        key = (key1, key2) if key1 <= key2 else (key2, key1)
        pair_values.setdefault(key, []).append(distance)

    n_total = len(axis_keys)
    id_dist = np.full((n_total, n_total), np.nan, dtype=float)
    csv_rows: List[Dict[str, Any]] = []
    for (key1, key2), vals in sorted(pair_values.items()):
        if key1 not in key_to_idx or key2 not in key_to_idx:
            continue
        i = key_to_idx[key1]
        j = key_to_idx[key2]
        mean_val = float(np.mean(vals))
        id_dist[i, j] = mean_val
        id_dist[j, i] = mean_val
        csv_rows.append({
            "mouse_block_1": key1[0],
            "id_label_1": key1[1],
            "mouse_block_2": key2[0],
            "id_label_2": key2[1],
            "mean_distance": mean_val,
            "n_trial_pairs": len(vals),
        })

    return axis_keys, axis_labels, n1, n2, id_dist, pd.DataFrame(csv_rows)


def _write_id_aggregated_csv(records: List[Dict[str, Any]], output_path: Path) -> bool:
    _, _, _, _, _, df_out = _build_id_aggregated_outputs(records)
    if df_out.empty:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    return True


def _plot_id_aggregated_heatmap(
    records: List[Dict[str, Any]],
    mouse1: str,
    mouse2: str,
    output_path: Path,
) -> None:
    """Plot unique-ID x unique-ID heatmap by averaging distances per ID pair.
    
    Separates IDs by mouse: rows/cols ordered as [mouse1_IDs, mouse2_IDs].
    Adds a white dashed separator line between the two mouse blocks.
    """
    axis_keys, axis_labels, n1, n2, id_dist, _df_out = _build_id_aggregated_outputs(records)
    if len(axis_keys) == 0:
        return
    n_total = len(axis_keys)

    masked = np.ma.masked_invalid(id_dist)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Euclidean Distance (PCA space)")
    
    # Add dashed separator between mouse1 and mouse2 IDs (if both present)
    if 0 < n1 < n_total:
        ax.axhline(n1 - 0.5, color="white", linewidth=2, linestyle="--")
        ax.axvline(n1 - 0.5, color="white", linewidth=2, linestyle="--")

    ax.set_xticks(np.arange(n_total))
    ax.set_yticks(np.arange(n_total))
    ax.set_xticklabels(axis_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(axis_labels, fontsize=8)
    ax.set_xlabel(f"Stimulus ID ({mouse1.split('-')[0]} IDs first)")
    ax.set_ylabel(f"Stimulus ID ({mouse1.split('-')[0]} IDs first)")
    
    m1_short = mouse1.split("-")[0]
    m2_short = mouse2.split("-")[0]
    ax.set_title(
        f"ID-aggregated distance matrix: {m1_short} vs {m2_short}\n"
        f"({n1} {m1_short} IDs × {n2} {m2_short} IDs; dashed line separates mice)\n"
        "(each cell = mean distance over all trial pairs for the ID pair)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Pair analysis
# ---------------------------------------------------------------------------

def _analyse_pair(
    state: RunState,
    mouse1: str,
    mouse2: str,
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    common_ids: set,
    output_folder: Path,
) -> List[Dict[str, Any]]:
    m1_short = mouse1.split("-")[0]
    m2_short = mouse2.split("-")[0]
    pair_tag = f"{m1_short}_vs_{m2_short}"
    pair_folder = output_folder / pair_tag
    pair_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n  === Pair: {pair_tag} ===")
    print(f"  Common IDs: {len(common_ids)}")

    df1 = data1["df_filtered"]
    df2 = data2["df_filtered"]
    labels1 = set(df1[df1["video_ID"].isin(common_ids)]["label"].unique())
    labels2 = set(df2[df2["video_ID"].isin(common_ids)]["label"].unique())
    shared_labels = sorted(labels1 & labels2)

    if not shared_labels:
        print("  No shared labels for common IDs — skipping pair.")
        return []

    print(f"  Shared labels: {shared_labels}")

    summary_rows: List[Dict[str, Any]] = []
    all_distance_records: List[Dict[str, Any]] = []
    heatmap_blocks: List[Dict[str, Any]] = []

    for label in shared_labels:
        print(f"  Label: {label}")

        sub1 = _select_trials_for_label(data1, label)
        sub2 = _select_trials_for_label(data2, label)

        if sub1 is None or sub2 is None:
            print("    Insufficient data in one mouse — skipping label.")
            continue

        common_ids_arr = np.array(list(common_ids))
        mask1 = np.isin(sub1["id_array"], common_ids_arr)
        mask2 = np.isin(sub2["id_array"], common_ids_arr)

        X1_raw = sub1["xmat_raw"][mask1]
        ids1 = sub1["id_array"][mask1]
        tids1 = [sub1["trial_ids"][i] for i in range(len(sub1["trial_ids"])) if mask1[i]]

        X2_raw = sub2["xmat_raw"][mask2]
        ids2 = sub2["id_array"][mask2]
        tids2 = [sub2["trial_ids"][i] for i in range(len(sub2["trial_ids"])) if mask2[i]]

        if len(X1_raw) == 0 or len(X2_raw) == 0:
            print("    No shared-ID trials remaining — skipping label.")
            continue

        n1, n2 = len(X1_raw), len(X2_raw)
        print(f"    Trials — m1: {n1}, m2: {n2}")

        X_pooled = np.vstack([X1_raw, X2_raw])
        n_comp = state.n_pca_components
        if n_comp is not None:
            n_comp = max(1, min(n_comp, X_pooled.shape[0] - 1, X_pooled.shape[1]))
            if X_pooled.shape[0] < 2:
                print("    Too few pooled samples for PCA — skipping label.")
                continue

        # Fit preprocessing jointly on pooled data to place both mice in the same space.
        _, pca_model, scaler_model = _normalize_and_pca(X_pooled, n_comp, state.seed)
        X1_proj, _, _ = _normalize_and_pca(
            X1_raw, n_comp, state.seed, pca_model=pca_model, scaler_model=scaler_model
        )
        X2_proj, _, _ = _normalize_and_pca(
            X2_raw, n_comp, state.seed, pca_model=pca_model, scaler_model=scaler_model
        )

        dist_w1 = squareform(pdist(X1_proj, metric="euclidean"))
        dist_w2 = squareform(pdist(X2_proj, metric="euclidean"))
        dist_cross = cdist(X1_proj, X2_proj, metric="euclidean")

        dist_full = np.zeros((n1 + n2, n1 + n2))
        dist_full[:n1, :n1] = dist_w1
        dist_full[n1:, n1:] = dist_w2
        dist_full[:n1, n1:] = dist_cross
        dist_full[n1:, :n1] = dist_cross.T

        heatmap_blocks.append({
            "dist": dist_full,
            "mouse_tags": np.array(["m1"] * n1 + ["m2"] * n2),
            "id_arrays": np.concatenate([ids1, ids2]),
            "label_arrays": np.array([label] * (n1 + n2)),
        })

        # Accumulate tidy distance records
        for i in range(n1):
            for j in range(i + 1, n1):
                all_distance_records.append({
                    "mouse_1": mouse1, "mouse_2": mouse1,
                    "trial_id_1": tids1[i], "trial_id_2": tids1[j],
                    "id_label_1": ids1[i], "id_label_2": ids1[j],
                    "label": label, "comparison_type": "within_m1",
                    "id_relation": "same" if ids1[i] == ids1[j] else "different",
                    "distance": float(dist_w1[i, j]),
                })
        for i in range(n2):
            for j in range(i + 1, n2):
                all_distance_records.append({
                    "mouse_1": mouse2, "mouse_2": mouse2,
                    "trial_id_1": tids2[i], "trial_id_2": tids2[j],
                    "id_label_1": ids2[i], "id_label_2": ids2[j],
                    "label": label, "comparison_type": "within_m2",
                    "id_relation": "same" if ids2[i] == ids2[j] else "different",
                    "distance": float(dist_w2[i, j]),
                })
        for i in range(n1):
            for j in range(n2):
                all_distance_records.append({
                    "mouse_1": mouse1, "mouse_2": mouse2,
                    "trial_id_1": tids1[i], "trial_id_2": tids2[j],
                    "id_label_1": ids1[i], "id_label_2": ids2[j],
                    "label": label, "comparison_type": "cross",
                    "id_relation": "same" if ids1[i] == ids2[j] else "different",
                    "distance": float(dist_cross[i, j]),
                })

        w1_upper = dist_w1[np.triu_indices(n1, k=1)] if n1 > 1 else np.array([float("nan")])
        w2_upper = dist_w2[np.triu_indices(n2, k=1)] if n2 > 1 else np.array([float("nan")])
        summary_rows.append({
            "mouse_1": mouse1, "mouse_2": mouse2, "label": label,
            "n_trials_m1": n1, "n_trials_m2": n2,
            "n_common_ids": len(set(ids1.tolist()) & set(ids2.tolist())),
            "mean_dist_within_m1": float(np.nanmean(w1_upper)),
            "mean_dist_within_m2": float(np.nanmean(w2_upper)),
            "mean_dist_cross": float(dist_cross.mean()),
        })

    if not heatmap_blocks:
        return summary_rows

    # Build combined (across labels) distance matrix for heatmap
    n_total = sum(b["dist"].shape[0] for b in heatmap_blocks)
    dist_combined = np.zeros((n_total, n_total))
    cur = 0
    for b in heatmap_blocks:
        nb = b["dist"].shape[0]
        dist_combined[cur : cur + nb, cur : cur + nb] = b["dist"]
        cur += nb

    _plot_combined_heatmap(
        dist_combined,
        np.concatenate([b["mouse_tags"] for b in heatmap_blocks]),
        np.concatenate([b["id_arrays"] for b in heatmap_blocks]),
        np.concatenate([b["label_arrays"] for b in heatmap_blocks]),
        mouse1, mouse2,
        pair_folder / "heatmap_combined.png",
    )
    print(f"  Saved heatmap: {pair_folder / 'heatmap_combined.png'}")

    _plot_boxplot_by_comparison(
        all_distance_records, mouse1, mouse2,
        pair_folder / "boxplot_distances.png",
    )
    print(f"  Saved boxplot: {pair_folder / 'boxplot_distances.png'}")

    _plot_id_aggregated_heatmap(
        all_distance_records,
        mouse1,
        mouse2,
        pair_folder / "heatmap_ids_aggregated.png",
    )
    print(f"  Saved ID-aggregated heatmap: {pair_folder / 'heatmap_ids_aggregated.png'}")

    if _write_id_aggregated_csv(all_distance_records, pair_folder / "id_distances.csv"):
        print(f"  Saved ID distances CSV: {pair_folder / 'id_distances.csv'}")

    if all_distance_records:
        pd.DataFrame(all_distance_records).to_csv(pair_folder / "distances.csv", index=False)
        print(f"  Saved distances CSV: {pair_folder / 'distances.csv'}")

    return summary_rows


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(state: RunState) -> None:
    state.output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("Analysis: Cross-mouse ID similarity")
    print(f"{'=' * 70}")
    print(f"Started:              {datetime.now().isoformat()}")
    print(f"Output folder:        {state.output_folder}")
    print(f"Data root:            {state.data_root}")
    print(f"Meta root:            {state.meta_root}")
    print(f"Vectorization method: {state.method}")
    print(f"P-active:             {state.p_active}")
    print(f"Min ID repetitions:   {state.min_id_repetitions}")
    print(f"N PCA components:     {state.n_pca_components if state.n_pca_components is not None else 'None (PCA disabled)'}")
    print(f"Seed:                 {state.seed}")

    discovered_mice = _discover_mice(state.data_root)
    selected_mice = state.mice if state.mice is not None else discovered_mice
    selected_mice = [m for m in selected_mice if m in discovered_mice]

    if len(selected_mice) < 2:
        raise RuntimeError(
            f"Need at least 2 valid mice. Got: {selected_mice}"
        )

    print(f"\nSelected mice ({len(selected_mice)}): {selected_mice}")
    print("\n--- Loading per-mouse data ---")

    mouse_data: Dict[str, Dict[str, Any]] = {}
    for mouse_name in selected_mice:
        print(f"\n  Mouse: {mouse_name}")
        data = _load_mouse(state, mouse_name)
        if data is None:
            continue
        mouse_data[mouse_name] = data
        print(f"    Unique IDs (filtered): {len(data['id_set'])}")
        print(f"    Feature matrix:        {data['xmat'].shape}")

    if len(mouse_data) < 2:
        raise RuntimeError("Fewer than 2 usable mice after data loading.")

    print("\n--- Enumerating mouse pairs ---")
    all_summary: List[Dict[str, Any]] = []
    pair_count = 0

    for (m1, d1), (m2, d2) in itertools.combinations(mouse_data.items(), 2):
        common_ids = d1["id_set"] & d2["id_set"]
        
        if not common_ids:
            m1_short = m1.split("-")[0]
            m2_short = m2.split("-")[0]
            print(
                f"  {m1_short} × {m2_short}: "
                "no common IDs — skipping."
            )
            continue
        
        m1_short = m1.split("-")[0]
        m2_short = m2.split("-")[0]
        pair_tag = f"{m1_short}_vs_{m2_short}"
        pair_folder = state.output_folder / pair_tag
        pair_folder.mkdir(parents=True, exist_ok=True)
        pair_log = pair_folder / "run.log"
        
        print(
            f"  {m1_short} × {m2_short}: "
            f"{len(common_ids)} common IDs"
        )
        try:
            with pair_log.open("w", encoding="utf-8") as log_fh:
                tee_stream = _Tee(sys.stdout, log_fh)
                with redirect_stdout(tee_stream), redirect_stderr(tee_stream):
                    rows = _analyse_pair(state, m1, m2, d1, d2, common_ids, state.output_folder)
            all_summary.extend(rows)
            pair_count += 1
            print(f"  Pair log saved: {pair_log}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()

    print(f"\n--- Analysed {pair_count} mouse pair(s) ---")

    if all_summary:
        df_summary = pd.DataFrame(all_summary)
        summary_path = state.output_folder / "summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
        print(df_summary.to_string())
    else:
        print("\nNo results to summarize.")

    print(f"\nCompleted: {datetime.now().isoformat()}")


def main() -> int:
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
            seed=args.seed,
        )
        run_pipeline(state)
        return 0
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
