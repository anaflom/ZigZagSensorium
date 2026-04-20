#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified shuffle ablation study for both within-mouse and cross-mouse classification.

For each randomization iteration:
1. Load grids for all trials
2. Shuffle grids along selected dimension (time or spatial)
3. Compute zigzag persistence
4. Run within-mouse StratifiedKFold CV (LogReg + 3D-CNN)
5. Run cross-mouse LOMO CV (LogReg + 3D-CNN)
6. Save results and aggregate metrics
7. Clean up shuffled data

Results are aggregated with shuffle iteration as a column/key.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import random
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset

from classification_models import (
    CNN3D,
    GridTrialDataset,
    VectorDataset,
    run_logreg_cv,
    run_nn_cv,
    train_eval_logreg,
)
from shuffle_utils import (
    compute_threshold_from_grid_sample,
    compute_zigzag_from_grid,
    shuffle_grid_phase,
    shuffle_grid_spatial_dimensions,
    shuffle_grid_time_dimension,
    validate_mice_for_ablation,
)
from utils import (
    _build_zz_folder,
    _discover_mice,
    _opt_csv_list,
    _opt_int,
    _resolve_mouse_cache_dir,
    _str2bool,
    _short_mouse_name,
    build_vectorization_cache_stem,
    create_vectorization,
    load_labelled_barcodes,
    load_labelled_grid_paths,
    load_vectorization_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RunState:
    """Configuration state for ablation study."""

    output_folder: Path
    data_root: Path
    meta_root: Path
    p_active: int
    per_trial_thresh: bool
    zz_folder: str
    vectorization_method: str
    mice: Optional[List[str]]
    clip_frames: Optional[int]
    grid_subdir: str
    cache_dir: Optional[Path]
    force_recompute: bool
    n_shuffles: int
    shuffle_type: str
    skip_within_mouse: bool
    skip_cross_mouse: bool
    skip_existing_shuffles: bool
    max_trials: Optional[int]
    batch_size_vec: int
    batch_size_grid: int
    epochs_cnn3d: int
    lr_cnn3d: float
    weight_decay: float
    early_stop_patience: int
    seed: int
    device: str
    num_workers_dl: int
    max_dim: int


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shuffle_cache_stem(base_stem: str, shuffle_type: str, shuffle_id: int) -> str:
    """Build a deterministic cache stem that encodes shuffle type and iteration."""
    return f"{base_stem}_{shuffle_type}_shuffle{shuffle_id:04d}"


def _find_missing_shuffle_ids(
    cache_dir: Path,
    base_stem: str,
    shuffle_type: str,
    n_shuffles: int,
) -> List[int]:
    """Return the shuffle IDs whose vectorization cache does not yet exist."""
    missing = []
    for sid in range(n_shuffles):
        stem = _shuffle_cache_stem(base_stem, shuffle_type, sid)
        if not (cache_dir / f"{stem}.npz").exists():
            missing.append(sid)
    return missing


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(description="Shuffle ablation classification study")

    # Data paths
    parser.add_argument("--output-folder", type=Path, required=True, help="Output folder for results")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder with neural data")
    parser.add_argument("--meta-root", type=Path, required=True, help="Root folder with metadata")

    # Ablation/vectorization parameters
    parser.add_argument("--p-active", type=int, default=30, help="Percentile threshold for zigzag")
    parser.add_argument("--per-trial-thresh", type=_str2bool, default=False, help="Per-trial threshold (True/False)")
    parser.add_argument("--zz-folder", type=str, default=None, help="Zigzag folder name (auto if None)")
    parser.add_argument("--vectorization-method", type=str, default="BettiCurve", help="Zigzag vectorization method")
    parser.add_argument("--mice", type=_opt_csv_list, default=None, help="Comma-separated mouse names (None=all)")
    parser.add_argument("--clip-frames", type=_opt_int, default=None, help="Clip frames (None=auto-detect)")
    parser.add_argument("--grid-subdir", type=str, default="trials_grid", help="Grid subdirectory")
    parser.add_argument("--cache-dir", type=lambda x: Path(x) if x else None, default=None, help="Cache folder")
    parser.add_argument("--force-recompute", type=_str2bool, default=False, help="Force recompute cache")
    parser.add_argument("--max-trials", type=_opt_int, default=None, help="Max trials per mouse")

    # Shuffle parameters
    parser.add_argument("--n-shuffles", type=int, default=5, help="Number of shuffle iterations")
    parser.add_argument(
        "--shuffle-type",
        type=str,
        choices=["time", "spatial", "phase"],
        default="time",
        help="Shuffle mode: 'time' permutes per-voxel time axis; 'spatial' permutes 3D positions and reuses mapping across frames; 'phase' applies FFT phase shifting",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2,
        help="Maximum homology dimension for zigzag persistence (0=components, 1=loops, 2=voids)",
    )
    parser.add_argument(
        "--skip-within-mouse",
        action="store_true",
        default=False,
        help="Skip within-mouse CV analysis",
    )
    parser.add_argument(
        "--skip-cross-mouse",
        action="store_true",
        default=False,
        help="Skip cross-mouse LOMO analysis",
    )
    parser.add_argument(
        "--skip-existing-shuffles",
        type=_str2bool,
        default=True,
        help="If True, re-use already-cached shuffle vectorizations instead of recomputing (default True)",
    )

    # Training parameters
    parser.add_argument("--batch-size-grid", type=int, default=16, help="Batch size for grid data")
    parser.add_argument("--epochs-cnn3d", type=int, default=100, help="Epochs for 3D CNN")
    parser.add_argument("--lr-cnn3d", type=float, default=0.001, help="Learning rate for 3D CNN")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--num-workers-dl", type=int, default=4, help="DataLoader workers")

    return parser


def run_within_mouse_analysis(
    state: RunState,
    eligible_mice: List[str],
    shuffled_grids_by_mouse: Dict[str, Dict[str, Any]],
    device: torch.device,
    shuffle_id: int,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """Run within-mouse ablation for this shuffle iteration.
    
    Returns:
        (per_mouse results dict, confusion matrix payload dict)
    """
    per_mouse: Dict[str, Dict[str, object]] = {}
    confusion_payload: Dict[str, Dict[str, object]] = {}

    model_order = ["logreg", "cnn3d"]

    for mouse_name in eligible_mice:
        print(f"  Within-mouse: {mouse_name} ...", end=" ", flush=True)
        try:
            data = shuffled_grids_by_mouse[mouse_name]
            barcodes = data["barcodes"]
            labels = data["labels"]
            trial_ids = data["trial_ids"]
            valid_frames = data["valid_frames"]
            grid_paths = data["grid_paths"]
            x_vec = data["x_vec"]
            clip_used = data["clip_used"]

            if len(np.unique(labels)) < 2:
                print("skipped (only 1 class)")
                continue

            # Prepare for CV
            le = LabelEncoder().fit(labels)
            y_int = le.transform(labels)
            class_labels = list(le.classes_)

            # StratifiedKFold
            cv = StratifiedKFold(n_splits=min(5, int(np.bincount(y_int).min())), shuffle=True, random_state=state.seed)
            splits = list(cv.split(np.zeros(len(y_int)), y_int))

            # LogReg
            logreg_metrics, logreg_pred = run_logreg_cv(x_vec, y_int, splits)

            # 3D-CNN
            grid_dataset = GridTrialDataset(
                grid_paths=grid_paths,
                y=y_int,
                valid_frames=valid_frames,
                clip_frames=int(clip_used),
            )

            def build_grid_dataset(train_idx: np.ndarray, val_idx: np.ndarray):
                return Subset(grid_dataset, train_idx), Subset(grid_dataset, val_idx)

            cnn3d_metrics, cnn3d_pred = run_nn_cv(
                make_model=lambda: CNN3D(
                    n_classes=len(class_labels),
                    in_channels=grid_dataset.in_channels,
                ),
                train_dataset_builder=build_grid_dataset,
                y_int=y_int,
                splits=splits,
                epochs=state.epochs_cnn3d,
                lr=state.lr_cnn3d,
                batch_size=state.batch_size_grid,
                patience=state.early_stop_patience,
                weight_decay=state.weight_decay,
                device=device,
                num_workers=state.num_workers_dl,
            )

            best_model = max(model_order, key=lambda m: {"logreg": logreg_metrics, "cnn3d": cnn3d_metrics}[m]["mean_f1"])

            per_mouse[mouse_name] = {
                "shuffle_id": shuffle_id,
                "n_trials": int(len(y_int)),
                "n_features": int(x_vec.shape[1]),
                "clip_frames": int(clip_used),
                "class_labels": class_labels,
                "cv_folds": int(len(splits)),
                "best_model": best_model,
                "models": {
                    "logreg": logreg_metrics,
                    "cnn3d": cnn3d_metrics,
                },
            }

            model_cms: Dict[str, np.ndarray] = {
                mk: confusion_matrix(y_int, {"logreg": logreg_pred, "cnn3d": cnn3d_pred}[mk], labels=np.arange(len(class_labels)))
                for mk in model_order
            }
            confusion_payload[mouse_name] = {
                "labels": class_labels,
                "cms": model_cms,
                "best_model": best_model,
            }

            print(f"ok (best={best_model}:{logreg_metrics['mean_f1']:.3f})")
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()

    return per_mouse, confusion_payload


def run_cross_mouse_analysis(
    state: RunState,
    eligible_mice: List[str],
    shuffled_grids_by_mouse: Dict[str, Dict[str, Any]],
    device: torch.device,
    shuffle_id: int,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    """Run cross-mouse LOMO ablation for this shuffle iteration.
    
    Returns:
        (per fold results dict, confusion matrix payload dict)
    """
    per_fold: Dict[str, Dict[str, object]] = {}
    confusion_payload: Dict[str, Dict[str, object]] = {}

    model_order = ["logreg", "cnn3d"]

    for test_mouse in eligible_mice:
        candidate_train_mice = [m for m in eligible_mice if m != test_mouse]
        print(f"  Cross-mouse: test={test_mouse}, train_pool={len(candidate_train_mice)} ...", end=" ", flush=True)

        try:
            test_data = shuffled_grids_by_mouse[test_mouse]
            test_labels_set = set(np.unique(test_data["labels"]).tolist())
            train_label_union = set()
            for mouse_name in candidate_train_mice:
                train_label_union.update(np.unique(shuffled_grids_by_mouse[mouse_name]["labels"]).tolist())

            shared_labels = sorted(test_labels_set & train_label_union)
            if len(shared_labels) < 2:
                print(f"skipped (shared_labels={len(shared_labels)} < 2)")
                continue

            # Prepare test data
            encoder = LabelEncoder().fit(shared_labels)
            test_mask = np.isin(test_data["labels"], shared_labels)
            if not np.any(test_mask):
                print("skipped (no test samples)")
                continue

            test_x_vec = np.asarray(test_data["x_vec"][test_mask], dtype=np.float64)
            test_labels = np.asarray(test_data["labels"][test_mask])
            test_y = encoder.transform(test_labels)
            test_grid_paths = [test_data["grid_paths"][i] for i in np.where(test_mask)[0]]
            test_grid_frames = np.asarray(test_data["valid_frames"][test_mask], dtype=np.int64)
            clip_used = test_data["clip_used"]

            # Prepare training data
            train_x_parts: List[np.ndarray] = []
            train_y_parts: List[np.ndarray] = []
            train_grid_paths: List[Path] = []
            train_grid_frames_list: List[int] = []
            active_train_mice: List[str] = []

            for mouse_name in candidate_train_mice:
                train_data = shuffled_grids_by_mouse[mouse_name]
                mask = np.isin(train_data["labels"], shared_labels)
                if not np.any(mask):
                    continue
                active_train_mice.append(mouse_name)
                train_x_parts.append(np.asarray(train_data["x_vec"][mask], dtype=np.float64))
                train_y_parts.append(encoder.transform(np.asarray(train_data["labels"][mask])))
                idxs = np.where(mask)[0]
                train_grid_paths.extend([train_data["grid_paths"][i] for i in idxs])
                train_grid_frames_list.extend(np.asarray(train_data["valid_frames"][mask], dtype=np.int64).tolist())

            if not train_x_parts:
                print("skipped (no training samples)")
                continue

            train_x_vec = np.concatenate(train_x_parts, axis=0)
            train_y = np.concatenate(train_y_parts, axis=0)
            train_grid_frames = np.asarray(train_grid_frames_list, dtype=np.int64)

            if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
                print("skipped (collapsed class)")
                continue

            # LogReg
            logreg_metrics, logreg_pred = train_eval_logreg(train_x_vec, train_y, test_x_vec, test_y)

            # 3D-CNN
            train_grid_ds = GridTrialDataset(
                grid_paths=train_grid_paths,
                y=train_y,
                valid_frames=train_grid_frames,
                clip_frames=int(clip_used),
            )
            test_grid_ds = GridTrialDataset(
                grid_paths=test_grid_paths,
                y=test_y,
                valid_frames=test_grid_frames,
                clip_frames=int(clip_used),
            )
            cnn3d_metrics, cnn3d_pred = run_nn_cv(
                make_model=lambda: CNN3D(
                    n_classes=len(shared_labels),
                    in_channels=test_grid_ds.in_channels,
                ),
                train_dataset_builder=lambda tr_idx, _: (Subset(train_grid_ds, tr_idx), test_grid_ds),
                y_int=test_y,
                splits=[(np.arange(len(train_y)), np.arange(len(test_y)))],
                epochs=state.epochs_cnn3d,
                lr=state.lr_cnn3d,
                batch_size=state.batch_size_grid,
                patience=state.early_stop_patience,
                weight_decay=state.weight_decay,
                device=device,
                num_workers=state.num_workers_dl,
            )

            best_model = max(model_order, key=lambda m: {"logreg": logreg_metrics, "cnn3d": cnn3d_metrics}[m]["macro_f1"])

            per_fold[test_mouse] = {
                "shuffle_id": shuffle_id,
                "test_mouse": test_mouse,
                "train_mice": active_train_mice,
                "n_train_mice": len(active_train_mice),
                "n_train_trials": int(len(train_y)),
                "n_test_trials": int(len(test_y)),
                "n_features": int(train_x_vec.shape[1]),
                "clip_frames": int(clip_used),
                "shared_labels": shared_labels,
                "n_classes": int(len(shared_labels)),
                "best_model": best_model,
                "models": {
                    "logreg": logreg_metrics,
                    "cnn3d": cnn3d_metrics,
                },
            }

            model_cms: Dict[str, np.ndarray] = {
                mk: confusion_matrix(test_y, {"logreg": logreg_pred, "cnn3d": cnn3d_pred}[mk], labels=np.arange(len(shared_labels)))
                for mk in model_order
            }
            confusion_payload[test_mouse] = {
                "labels": shared_labels,
                "cms": model_cms,
                "best_model": best_model,
            }

            print(f"ok (best={best_model}:{logreg_metrics['macro_f1']:.3f})")
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()

    return per_fold, confusion_payload


def run_pipeline(state: RunState) -> Dict[str, object]:
    """Run the unified shuffle ablation pipeline."""
    _set_seed(state.seed)
    device = _resolve_device(state.device)

    if state.skip_within_mouse and state.skip_cross_mouse:
        raise ValueError("Cannot skip both within-mouse and cross-mouse analyses")

    output_folder = state.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    figures_dir = output_folder / "figures"
    logs_dir = output_folder / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"

    with open(log_path, "w", encoding="utf-8") as log_fp, redirect_stderr(log_fp), redirect_stdout(log_fp):
        print("=" * 90)
        print("Unified Shuffle Ablation Study: Within-Mouse and Cross-Mouse LOMO")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)

        print("\nResolved arguments:")
        print(f"  output_folder:          {state.output_folder}")
        print(f"  data_root:              {state.data_root}")
        print(f"  meta_root:              {state.meta_root}")
        print(f"  p_active:               {state.p_active}")
        print(f"  vectorization_method:   {state.vectorization_method}")
        print(f"  n_shuffles:             {state.n_shuffles}")
        print(f"  shuffle_type:           {state.shuffle_type}")
        print(f"  skip_within_mouse:      {state.skip_within_mouse}")
        print(f"  skip_cross_mouse:       {state.skip_cross_mouse}")
        print(f"  device:                 {device}")

        discovered_mice = _discover_mice(state.data_root)
        print(f"\nDiscovered mice: {len(discovered_mice)}")

        # Upfront validation
        eligible_mice = validate_mice_for_ablation(state.mice, state.data_root, state.meta_root, discovered_mice)
        print(f"Eligible mice (≥2 labels): {len(eligible_mice)}")
        if not eligible_mice:
            raise RuntimeError("No eligible mice found")
        print(f"  {eligible_mice}")

        # Initialize result containers
        all_within_results: Dict[int, Dict[str, Dict[str, object]]] = {}
        all_cross_results: Dict[int, Dict[str, Dict[str, object]]] = {}
        all_within_cms: Dict[int, Dict[str, Dict[str, object]]] = {}
        all_cross_cms: Dict[int, Dict[str, Dict[str, object]]] = {}

        # --- Determine global clip frames ONCE before the shuffle loop ---
        if state.clip_frames is not None:
            global_clip = int(state.clip_frames)
        else:
            print("\nPre-scanning valid_frames to determine global clip_frames ...")
            min_frames_list: List[int] = []
            for mouse_name in eligible_mice:
                try:
                    _, _, _, valid_frames_scan = load_labelled_barcodes(
                        state.data_root,
                        state.meta_root,
                        mouse_name,
                        state.zz_folder,
                        max_trials=state.max_trials,
                    )
                    if len(valid_frames_scan) > 0:
                        min_frames_list.append(int(valid_frames_scan.min()))
                except Exception as exc:
                    print(f"    Warning: could not pre-scan {mouse_name}: {exc}")
            if not min_frames_list:
                raise RuntimeError("Could not determine global clip_frames")
            global_clip = min(min_frames_list)
            print(f"  global_clip_frames={global_clip}")

        # --- Report existing shuffle caches ---
        print("\nChecking existing shuffle caches ...")
        for mouse_name in eligible_mice:
            mouse_cache_dir = _resolve_mouse_cache_dir(state, mouse_name)
            base_stem = build_vectorization_cache_stem(
                mouse_name=mouse_name,
                method=state.vectorization_method,
                p_active=state.p_active,
                per_trial_thresh=state.per_trial_thresh,
                clip_frames=global_clip,
            )
            missing = _find_missing_shuffle_ids(
                mouse_cache_dir, base_stem, state.shuffle_type, state.n_shuffles
            )
            n_existing = state.n_shuffles - len(missing)
            print(
                f"  {mouse_name}: {n_existing}/{state.n_shuffles} cached"
                + (f"; missing ids: {missing}" if missing else " (all present)")
            )

        # Main shuffle loop
        for shuffle_id in range(state.n_shuffles):
            print(f"\n{'='*90}")
            print(f"Shuffle iteration {shuffle_id + 1}/{state.n_shuffles}")
            print(f"{'='*90}")

            shuffle_seed = state.seed + shuffle_id * 1000
            _set_seed(shuffle_seed)

            # Load / compute shuffled vectorizations for all mice
            print(f"\nPreparing shuffle {shuffle_id} ({state.shuffle_type}) ...")
            shuffled_grids_by_mouse: Dict[str, Dict[str, Any]] = {}

            for mouse_name in eligible_mice:
                try:
                    # Load barcodes and grids
                    barcodes, labels, trial_ids, valid_frames = load_labelled_barcodes(
                        state.data_root,
                        state.meta_root,
                        mouse_name,
                        state.zz_folder,
                        max_trials=state.max_trials,
                    )
                    grid_paths, grid_labels, grid_trial_ids, grid_valid_frames = load_labelled_grid_paths(
                        state.data_root,
                        state.meta_root,
                        mouse_name,
                        grid_subdir=state.grid_subdir,
                    )

                    if len(grid_paths) == 0 or len(barcodes) == 0:
                        print(f"  {mouse_name}: skipped (no grids or barcodes)")
                        continue

                    # Align trial IDs
                    vec_idx = {tid: i for i, tid in enumerate(trial_ids)}
                    grid_idx = {int(tid): i for i, tid in enumerate(grid_trial_ids)}
                    common_trial_ids = [tid for tid in trial_ids if tid in grid_idx]

                    if len(common_trial_ids) == 0:
                        print(f"  {mouse_name}: skipped (no overlapping trials)")
                        continue

                    vec_take = np.array([vec_idx[tid] for tid in common_trial_ids], dtype=np.int64)
                    grid_take = np.array([grid_idx[tid] for tid in common_trial_ids], dtype=np.int64)

                    barcodes_common = [barcodes[i] for i in vec_take]
                    labels_common = labels[vec_take]
                    grid_paths_common = [grid_paths[i] for i in grid_take]
                    grid_frames_common = grid_valid_frames[grid_take]

                    # Build cache path for this shuffle iteration
                    base_stem = build_vectorization_cache_stem(
                        mouse_name=mouse_name,
                        method=state.vectorization_method,
                        p_active=state.p_active,
                        per_trial_thresh=state.per_trial_thresh,
                        clip_frames=global_clip,
                    )
                    mouse_cache_dir = _resolve_mouse_cache_dir(state, mouse_name)
                    mouse_cache_dir.mkdir(parents=True, exist_ok=True)
                    shuffle_stem = _shuffle_cache_stem(base_stem, state.shuffle_type, shuffle_id)
                    cache_path = mouse_cache_dir / f"{shuffle_stem}.npz"

                    # --- Load from cache if available and not forcing recompute ---
                    if cache_path.exists() and not state.force_recompute and state.skip_existing_shuffles:
                        print(f"  {mouse_name}: loading cached shuffle {shuffle_id} ...", end=" ", flush=True)
                        cached = load_vectorization_cache(cache_path)
                        x_vec = np.nan_to_num(np.asarray(
                            cached["features"] if "features" in cached else cached["X"]
                        ))
                        shuffled_grids_by_mouse[mouse_name] = {
                            "barcodes": None,  # not needed for classification
                            "labels": labels_common,
                            "trial_ids": trial_ids[vec_take],
                            "valid_frames": valid_frames[vec_take],
                            "grid_paths": grid_paths_common,
                            "x_vec": x_vec,
                            "clip_used": global_clip,
                            "cache_path": str(cache_path),
                        }
                        print(f"ok (cached, {len(x_vec)} trials)")
                        continue

                    # --- Otherwise: load grids one-by-one, shuffle, compute zigzag ---
                    print(f"  {mouse_name}: shuffling + computing zigzag ...", end=" ", flush=True)

                    # Compute threshold based on per_trial_thresh flag
                    if state.per_trial_thresh:
                        # Per-trial: will be computed inside loop for each grid
                        threshold = None  # placeholder; computed per grid
                    else:
                        # Global: compute once before loop
                        threshold = compute_threshold_from_grid_sample(
                            grid_paths_common, state.p_active, n_sample=min(5, len(grid_paths_common))
                        )

                    barcodes_shuffled: List[List[Tuple[int, float, float]]] = []
                    for gpath in grid_paths_common:
                        grid = np.load(gpath)
                        if state.shuffle_type == "time":
                            shuffled_grid = shuffle_grid_time_dimension(grid, seed=shuffle_seed)
                        elif state.shuffle_type == "spatial":
                            shuffled_grid = shuffle_grid_spatial_dimensions(grid, seed=shuffle_seed)
                        elif state.shuffle_type == "phase":
                            shuffled_grid = shuffle_grid_phase(grid, seed=shuffle_seed)
                        else:
                            raise ValueError(f"Unknown shuffle_type: {state.shuffle_type}")
                        
                        try:
                            # Recompute threshold per grid if per_trial_thresh=True
                            if state.per_trial_thresh:
                                grid_threshold = compute_threshold_from_grid_sample(
                                    [gpath], state.p_active, n_sample=1
                                )
                            else:
                                grid_threshold = threshold
                            
                            bars = compute_zigzag_from_grid(
                                shuffled_grid, threshold=grid_threshold, p_active=state.p_active,
                                max_dim=state.max_dim
                            )
                            barcodes_shuffled.append(bars)
                        except Exception as e:
                            print(f"\n    Error computing zigzag for {mouse_name}: {e}")
                            raise
                        finally:
                            del grid, shuffled_grid  # free RAM immediately

                    # Vectorize and **persist** to cache (never deleted)
                    vec_out = create_vectorization(
                        barcodes_shuffled,
                        state.vectorization_method,
                        clip_frames=global_clip,
                        output_folder=mouse_cache_dir,
                        cache_stem=shuffle_stem,
                        mouse_name=mouse_name,
                        labels=labels_common,
                        trial_ids=trial_ids[vec_take],
                        valid_frames=valid_frames[vec_take],
                    )
                    x_vec = np.nan_to_num(np.asarray(vec_out["features"]))

                    shuffled_grids_by_mouse[mouse_name] = {
                        "barcodes": barcodes_shuffled,
                        "labels": labels_common,
                        "trial_ids": trial_ids[vec_take],
                        "valid_frames": valid_frames[vec_take],
                        "grid_paths": grid_paths_common,
                        "x_vec": x_vec,
                        "clip_used": global_clip,
                        "cache_path": str(cache_path),
                    }
                    print(f"ok ({len(barcodes_shuffled)} trials, saved to cache)")
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    traceback.print_exc()

            eligible_mice_with_data = list(shuffled_grids_by_mouse.keys())
            if len(eligible_mice_with_data) < 2:
                print(f"\n  Warning: only {len(eligible_mice_with_data)} mice have shuffled data; need ≥2 for cross-mouse")
                if not state.skip_cross_mouse:
                    print("  Skipping cross-mouse analysis for this shuffle")
                    state.skip_cross_mouse = True

            # Run analyses
            if not state.skip_within_mouse:
                print(f"\nWithin-mouse CV analysis (shuffle_id={shuffle_id}):")
                within_results, within_cms = run_within_mouse_analysis(
                    state, eligible_mice_with_data, shuffled_grids_by_mouse, device, shuffle_id
                )
                all_within_results[shuffle_id] = within_results
                all_within_cms[shuffle_id] = within_cms

            if not state.skip_cross_mouse and len(eligible_mice_with_data) >= 2:
                print(f"\nCross-mouse LOMO analysis (shuffle_id={shuffle_id}):")
                cross_results, cross_cms = run_cross_mouse_analysis(
                    state, eligible_mice_with_data, shuffled_grids_by_mouse, device, shuffle_id
                )
                all_cross_results[shuffle_id] = cross_results
                all_cross_cms[shuffle_id] = cross_cms

            # Memory cleanup after each shuffle iteration
            gc.collect()

        # Generate results and figures
        print(f"\n{'='*90}")
        print("Generating aggregated results and figures")
        print(f"{'='*90}")

        result_dict = {
            "log_path": str(log_path),
            "shuffle_results": {},
        }

        # Within-mouse results
        if all_within_results:
            within_summary_json = output_folder / "within_mouse_ablation_shuffle_metrics.json"
            within_summary_csv = output_folder / "within_mouse_ablation_shuffle_metrics.csv"

            within_payload = {
                "method": state.vectorization_method,
                "p_active": state.p_active,
                "n_shuffles": state.n_shuffles,
                "shuffle_type": state.shuffle_type,
                "eligible_mice": eligible_mice_with_data,
                "per_shuffle": all_within_results,
            }
            with open(within_summary_json, "w", encoding="utf-8") as fp:
                json.dump(within_payload, fp, indent=2)
            print(f"Wrote within-mouse summary JSON: {within_summary_json}")

            with open(within_summary_csv, "w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow([
                    "shuffle_id",
                    "mouse",
                    "model",
                    "n_trials",
                    "cv_folds",
                    "mean_acc",
                    "std_acc",
                    "mean_f1",
                    "std_f1",
                    "best_model",
                ])
                for shuffle_id in sorted(all_within_results.keys()):
                    for mouse_name, row in all_within_results[shuffle_id].items():
                        for mk in ["logreg", "cnn3d"]:
                            mr = row["models"][mk]
                            writer.writerow([
                                shuffle_id,
                                mouse_name,
                                mk,
                                row["n_trials"],
                                row["cv_folds"],
                                mr.get("mean_acc", mr.get("accuracy")),
                                mr.get("std_acc", 0),
                                mr.get("mean_f1", mr.get("macro_f1")),
                                mr.get("std_f1", 0),
                                row["best_model"],
                            ])
            print(f"Wrote within-mouse summary CSV: {within_summary_csv}")
            result_dict["within_mouse_json"] = str(within_summary_json)
            result_dict["within_mouse_csv"] = str(within_summary_csv)

        # Cross-mouse results
        if all_cross_results:
            cross_summary_json = output_folder / "cross_mouse_ablation_shuffle_metrics.json"
            cross_summary_csv = output_folder / "cross_mouse_ablation_shuffle_metrics.csv"

            cross_payload = {
                "method": state.vectorization_method,
                "p_active": state.p_active,
                "n_shuffles": state.n_shuffles,
                "shuffle_type": state.shuffle_type,
                "eligible_mice": eligible_mice_with_data,
                "per_shuffle": all_cross_results,
            }
            with open(cross_summary_json, "w", encoding="utf-8") as fp:
                json.dump(cross_payload, fp, indent=2)
            print(f"Wrote cross-mouse summary JSON: {cross_summary_json}")

            with open(cross_summary_csv, "w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow([
                    "shuffle_id",
                    "test_mouse",
                    "model",
                    "n_train_trials",
                    "n_test_trials",
                    "accuracy",
                    "macro_f1",
                    "best_model",
                ])
                for shuffle_id in sorted(all_cross_results.keys()):
                    for test_mouse, row in all_cross_results[shuffle_id].items():
                        for mk in ["logreg", "cnn3d"]:
                            mr = row["models"][mk]
                            writer.writerow([
                                shuffle_id,
                                test_mouse,
                                mk,
                                row["n_train_trials"],
                                row["n_test_trials"],
                                mr["accuracy"],
                                mr["macro_f1"],
                                row["best_model"],
                            ])
            print(f"Wrote cross-mouse summary CSV: {cross_summary_csv}")
            result_dict["cross_mouse_json"] = str(cross_summary_json)
            result_dict["cross_mouse_csv"] = str(cross_summary_csv)

    return result_dict


def main() -> None:
    """Main entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.zz_folder is None:
        args.zz_folder = _build_zz_folder(args.p_active, args.per_trial_thresh)

    state = RunState(
        output_folder=args.output_folder,
        data_root=args.data_root,
        meta_root=args.meta_root,
        p_active=args.p_active,
        per_trial_thresh=args.per_trial_thresh,
        zz_folder=args.zz_folder,
        vectorization_method=args.vectorization_method,
        mice=args.mice,
        clip_frames=args.clip_frames,
        grid_subdir=args.grid_subdir,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
        n_shuffles=args.n_shuffles,
        shuffle_type=args.shuffle_type,
        skip_within_mouse=args.skip_within_mouse,
        skip_cross_mouse=args.skip_cross_mouse,
        skip_existing_shuffles=args.skip_existing_shuffles,
        max_trials=args.max_trials,
        batch_size_vec=16,  # Not used in shuffle ablation
        batch_size_grid=args.batch_size_grid,
        epochs_cnn3d=args.epochs_cnn3d,
        lr_cnn3d=args.lr_cnn3d,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        device=args.device,
        num_workers_dl=args.num_workers_dl,
        max_dim=args.max_dim,
    )

    try:
        result = run_pipeline(state)
        print("\n" + "="*90)
        print("SUCCESS!")
        print("="*90)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
