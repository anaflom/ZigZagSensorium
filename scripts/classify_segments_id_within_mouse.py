#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Within-mouse, within-label segment-ID decoding from zigzag + grid activity.

This script builds segment-level samples using video metadata JSON files:
  <meta_root>/global_meta/videos/<video_id>.json

Only valid trials are used (valid_response & valid_trial), and only labels:
  NaturalImages, PinkNoise, RandomDots, Gabor, GaussianDot

For each mouse and label independently:
  1) Build segment samples using segment start from metadata and fixed
     label-specific segment lengths.
  2) Turnover vector branch: extract H0/H1/H2 sub-profiles over the segment
     window and concatenate (dimension = 3 * seg_length).
  3) Grid branch: slice (C, T, H, W) segment windows and classify with 3D-CNN.
  4) Run Leave-One-Segment-Out CV (one held-out segment sample per fold).

Models:
    - LogReg, MLP, 1D-CNN (vector branch)
    - 3D-CNN raw grid, 3D-CNN normalized grid (grid branch)

Outputs:
  - summary JSON + CSV
  - confusion matrices JSON
  - predictions JSON
  - figures with macro-F1/accuracy and chance lines
  - run log
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

# Force a non-interactive backend for cluster/headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset

from classification_models import (
    CNN1D,
    CNN3D,
    MLP,
    SegmentGridDataset,
    VectorDataset,
    infer_cnn1d_shape,
    train_eval_logreg,
    train_eval_nn,
)
from utils import (
    _build_zz_folder,
    _discover_mice,
    _opt_csv_list,
    _opt_int,
    _resolve_mouse_cache_dir,
    _short_mouse_name,
    _str2bool,
    build_segment_sample_records,
    load_labelled_barcodes,
    load_labelled_grid_paths,
    load_or_compute_vectorization_features,
)


TARGET_LABELS: Tuple[str, ...] = (
    "NaturalImages",
    "PinkNoise",
    "RandomDots",
    "Gabor",
    "GaussianDot",
)

DEFAULT_SEGMENT_LENGTHS: Dict[str, int] = {
    "NaturalImages": 12,
    "PinkNoise": 27,
    "RandomDots": 60,
    "Gabor": 25,
    "GaussianDot": 9,
}

MODEL_CHOICES = ["logreg", "mlp", "cnn1d", "cnn3d_raw", "cnn3d_norm"]
DEFAULT_MODELS = ["logreg", "mlp", "cnn1d", "cnn3d_raw", "cnn3d_norm"]
MODEL_TITLES = {
    "logreg": "LogReg (vector)",
    "mlp": "MLP (vector)",
    "cnn1d": "1D-CNN (vector)",
    "cnn3d_raw": "3D-CNN (raw grid)",
    "cnn3d_norm": "3D-CNN (normalized grid)",
}
MODEL_COLORS = {
    "logreg": "#4C72B0",
    "mlp": "#55A868",
    "cnn1d": "#8172B2",
    "cnn3d_raw": "#DD8452",
    "cnn3d_norm": "#937860",
}


def _resolve_models(models: Optional[List[str]]) -> List[str]:
    chosen = DEFAULT_MODELS if models is None else [m.strip() for m in models if str(m).strip()]
    if not chosen:
        raise ValueError("--models resolved to an empty list")
    unknown = sorted({m for m in chosen if m not in MODEL_CHOICES})
    if unknown:
        raise ValueError(f"Unknown model(s) in --models: {unknown}. Allowed: {MODEL_CHOICES}")
    deduped: List[str] = []
    for mk in chosen:
        if mk not in deduped:
            deduped.append(mk)
    return deduped


def _grid_mode_label(model_order: Sequence[str]) -> str:
    parts: List[str] = []
    if "cnn3d_raw" in model_order:
        parts.append("raw-grid")
    if "cnn3d_norm" in model_order:
        parts.append("segment-l1-normalized-grid")
    return "+".join(parts) if parts else "no-grid-model"


@dataclass
class RunState:
    output_folder: Path
    data_root: Path
    meta_root: Path
    p_active: int
    per_trial_thresh: bool
    zz_folder: str
    mice: Optional[List[str]]
    models: List[str]
    clip_frames: Optional[int]
    grid_subdir: str
    cache_dir: Optional[Path]
    max_trials: Optional[int]
    batch_size_vec: int
    epochs_cnn3d: int
    epochs_mlp: int
    epochs_cnn1d: int
    lr_vec: float
    lr_cnn3d: float
    weight_decay: float
    early_stop_patience: int
    batch_size_grid: int
    seed: int
    device: str
    num_workers_dl: int
    segment_lengths: Dict[str, int]
    cv_scheme_logreg: str
    cv_scheme_cnn3d: str
    cv_n_splits_logreg: int
    cv_n_splits_cnn3d: int


class _TeeStream:
    """Mirror stream writes to multiple streams and flush immediately."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
                s.flush()
            except Exception:
                continue
        return n

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                continue


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


def _safe_label_token(label: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(label))
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _build_loso_splits(n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    idx_all = np.arange(n_samples, dtype=np.int64)
    for i in range(n_samples):
        val_idx = np.array([i], dtype=np.int64)
        train_idx = idx_all[idx_all != i]
        splits.append((train_idx, val_idx))
    return splits


def _build_cv_splits(
    n_samples: int,
    *,
    scheme: str,
    n_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build CV splits with selectable schemes.

    Supported schemes:
      - loso: Leave-One-Sample-Out
      - logo: Leave-One-Group-Out (requires groups)
      - groupkfold: GroupKFold (requires groups)
      - kfold: KFold (sample-level)
    """
    scheme_norm = str(scheme).strip().lower()
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for CV, got {n_samples}")

    if scheme_norm == "loso":
        return _build_loso_splits(n_samples)

    if scheme_norm == "logo":
        if groups is None:
            raise ValueError("cv scheme 'logo' requires trial groups")
        grp = np.asarray(groups)
        n_groups = int(np.unique(grp).shape[0])
        if n_groups < 2:
            raise ValueError(f"cv scheme 'logo' needs >=2 groups, got {n_groups}")
        logo = LeaveOneGroupOut()
        return [(tr.astype(np.int64), va.astype(np.int64)) for tr, va in logo.split(np.arange(n_samples), groups=grp)]

    if scheme_norm == "groupkfold":
        if groups is None:
            raise ValueError("cv scheme 'groupkfold' requires trial groups")
        grp = np.asarray(groups)
        n_groups = int(np.unique(grp).shape[0])
        actual_splits = int(min(max(2, int(n_splits)), n_groups))
        if actual_splits < 2:
            raise ValueError(f"cv scheme 'groupkfold' needs >=2 groups, got {n_groups}")
        gkf = GroupKFold(n_splits=actual_splits)
        return [
            (tr.astype(np.int64), va.astype(np.int64))
            for tr, va in gkf.split(np.arange(n_samples), groups=grp)
        ]

    if scheme_norm == "kfold":
        actual_splits = int(min(max(2, int(n_splits)), n_samples))
        if actual_splits < 2:
            raise ValueError(f"cv scheme 'kfold' needs >=2 samples, got {n_samples}")
        kf = KFold(n_splits=actual_splits, shuffle=True, random_state=int(seed))
        return [(tr.astype(np.int64), va.astype(np.int64)) for tr, va in kf.split(np.arange(n_samples))]

    raise ValueError(f"Unsupported cv scheme: {scheme}. Use one of loso|logo|groupkfold|kfold")


def _extract_turnover_segment_vector(
    full_feature: np.ndarray,
    *,
    clip_frames: int,
    start_frame: int,
    seg_length: int,
) -> np.ndarray:
    """Extract [H0, H1, H2] segment sub-vector from full Turnover feature row."""
    feat = np.asarray(full_feature, dtype=np.float64)
    if clip_frames <= 0:
        raise ValueError(f"Invalid clip_frames={clip_frames}")
    if feat.ndim != 1:
        raise ValueError(f"Expected 1D feature row, got shape {feat.shape}")
    if feat.shape[0] % clip_frames != 0:
        raise ValueError(
            f"Feature length {feat.shape[0]} not divisible by clip_frames={clip_frames}"
        )

    n_dims = int(feat.shape[0] // clip_frames)
    if n_dims < 3:
        raise ValueError(f"Need at least 3 homology dimensions, found {n_dims}")

    if start_frame < 0 or seg_length <= 0:
        raise ValueError(f"Invalid segment window start={start_frame}, len={seg_length}")
    end = int(start_frame + seg_length)
    if end > clip_frames:
        raise ValueError(
            f"Segment end {end} exceeds clip_frames={clip_frames} for vector extraction"
        )

    blocks: List[np.ndarray] = []
    for dim in (0, 1, 2):
        base = dim * clip_frames
        blocks.append(feat[base + start_frame : base + end])

    out = np.concatenate(blocks, axis=0)
    expected = 3 * int(seg_length)
    if int(out.shape[0]) != expected:
        raise ValueError(f"Unexpected segment vector length {out.shape[0]} (expected {expected})")
    return np.asarray(out, dtype=np.float64)


def _constant_prediction(y_train: np.ndarray, n: int) -> np.ndarray:
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) == 0:
        return np.zeros((n,), dtype=np.int64)
    return np.full((n,), fill_value=int(labels[np.argmax(counts)]), dtype=np.int64)


def run_logreg_loso(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    log_fn: Optional[Any] = None,
    progress_every: int = 25,
) -> np.ndarray:
    """LOSO CV for LogReg returning out-of-fold predictions."""
    oof = np.full_like(y, fill_value=-1, dtype=np.int64)
    n_folds = int(len(splits))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if log_fn is not None and (
            fold_idx == 0
            or (fold_idx + 1) % int(max(1, progress_every)) == 0
            or (fold_idx + 1) == n_folds
        ):
            log_fn(f"    [LogReg] fold {fold_idx + 1}/{n_folds}")
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_train)) < 2:
            pred = _constant_prediction(y_train, len(val_idx))
        else:
            _metrics, pred = train_eval_logreg(x_train, y_train, x_val, y_val)
        oof[val_idx] = pred
    return oof


def run_nn_loso(
    dataset: Any,
    make_model: Any,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
    num_workers: int,
    seed: int,
    model_tag: str,
    log_fn: Optional[Any] = None,
    progress_every_fold: int = 1,
    progress_every_epoch: int = 5,
) -> np.ndarray:
    """Cross-validation for neural models returning out-of-fold predictions."""
    oof = np.full_like(y, fill_value=-1, dtype=np.int64)

    n_folds = int(len(splits))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        y_train = y[train_idx]
        should_log_fold = (
            log_fn is not None
            and (
                fold_idx == 0
                or (fold_idx + 1) % int(max(1, progress_every_fold)) == 0
                or (fold_idx + 1) == n_folds
            )
        )
        if should_log_fold:
            log_fn(
                f"    [{model_tag}] fold {fold_idx + 1}/{n_folds} "
                f"train_n={len(train_idx)} val_n={len(val_idx)}"
            )

        if len(np.unique(y_train)) < 2:
            pred = _constant_prediction(y_train, len(val_idx))
            oof[val_idx] = pred
            if should_log_fold:
                log_fn(f"    [{model_tag}] fold {fold_idx + 1}/{n_folds} skipped training (single class)")
            continue

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        val_y = y[val_idx]

        # Offset seed by fold for deterministic but non-identical fold runs.
        fold_seed = int(seed + fold_idx)
        fold_metrics, pred = train_eval_nn(
            make_model=make_model,
            train_ds=train_ds,
            y_train=y_train,
            test_ds=val_ds,
            test_y=val_y,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            weight_decay=weight_decay,
            device=device,
            num_workers=num_workers,
            seed=fold_seed,
            progress_prefix=f"    [{model_tag}] fold {fold_idx + 1}/{n_folds} ",
            progress_every=progress_every_epoch,
            log_fn=log_fn,
        )
        oof[val_idx] = pred
        if should_log_fold:
            log_fn(
                f"    [{model_tag}] fold {fold_idx + 1}/{n_folds} "
                f"done f1={fold_metrics['macro_f1']:.3f} acc={fold_metrics['accuracy']:.3f}"
            )

    return oof


def _build_mouse_figures(
    *,
    mouse_name: str,
    per_label_results: Dict[str, Dict[str, Any]],
    model_order: Sequence[str],
    model_titles: Dict[str, str],
    model_colors: Dict[str, str],
    figures_dir: Path,
) -> Optional[Path]:
    labels = [lab for lab in TARGET_LABELS if lab in per_label_results]
    if len(labels) == 0:
        return None
    if len(model_order) == 0:
        return None

    x = np.arange(len(labels))
    width = min(0.82 / len(model_order), 0.22)

    chance = [float(per_label_results[l]["chance_level"]) for l in labels]

    fig, (ax_acc, ax_f1) = plt.subplots(2, 1, figsize=(max(10, len(labels) * 2.0), 8.2), sharex=True)

    for idx, mk in enumerate(model_order):
        offset = (idx - (len(model_order) - 1) / 2.0) * width
        acc_vals = [float(per_label_results[l][mk]["accuracy"]) for l in labels]
        f1_vals = [float(per_label_results[l][mk]["macro_f1"]) for l in labels]
        ax_acc.bar(
            x + offset,
            acc_vals,
            width=width,
            color=model_colors.get(mk, "#4C72B0"),
            alpha=0.88,
            label=model_titles.get(mk, mk),
        )
        ax_f1.bar(
            x + offset,
            f1_vals,
            width=width,
            color=model_colors.get(mk, "#4C72B0"),
            alpha=0.88,
            label=model_titles.get(mk, mk),
        )

    ax_acc.plot(x, chance, "k--", linewidth=1.3, marker="o", markersize=4, label="Chance")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{_short_mouse_name(mouse_name)} - Segment-ID decoding (accuracy)")
    ax_acc.grid(axis="y", alpha=0.25)
    ax_acc.legend(loc="upper right", fontsize=8)

    ax_f1.plot(x, chance, "k--", linewidth=1.3, marker="o", markersize=4, label="Chance")
    ax_f1.set_ylim(0.0, 1.05)
    ax_f1.set_ylabel("Macro-F1")
    ax_f1.set_title(f"{_short_mouse_name(mouse_name)} - Segment-ID decoding (macro-F1)")
    ax_f1.grid(axis="y", alpha=0.25)

    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(labels, rotation=25, ha="right")

    fig.tight_layout()
    out_path = figures_dir / f"mouse_{_safe_label_token(mouse_name)}_segment_id_scores.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_pipeline(state: RunState) -> Dict[str, Any]:
    _set_seed(state.seed)
    device = _resolve_device(state.device)

    output_folder = state.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    figures_dir = output_folder / "figures"
    logs_dir = output_folder / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "run.log"

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_fp:
        tee_out = _TeeStream(sys.stdout, log_fp)
        tee_err = _TeeStream(sys.stderr, log_fp)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = tee_out
        sys.stderr = tee_err
        print("=" * 90)
        print("Within-Mouse Segment-ID Decoding: selectable vector/grid models")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)

        print("\nResolved arguments:")
        print(f"  output_folder:     {state.output_folder}")
        print(f"  data_root:         {state.data_root}")
        print(f"  meta_root:         {state.meta_root}")
        print(f"  p_active:          {state.p_active}")
        print(f"  per_trial_thresh:  {state.per_trial_thresh}")
        print(f"  zz_folder:         {state.zz_folder}")
        print(f"  mice:              {state.mice}")
        print(f"  models:            {state.models}")
        print(f"  clip_frames:       {state.clip_frames}")
        print(f"  grid_subdir:       {state.grid_subdir}")
        print(f"  max_trials:        {state.max_trials}")
        print(f"  seed:              {state.seed}")
        print(f"  device:            {device}")
        print(f"  batch_size_vec:    {state.batch_size_vec}")
        print(f"  epochs_mlp:        {state.epochs_mlp}")
        print(f"  epochs_cnn1d:      {state.epochs_cnn1d}")
        print(f"  lr_vec:            {state.lr_vec}")
        print(f"  cv_scheme_logreg:  {state.cv_scheme_logreg}")
        print(f"  cv_scheme_cnn3d:   {state.cv_scheme_cnn3d}")
        print(f"  cv_n_splits_logreg: {state.cv_n_splits_logreg}")
        print(f"  cv_n_splits_cnn3d: {state.cv_n_splits_cnn3d}")
        print("  target_labels:")
        for lbl in TARGET_LABELS:
            print(f"    - {lbl}: seg_length={state.segment_lengths[lbl]}")

        discovered_mice = _discover_mice(state.data_root)
        selected_mice = state.mice if state.mice is not None else discovered_mice
        selected_mice = [m for m in selected_mice if m in discovered_mice]

        if not selected_mice:
            raise RuntimeError("No valid mice selected for segment decoding.")

        print(f"\nDiscovered mice: {len(discovered_mice)}")
        print(f"Selected mice: {len(selected_mice)}")

        model_order = list(state.models)
        model_titles = {mk: MODEL_TITLES[mk] for mk in model_order}
        model_colors = {mk: MODEL_COLORS[mk] for mk in model_order}
        grid_mode = _grid_mode_label(model_order)

        summary_rows: List[Dict[str, Any]] = []
        skip_rows: List[Dict[str, Any]] = []
        missing_vectorization: Dict[str, str] = {}

        summary_payload: Dict[str, Any] = {
            "method": "Turnover",
            "models": model_order,
            "grid_mode": grid_mode,
            "cv": {
                "vector": {
                    "scheme": state.cv_scheme_logreg,
                    "n_splits": int(state.cv_n_splits_logreg),
                },
                "grid": {
                    "scheme": state.cv_scheme_cnn3d,
                    "n_splits": int(state.cv_n_splits_cnn3d),
                },
            },
            "target_labels": list(TARGET_LABELS),
            "segment_lengths": dict(state.segment_lengths),
            "mice": [],
            "results": {},
            "figures": [],
            "log_path": str(log_path),
            "device": str(device),
        }
        confusion_payload: Dict[str, Any] = {}
        prediction_payload: Dict[str, Any] = {}

        n_mice_total = len(selected_mice)
        for mouse_idx, mouse_name in enumerate(selected_mice, start=1):
            print(f"\n## Mouse [{mouse_idx}/{n_mice_total}]: {mouse_name}")

            try:
                # Build metadata-derived segment records first (controls required temporal support).
                seg_df, seg_counters = build_segment_sample_records(
                    meta_root=state.meta_root,
                    mouse_name=mouse_name,
                    target_labels=TARGET_LABELS,
                    segment_length_by_label=state.segment_lengths,
                    videos_subdir="global_meta/videos",
                )
                print(f"  Segment metadata records: {len(seg_df)}")
                print(f"  Segment metadata counters: {seg_counters}")

                if len(seg_df) == 0:
                    skip_rows.append(
                        {
                            "mouse": mouse_name,
                            "label": "ALL",
                            "reason": "no_segment_records",
                            "detail": json.dumps(seg_counters),
                        }
                    )
                    continue

                barcodes, vec_labels, vec_trial_ids, vec_valid_frames = load_labelled_barcodes(
                    state.data_root,
                    state.meta_root,
                    mouse_name,
                    state.zz_folder,
                    max_trials=state.max_trials,
                )
                if len(barcodes) == 0:
                    skip_rows.append(
                        {"mouse": mouse_name, "label": "ALL", "reason": "no_labelled_barcodes", "detail": ""}
                    )
                    print("  No labelled barcodes found; skipping mouse.")
                    continue

                # Cache lookup uses requested clip mode (None => full-trial cache key).
                cache_clip = state.clip_frames

                vec_trial_ids_int = [int(t) for t in vec_trial_ids]
                xmat, vec_source, cache_path = load_or_compute_vectorization_features(
                    data_root=state.data_root,
                    mouse_name=mouse_name,
                    method="Turnover",
                    p_active=state.p_active,
                    per_trial_thresh=state.per_trial_thresh,
                    clip_frames=cache_clip,
                    barcodes=barcodes,
                    labels=vec_labels,
                    trial_ids=vec_trial_ids,
                    valid_frames=vec_valid_frames,
                    cache_dir=_resolve_mouse_cache_dir(state, mouse_name),
                    expected_trial_ids=vec_trial_ids_int,
                    message_prefix="  ",
                )

                # Turnover feature rows are [H0 | H1 | H2], each block length = clip_used.
                if xmat.ndim != 2 or xmat.shape[1] % 3 != 0:
                    raise RuntimeError(
                        f"Unexpected Turnover feature matrix shape for {mouse_name}: {xmat.shape}"
                    )
                clip_used = int(xmat.shape[1] // 3)
                print(f"  Turnover vectorization source={vec_source}, cache={cache_path}")
                print(f"  Turnover matrix shape={xmat.shape}, clip_used={clip_used}")

                grid_paths, grid_labels, grid_trial_ids, grid_valid_frames = load_labelled_grid_paths(
                    state.data_root,
                    state.meta_root,
                    mouse_name,
                    grid_subdir=state.grid_subdir,
                )
                if len(grid_paths) == 0:
                    skip_rows.append(
                        {
                            "mouse": mouse_name,
                            "label": "ALL",
                            "reason": "no_grid_paths",
                            "detail": state.grid_subdir,
                        }
                    )
                    print(f"  No grid files found in {state.grid_subdir}; skipping mouse.")
                    continue

                vec_idx = {int(tid): i for i, tid in enumerate(vec_trial_ids_int)}
                grid_idx = {int(tid): i for i, tid in enumerate(grid_trial_ids)}

                # Build aligned segment samples that are valid in both vector and grid branches.
                sample_records: List[Dict[str, Any]] = []
                align_counters = defaultdict(int)

                for row in seg_df.itertuples(index=False):
                    tid = int(row.trial_id)
                    label = str(row.label)
                    start = int(row.start_frame)
                    seg_len = int(row.seg_length)

                    if tid not in vec_idx:
                        align_counters["missing_vector_trial"] += 1
                        continue
                    if tid not in grid_idx:
                        align_counters["missing_grid_trial"] += 1
                        continue

                    v_i = int(vec_idx[tid])
                    g_i = int(grid_idx[tid])

                    if str(vec_labels[v_i]) != label:
                        align_counters["vector_label_mismatch"] += 1
                        continue
                    if str(grid_labels[g_i]) != label:
                        align_counters["grid_label_mismatch"] += 1
                        continue

                    try:
                        seg_vec = _extract_turnover_segment_vector(
                            full_feature=np.asarray(xmat[v_i]),
                            clip_frames=int(clip_used),
                            start_frame=start,
                            seg_length=seg_len,
                        )
                    except Exception:
                        align_counters["vector_slice_error"] += 1
                        continue

                    sample_records.append(
                        {
                            "mouse": mouse_name,
                            "label": label,
                            "trial_id": tid,
                            "segment_id": str(row.segment_id),
                            "segment_index": int(row.segment_index),
                            "start_frame": start,
                            "seg_length": seg_len,
                            "video_id": str(row.video_id),
                            "vector": seg_vec,
                            "grid_path": grid_paths[g_i],
                            "grid_valid_frames": int(grid_valid_frames[g_i]),
                        }
                    )

                print(f"  Aligned segment samples: {len(sample_records)}")
                print(f"  Alignment counters: {dict(align_counters)}")

                if len(sample_records) == 0:
                    skip_rows.append(
                        {
                            "mouse": mouse_name,
                            "label": "ALL",
                            "reason": "no_aligned_segment_samples",
                            "detail": json.dumps(dict(align_counters)),
                        }
                    )
                    continue

                per_label_results: Dict[str, Dict[str, Any]] = {}
                per_label_confusions: Dict[str, Any] = {}
                per_label_predictions: Dict[str, Any] = {}

                n_labels_total = len(TARGET_LABELS)
                for label_idx, label in enumerate(TARGET_LABELS, start=1):
                    print(f"  -> Label [{label_idx}/{n_labels_total}]: {label}")
                    group = [r for r in sample_records if r["label"] == label]
                    if len(group) == 0:
                        skip_rows.append(
                            {"mouse": mouse_name, "label": label, "reason": "no_samples_for_label", "detail": ""}
                        )
                        continue

                    seg_len_set = sorted({int(r["seg_length"]) for r in group})
                    if len(seg_len_set) != 1:
                        skip_rows.append(
                            {
                                "mouse": mouse_name,
                                "label": label,
                                "reason": "inconsistent_segment_lengths",
                                "detail": str(seg_len_set),
                            }
                        )
                        continue

                    n_samples = len(group)
                    seg_len = int(seg_len_set[0])
                    y_segment_ids = np.asarray([str(r["segment_id"]) for r in group])
                    le = LabelEncoder().fit(y_segment_ids)
                    y_int = le.transform(y_segment_ids)
                    class_labels = [str(c) for c in le.classes_]
                    n_classes = int(len(class_labels))
                    chance_level = 1.0 / float(n_classes) if n_classes > 0 else 0.0

                    if n_samples < 2:
                        skip_rows.append(
                            {
                                "mouse": mouse_name,
                                "label": label,
                                "reason": "insufficient_samples",
                                "detail": str(n_samples),
                            }
                        )
                        continue
                    if n_classes < 2:
                        skip_rows.append(
                            {
                                "mouse": mouse_name,
                                "label": label,
                                "reason": "insufficient_classes",
                                "detail": str(n_classes),
                            }
                        )
                        continue

                    trial_groups = np.asarray([int(r["trial_id"]) for r in group], dtype=np.int64)
                    try:
                        splits_vector = _build_cv_splits(
                            n_samples,
                            scheme=state.cv_scheme_logreg,
                            n_splits=state.cv_n_splits_logreg,
                            seed=state.seed,
                            groups=trial_groups,
                        )
                        splits_grid = _build_cv_splits(
                            n_samples,
                            scheme=state.cv_scheme_cnn3d,
                            n_splits=state.cv_n_splits_cnn3d,
                            seed=state.seed,
                            groups=trial_groups,
                        )
                    except Exception as exc:
                        skip_rows.append(
                            {
                                "mouse": mouse_name,
                                "label": label,
                                "reason": "cv_split_error",
                                "detail": str(exc),
                            }
                        )
                        print(f"    Skipping label due to CV split error: {exc}")
                        continue

                    x_vec = np.asarray([r["vector"] for r in group], dtype=np.float64)
                    vec_ds = VectorDataset(x_vec, y_int)

                    need_raw = "cnn3d_raw" in model_order
                    need_norm = "cnn3d_norm" in model_order
                    grid_ds_raw = None
                    grid_ds_norm = None
                    if need_raw or need_norm:
                        grid_kwargs = {
                            "grid_paths": [r["grid_path"] for r in group],
                            "y": y_int,
                            "valid_frames": np.asarray([r["grid_valid_frames"] for r in group], dtype=np.int64),
                            "start_frames": np.asarray([r["start_frame"] for r in group], dtype=np.int64),
                            "seg_lengths": np.asarray([r["seg_length"] for r in group], dtype=np.int64),
                        }
                        if need_raw:
                            grid_ds_raw = SegmentGridDataset(normalize_by_trial=False, **grid_kwargs)
                        if need_norm:
                            grid_ds_norm = SegmentGridDataset(normalize_by_trial=True, **grid_kwargs)

                    model_preds: Dict[str, np.ndarray] = {}
                    model_metrics: Dict[str, Dict[str, float]] = {}

                    for mk in model_order:
                        if mk == "logreg":
                            y_pred = run_logreg_loso(
                                x_vec,
                                y_int,
                                splits_vector,
                                log_fn=print,
                                progress_every=25,
                            )
                        elif mk == "mlp":
                            y_pred = run_nn_loso(
                                dataset=vec_ds,
                                make_model=lambda n_classes=n_classes, dim=int(x_vec.shape[1]): MLP(
                                    n_classes=n_classes,
                                    input_dim=dim,
                                ),
                                y=y_int,
                                splits=splits_vector,
                                epochs=state.epochs_mlp,
                                lr=state.lr_vec,
                                batch_size=state.batch_size_vec,
                                patience=state.early_stop_patience,
                                weight_decay=state.weight_decay,
                                device=device,
                                num_workers=state.num_workers_dl,
                                seed=state.seed,
                                model_tag="MLP",
                                log_fn=print,
                                progress_every_fold=1,
                                progress_every_epoch=5,
                            )
                        elif mk == "cnn1d":
                            in_channels, seq_len = infer_cnn1d_shape(int(x_vec.shape[1]), int(seg_len))
                            y_pred = run_nn_loso(
                                dataset=vec_ds,
                                make_model=lambda n_classes=n_classes, ic=in_channels, sl=seq_len: CNN1D(
                                    n_classes=n_classes,
                                    in_channels=ic,
                                    seq_len=sl,
                                ),
                                y=y_int,
                                splits=splits_vector,
                                epochs=state.epochs_cnn1d,
                                lr=state.lr_vec,
                                batch_size=state.batch_size_vec,
                                patience=state.early_stop_patience,
                                weight_decay=state.weight_decay,
                                device=device,
                                num_workers=state.num_workers_dl,
                                seed=state.seed,
                                model_tag="1D-CNN",
                                log_fn=print,
                                progress_every_fold=1,
                                progress_every_epoch=5,
                            )
                        elif mk == "cnn3d_raw":
                            if grid_ds_raw is None:
                                raise RuntimeError("cnn3d_raw selected but raw grid dataset is unavailable")
                            y_pred = run_nn_loso(
                                dataset=grid_ds_raw,
                                make_model=lambda n_classes=n_classes, in_ch=int(grid_ds_raw.in_channels): CNN3D(
                                    n_classes=n_classes,
                                    in_channels=in_ch,
                                ),
                                y=y_int,
                                splits=splits_grid,
                                epochs=state.epochs_cnn3d,
                                lr=state.lr_cnn3d,
                                batch_size=state.batch_size_grid,
                                patience=state.early_stop_patience,
                                weight_decay=state.weight_decay,
                                device=device,
                                num_workers=state.num_workers_dl,
                                seed=state.seed,
                                model_tag="3D-CNN raw",
                                log_fn=print,
                                progress_every_fold=1,
                                progress_every_epoch=5,
                            )
                        elif mk == "cnn3d_norm":
                            if grid_ds_norm is None:
                                raise RuntimeError("cnn3d_norm selected but normalized grid dataset is unavailable")
                            y_pred = run_nn_loso(
                                dataset=grid_ds_norm,
                                make_model=lambda n_classes=n_classes, in_ch=int(grid_ds_norm.in_channels): CNN3D(
                                    n_classes=n_classes,
                                    in_channels=in_ch,
                                ),
                                y=y_int,
                                splits=splits_grid,
                                epochs=state.epochs_cnn3d,
                                lr=state.lr_cnn3d,
                                batch_size=state.batch_size_grid,
                                patience=state.early_stop_patience,
                                weight_decay=state.weight_decay,
                                device=device,
                                num_workers=state.num_workers_dl,
                                seed=state.seed,
                                model_tag="3D-CNN norm",
                                log_fn=print,
                                progress_every_fold=1,
                                progress_every_epoch=5,
                            )
                        else:
                            raise ValueError(f"Unsupported model key: {mk}")

                        model_preds[mk] = y_pred
                        model_metrics[mk] = {
                            "accuracy": float(accuracy_score(y_int, y_pred)),
                            "macro_f1": float(f1_score(y_int, y_pred, average="macro", zero_division=0)),
                        }

                    cms_counts = {
                        mk: confusion_matrix(y_int, model_preds[mk], labels=np.arange(n_classes))
                        for mk in model_order
                    }
                    cms_norm = {
                        mk: np.divide(
                            cms_counts[mk],
                            cms_counts[mk].sum(axis=1, keepdims=True),
                            out=np.zeros_like(cms_counts[mk], dtype=float),
                            where=cms_counts[mk].sum(axis=1, keepdims=True) != 0,
                        )
                        for mk in model_order
                    }

                    class_counts = Counter([str(v) for v in y_segment_ids.tolist()])
                    n_trials = int(len({int(r["trial_id"]) for r in group}))

                    per_label_results[label] = {
                        "n_samples": int(n_samples),
                        "n_trials": int(n_trials),
                        "n_classes": int(n_classes),
                        "class_labels": class_labels,
                        "class_counts": {k: int(v) for k, v in sorted(class_counts.items())},
                        "seg_length": int(seg_len),
                        "chance_level": float(chance_level),
                        "cv": {
                            "vector": {
                                "scheme": state.cv_scheme_logreg,
                                "n_folds": int(len(splits_vector)),
                            },
                            "grid": {
                                "scheme": state.cv_scheme_cnn3d,
                                "n_folds": int(len(splits_grid)),
                            },
                        },
                        **model_metrics,
                    }

                    per_label_confusions[label] = {
                        "labels": class_labels,
                        "cms_counts": {mk: cms_counts[mk].tolist() for mk in model_order},
                        "cms_normalized": {mk: cms_norm[mk].tolist() for mk in model_order},
                    }

                    per_label_predictions[label] = {
                        "labels": class_labels,
                        "y_true": [int(v) for v in y_int.tolist()],
                        "predictions": {
                            mk: [int(v) for v in np.asarray(model_preds[mk]).tolist()] for mk in model_order
                        },
                        "samples": [
                            {
                                "trial_id": int(r["trial_id"]),
                                "segment_id": str(r["segment_id"]),
                                "segment_index": int(r["segment_index"]),
                                "start_frame": int(r["start_frame"]),
                                "seg_length": int(r["seg_length"]),
                                "video_id": str(r["video_id"]),
                            }
                            for r in group
                        ],
                    }

                    for model_name in model_order:
                        mm = model_metrics[model_name]
                        summary_rows.append(
                            {
                                "mouse": mouse_name,
                                "label": label,
                                "model": model_name,
                                "input": "grid" if model_name.startswith("cnn3d") else "vector",
                                "method": "Turnover",
                                "source": vec_source,
                                "cache_path": str(cache_path),
                                "clip_frames": int(clip_used),
                                "seg_length": int(seg_len),
                                "n_trials": int(n_trials),
                                "n_segments": int(n_samples),
                                "n_classes": int(n_classes),
                                "cv_scheme": state.cv_scheme_cnn3d if model_name.startswith("cnn3d") else state.cv_scheme_logreg,
                                "cv_folds": int(len(splits_grid)) if model_name.startswith("cnn3d") else int(len(splits_vector)),
                                "chance_level": float(chance_level),
                                "accuracy": float(mm["accuracy"]),
                                "macro_f1": float(mm["macro_f1"]),
                            }
                        )

                    model_msg = ", ".join(
                        [
                            f"{mk}(f1={model_metrics[mk]['macro_f1']:.3f}, acc={model_metrics[mk]['accuracy']:.3f})"
                            for mk in model_order
                        ]
                    )
                    print(
                        f"  {label}: samples={n_samples}, classes={n_classes}, seg_len={seg_len}, "
                        f"chance={chance_level:.3f}, "
                        f"cv_vector={state.cv_scheme_logreg}({len(splits_vector)} folds), "
                        f"cv_grid={state.cv_scheme_cnn3d}({len(splits_grid)} folds), "
                        f"{model_msg}"
                    )

                if len(per_label_results) == 0:
                    skip_rows.append(
                        {
                            "mouse": mouse_name,
                            "label": "ALL",
                            "reason": "no_label_results_after_filters",
                            "detail": "",
                        }
                    )
                    continue

                fig_path = _build_mouse_figures(
                    mouse_name=mouse_name,
                    per_label_results=per_label_results,
                    model_order=model_order,
                    model_titles=model_titles,
                    model_colors=model_colors,
                    figures_dir=figures_dir,
                )
                if fig_path is not None:
                    print(f"  Saved figure: {fig_path}")
                    summary_payload["figures"].append(str(fig_path))

                summary_payload["mice"].append(mouse_name)
                summary_payload["results"][mouse_name] = {
                    "source": vec_source,
                    "cache_path": str(cache_path),
                    "clip_frames": int(clip_used),
                    "segment_metadata_counters": seg_counters,
                    "alignment_counters": dict(align_counters),
                    "labels": per_label_results,
                }
                confusion_payload[mouse_name] = per_label_confusions
                prediction_payload[mouse_name] = per_label_predictions

            except Exception as exc:
                print(f"  FAILED: {exc}")
                traceback.print_exc()
                err_text = str(exc)
                if "vectorization cache" in err_text.lower() or "Cache mismatch" in err_text:
                    missing_vectorization[mouse_name] = err_text
                skip_rows.append(
                    {
                        "mouse": mouse_name,
                        "label": "ALL",
                        "reason": "exception",
                        "detail": str(exc),
                    }
                )

        if missing_vectorization:
            mouse_list = ", ".join(sorted(missing_vectorization.keys()))
            details = "\n".join(
                [f"- {m}: {missing_vectorization[m]}" for m in sorted(missing_vectorization.keys())]
            )
            raise RuntimeError(
                "Missing required precomputed vectorizations for mice: "
                f"{mouse_list}.\n"
                "Precompute them with scripts/generate_vectorization_cache.py before running segment decoding.\n"
                f"Details:\n{details}"
            )

        if len(summary_rows) == 0:
            raise RuntimeError("No segment-decoding results were produced.")

        # Export tables and JSON payloads.
        summary_csv_path = output_folder / "within_mouse_segment_id_metrics.csv"
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "mouse",
                    "label",
                    "model",
                    "input",
                    "method",
                    "source",
                    "cache_path",
                    "clip_frames",
                    "seg_length",
                    "n_trials",
                    "n_segments",
                    "n_classes",
                    "cv_scheme",
                    "cv_folds",
                    "chance_level",
                    "accuracy",
                    "macro_f1",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"Wrote summary CSV: {summary_csv_path}")

        skip_csv_path = output_folder / "within_mouse_segment_id_skips.csv"
        with open(skip_csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["mouse", "label", "reason", "detail"])
            writer.writeheader()
            for row in skip_rows:
                writer.writerow(row)
        print(f"Wrote skip CSV: {skip_csv_path}")

        confusion_json_path = output_folder / "within_mouse_segment_id_confusion_matrices.json"
        with open(confusion_json_path, "w", encoding="utf-8") as fp:
            json.dump(confusion_payload, fp, indent=2)
        print(f"Wrote confusion JSON: {confusion_json_path}")

        predictions_json_path = output_folder / "within_mouse_segment_id_prediction_outputs.json"
        with open(predictions_json_path, "w", encoding="utf-8") as fp:
            json.dump(prediction_payload, fp, indent=2)
        print(f"Wrote predictions JSON: {predictions_json_path}")

        summary_json_path = output_folder / "within_mouse_segment_id_metrics.json"
        summary_payload["summary_csv_path"] = str(summary_csv_path)
        summary_payload["skip_csv_path"] = str(skip_csv_path)
        summary_payload["confusion_json_path"] = str(confusion_json_path)
        summary_payload["predictions_json_path"] = str(predictions_json_path)
        with open(summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2)
        print(f"Wrote summary JSON: {summary_json_path}")

        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return {
        "log_path": str(log_path),
        "summary_csv_path": str(output_folder / "within_mouse_segment_id_metrics.csv"),
        "summary_json_path": str(output_folder / "within_mouse_segment_id_metrics.json"),
        "skip_csv_path": str(output_folder / "within_mouse_segment_id_skips.csv"),
        "confusion_json_path": str(output_folder / "within_mouse_segment_id_confusion_matrices.json"),
        "predictions_json_path": str(output_folder / "within_mouse_segment_id_prediction_outputs.json"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Within-mouse segment-ID decoding using Turnover segment vectors and grid segment windows "
            "with selectable models and configurable CV schemes."
        )
    )

    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)
    parser.add_argument("--p-active", required=True, type=int)
    parser.add_argument("--per-trial-thresh", required=True, type=_str2bool)

    parser.add_argument("--mice", default=None, type=_opt_csv_list)
    parser.add_argument(
        "--models",
        default=None,
        type=_opt_csv_list,
        help=(
            "Comma-separated models to run: "
            "logreg,mlp,cnn1d,cnn3d_raw,cnn3d_norm. "
            "Default: all models."
        ),
    )
    parser.add_argument("--clip-frames", default=None, type=_opt_int)
    parser.add_argument("--grid-subdir", default="trials_grid")
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Directory for .npz vectorization caches. Default: <data-root>/<mouse>/cache",
    )
    parser.add_argument("--max-trials", default=None, type=_opt_int)

    parser.add_argument("--batch-size-vec", default=64, type=int)
    parser.add_argument("--batch-size-grid", default=16, type=int)
    parser.add_argument("--epochs-mlp", default=60, type=int)
    parser.add_argument("--epochs-cnn1d", default=60, type=int)
    parser.add_argument("--epochs-cnn3d", default=40, type=int)
    parser.add_argument("--lr-vec", default=1e-3, type=float)
    parser.add_argument("--lr-cnn3d", default=5e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--early-stop-patience", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers-dl", default=0, type=int)
    parser.add_argument(
        "--cv-scheme-logreg",
        default="loso",
        choices=["loso", "logo", "groupkfold", "kfold"],
        help="CV scheme for LogReg branch.",
    )
    parser.add_argument(
        "--cv-scheme-cnn3d",
        default="loso",
        choices=["loso", "logo", "groupkfold", "kfold"],
        help="CV scheme for 3D-CNN branch.",
    )
    parser.add_argument(
        "--cv-n-splits-logreg",
        default=5,
        type=int,
        help="Number of folds for k-based schemes in LogReg branch (ignored by loso/logo).",
    )
    parser.add_argument(
        "--cv-n-splits-cnn3d",
        default=5,
        type=int,
        help="Number of folds for k-based schemes in 3D-CNN branch (ignored by loso/logo).",
    )

    parser.add_argument("--seg-len-naturalimages", default=12, type=int)
    parser.add_argument("--seg-len-pinknoise", default=27, type=int)
    parser.add_argument("--seg-len-randomdots", default=60, type=int)
    parser.add_argument("--seg-len-gabor", default=25, type=int)
    parser.add_argument("--seg-len-gaussiandot", default=9, type=int)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    seg_lengths = dict(DEFAULT_SEGMENT_LENGTHS)
    seg_lengths["NaturalImages"] = int(args.seg_len_naturalimages)
    seg_lengths["PinkNoise"] = int(args.seg_len_pinknoise)
    seg_lengths["RandomDots"] = int(args.seg_len_randomdots)
    seg_lengths["Gabor"] = int(args.seg_len_gabor)
    seg_lengths["GaussianDot"] = int(args.seg_len_gaussiandot)
    try:
        model_order = _resolve_models(args.models)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    state = RunState(
        output_folder=output_folder,
        data_root=args.data_root,
        meta_root=args.meta_root,
        p_active=args.p_active,
        per_trial_thresh=args.per_trial_thresh,
        zz_folder=_build_zz_folder(args.p_active, args.per_trial_thresh),
        mice=args.mice,
        models=model_order,
        clip_frames=args.clip_frames,
        grid_subdir=args.grid_subdir,
        cache_dir=args.cache_dir,
        max_trials=args.max_trials,
        batch_size_vec=args.batch_size_vec,
        epochs_cnn3d=args.epochs_cnn3d,
        epochs_mlp=args.epochs_mlp,
        epochs_cnn1d=args.epochs_cnn1d,
        lr_vec=args.lr_vec,
        lr_cnn3d=args.lr_cnn3d,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        batch_size_grid=args.batch_size_grid,
        seed=args.seed,
        device=args.device,
        num_workers_dl=args.num_workers_dl,
        segment_lengths=seg_lengths,
        cv_scheme_logreg=str(args.cv_scheme_logreg),
        cv_scheme_cnn3d=str(args.cv_scheme_cnn3d),
        cv_n_splits_logreg=int(args.cv_n_splits_logreg),
        cv_n_splits_cnn3d=int(args.cv_n_splits_cnn3d),
    )

    try:
        artifacts = run_pipeline(state)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("Run completed.")
    print(f"Log file: {artifacts['log_path']}")
    print(f"Summary JSON: {artifacts['summary_json_path']}")
    print(f"Summary CSV: {artifacts['summary_csv_path']}")
    print(f"Skip CSV: {artifacts['skip_csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
