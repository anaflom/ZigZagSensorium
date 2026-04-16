#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-mouse leave-one-mouse-out classification.

For each eligible test mouse, trains on all other eligible mice and evaluates
on the held-out mouse. Eligibility requires at least two distinct labels after
metadata filtering and vec/grid trial alignment.

Per fold, the label space is restricted to labels present in both the test
mouse and the pooled training mice.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from utils import (
    _discover_mice,
    build_vectorization_cache_stem,
    create_vectorization,
    load_labelled_barcodes,
    load_labelled_grid_paths,
    load_vectorization_cache,
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
    grid_subdir: str
    cache_dir: Optional[Path]
    force_recompute: bool
    max_trials: Optional[int]
    batch_size_vec: int
    batch_size_grid: int
    epochs_mlp: int
    epochs_cnn1d: int
    epochs_cnn3d: int
    lr_vec: float
    lr_cnn3d: float
    weight_decay: float
    early_stop_patience: int
    seed: int
    device: str
    num_workers_dl: int


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


def _opt_csv_list(value: str) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items if items else None


def _build_zz_folder(p_active: int, per_trial_thresh: bool) -> str:
    if per_trial_thresh:
        return f"trials_zz-thresh-{p_active}-per-trial"
    return f"trials_zz-thresh-{p_active}"


def _resolve_mouse_cache_dir(state: RunState, mouse_name: str) -> Path:
    if state.cache_dir is not None:
        return state.cache_dir
    return state.data_root / mouse_name / "cache"


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


def _infer_cnn1d_shape(feat_dim: int, clip_frames: int) -> Tuple[int, int]:
    if clip_frames > 0 and feat_dim % clip_frames == 0:
        return int(feat_dim // clip_frames), int(clip_frames)
    return 1, int(feat_dim)


class VectorDataset(Dataset):
    """Simple tensor dataset for vectorized features."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx])


class GridTrialDataset(Dataset):
    """Lazy grid loader from <data_root>/<mouse>/<trials_grid> files."""

    def __init__(
        self,
        grid_paths: Sequence[Path],
        y: np.ndarray,
        valid_frames: np.ndarray,
        clip_frames: int,
    ) -> None:
        self.grid_paths = list(grid_paths)
        self.y = np.asarray(y, dtype=np.int64)
        self.valid_frames = np.asarray(valid_frames, dtype=np.int64)
        self.clip_frames = int(clip_frames)

        if len(self.grid_paths) == 0:
            raise RuntimeError("GridTrialDataset received no paths")
        if len(self.grid_paths) != int(self.y.shape[0]):
            raise RuntimeError("Grid paths/labels length mismatch")
        if len(self.grid_paths) != int(self.valid_frames.shape[0]):
            raise RuntimeError("Grid paths/valid_frames length mismatch")

        first_shape = np.load(self.grid_paths[0], mmap_mode="r").shape
        if len(first_shape) != 4:
            raise RuntimeError(f"Grid file must be 4D, got {first_shape} for {self.grid_paths[0]}")
        self.in_channels = int(first_shape[2])

    def __len__(self) -> int:
        return len(self.grid_paths)

    def __getitem__(self, idx: int):
        arr = np.load(self.grid_paths[idx])
        if arr.ndim != 4:
            raise RuntimeError(f"Expected 4D grid array, got shape {arr.shape} in {self.grid_paths[idx]}")

        x = arr.transpose(2, 3, 0, 1).astype(np.float32, copy=False)
        t_eff = int(min(self.clip_frames, self.valid_frames[idx], x.shape[1]))
        if t_eff <= 0:
            raise RuntimeError(f"Invalid effective clip length={t_eff} for sample {self.grid_paths[idx]}")

        out = np.zeros((x.shape[0], self.clip_frames, x.shape[2], x.shape[3]), dtype=np.float32)
        out[:, :t_eff, :, :] = x[:, :t_eff, :, :]
        return torch.tensor(out, dtype=torch.float32), int(self.y[idx])


class MLP(nn.Module):
    """MLP baseline for vectorized features."""

    def __init__(self, n_classes: int, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """1D CNN baseline for vectorized features."""

    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        seq_len: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.conv1 = nn.Conv1d(self.in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = x.view(bsz, self.in_channels, self.seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, ceil_mode=True)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, ceil_mode=True)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class CNN3D(nn.Module):
    """3D CNN operating on grids shaped (C, T, H, W)."""

    def __init__(self, n_classes: int, in_channels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), ceil_mode=True)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), ceil_mode=True)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = torch.as_tensor(yb, dtype=torch.long, device=device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    for xb, _yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _build_train_val_indices(y_train: np.ndarray, seed: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    class_counts = np.bincount(y_train)
    if len(y_train) < 10 or int(class_counts.min()) < 2:
        return None, None

    val_size = max(len(class_counts), int(round(0.2 * len(y_train))))
    if val_size >= len(y_train):
        return None, None

    all_idx = np.arange(len(y_train))
    try:
        train_idx, val_idx = train_test_split(
            all_idx,
            test_size=val_size,
            stratify=y_train,
            random_state=seed,
        )
    except ValueError:
        return None, None
    return np.asarray(train_idx), np.asarray(val_idx)


def _train_eval_nn(
    make_model,
    train_ds: Dataset,
    y_train: np.ndarray,
    test_ds: Dataset,
    test_y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    train_idx, val_idx = _build_train_val_indices(y_train, seed=seed)

    if train_idx is None or val_idx is None:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = None
        val_y = None
    else:
        train_loader = DataLoader(
            Subset(train_ds, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            Subset(train_ds, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_y = y_train[val_idx]

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        if val_loader is not None
        else None
    )
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -1.0
    wait = 0

    for _epoch in range(int(epochs)):
        _train_epoch(model, train_loader, optimizer, criterion, device)
        if val_loader is not None and val_y is not None:
            val_pred = _predict(model, val_loader, device)
            val_f1 = float(f1_score(val_y, val_pred, average="macro", zero_division=0))
            scheduler.step(1.0 - val_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= int(patience):
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    pred = _predict(model, test_loader, device)
    metrics = {
        "accuracy": float(accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
    }
    return metrics, pred


def _train_eval_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }
    return metrics, pred


def _load_mouse_data(state: RunState, mouse_name: str, global_clip: int) -> Optional[Dict[str, Any]]:
    barcodes, labels, trial_ids, valid_frames = load_labelled_barcodes(
        state.data_root,
        state.meta_root,
        mouse_name,
        state.zz_folder,
        max_trials=state.max_trials,
    )
    if len(barcodes) == 0:
        return None
    if len(np.unique(labels)) < 2:
        return None

    cache_stem = build_vectorization_cache_stem(
        mouse_name=mouse_name,
        method=state.method,
        p_active=state.p_active,
        per_trial_thresh=state.per_trial_thresh,
        clip_frames=global_clip,
    )
    mouse_cache_dir = _resolve_mouse_cache_dir(state, mouse_name)
    mouse_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = mouse_cache_dir / f"{cache_stem}.npz"

    if cache_path.exists() and not state.force_recompute:
        cache = load_vectorization_cache(cache_path)
        if "features" in cache:
            xmat = np.asarray(cache["features"])
        elif "X" in cache:
            xmat = np.asarray(cache["X"])
        else:
            raise RuntimeError(f"Cache missing feature matrix: {cache_path}")
        xmat = np.nan_to_num(xmat)
        vec_source = "cache"
    else:
        vec_out = create_vectorization(
            barcodes,
            state.method,
            clip_frames=global_clip,
            output_folder=mouse_cache_dir,
            cache_stem=cache_stem,
            mouse_name=mouse_name,
            labels=labels,
            trial_ids=trial_ids,
            valid_frames=valid_frames,
        )
        xmat = np.asarray(vec_out["features"])
        vec_source = "computed"

    grid_paths, grid_labels, grid_trial_ids, grid_valid_frames = load_labelled_grid_paths(
        state.data_root,
        state.meta_root,
        mouse_name,
        grid_subdir=state.grid_subdir,
    )
    if len(grid_paths) == 0:
        return None

    vec_tid_to_i = {int(tid): i for i, tid in enumerate(trial_ids)}
    grid_tid_to_i = {int(tid): i for i, tid in enumerate(grid_trial_ids)}
    common_trial_ids = [tid for tid in vec_tid_to_i if tid in grid_tid_to_i]
    if not common_trial_ids:
        return None

    vec_take = np.array([vec_tid_to_i[tid] for tid in common_trial_ids], dtype=np.int64)
    grid_take = np.array([grid_tid_to_i[tid] for tid in common_trial_ids], dtype=np.int64)

    x_vec = np.asarray(xmat[vec_take], dtype=np.float64)
    labels_common = np.asarray(labels[vec_take])
    grid_paths_common = [grid_paths[i] for i in grid_take]
    grid_frames_common = np.asarray(grid_valid_frames[grid_take], dtype=np.int64)
    grid_labels_common = np.asarray(grid_labels[grid_take])

    if not np.all(labels_common == grid_labels_common):
        raise RuntimeError(f"Label mismatch between vectorization and grid data for {mouse_name}")
    if len(np.unique(labels_common)) < 2:
        return None

    first_shape = np.load(grid_paths_common[0], mmap_mode="r").shape
    in_channels = int(first_shape[2])

    return {
        "mouse": mouse_name,
        "x_vec": x_vec,
        "labels": labels_common,
        "grid_paths": grid_paths_common,
        "grid_frames": grid_frames_common,
        "in_channels": in_channels,
        "cache_path": str(cache_path),
        "vec_source": vec_source,
        "n_trials": int(len(common_trial_ids)),
    }


def run_pipeline(state: RunState) -> Dict[str, object]:
    _set_seed(state.seed)
    device = _resolve_device(state.device)

    output_folder = state.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    figures_dir = output_folder / "figures"
    logs_dir = output_folder / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"

    with open(log_path, "w", encoding="utf-8") as log_fp, redirect_stdout(log_fp), redirect_stderr(log_fp):
        print("=" * 90)
        print("Cross-Mouse LOMO Classification: LogReg/MLP/1D-CNN vs 3D-CNN")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)

        print("\nResolved arguments:")
        print(f"  output_folder:     {state.output_folder}")
        print(f"  data_root:         {state.data_root}")
        print(f"  meta_root:         {state.meta_root}")
        print(f"  p_active:          {state.p_active}")
        print(f"  per_trial_thresh:  {state.per_trial_thresh}")
        print(f"  zz_folder:         {state.zz_folder}")
        print(f"  method:            {state.method}")
        print(f"  mice:              {state.mice}")
        print(f"  clip_frames:       {state.clip_frames}")
        print(f"  grid_subdir:       {state.grid_subdir}")
        print(f"  force_recompute:   {state.force_recompute}")
        print(f"  max_trials:        {state.max_trials}")
        print(f"  device:            {device}")

        discovered_mice = _discover_mice(state.data_root)
        selected_mice = state.mice if state.mice is not None else discovered_mice
        selected_mice = [m for m in selected_mice if m in discovered_mice]
        if not selected_mice:
            raise RuntimeError("No valid mice selected for cross-mouse classification.")

        print(f"\nDiscovered mice: {len(discovered_mice)}")
        print(f"Selected mice: {len(selected_mice)}")

        if state.clip_frames is not None:
            global_clip = int(state.clip_frames)
            print(f"\nUsing user clip_frames={global_clip}")
        else:
            print("\nPre-scanning valid_frames to determine global clip_frames ...")
            min_frames_list: List[int] = []
            for mouse_name in selected_mice:
                try:
                    _barcodes, _labels, _trial_ids, valid_frames = load_labelled_barcodes(
                        state.data_root,
                        state.meta_root,
                        mouse_name,
                        state.zz_folder,
                        max_trials=state.max_trials,
                    )
                    if len(valid_frames) > 0:
                        min_frames_list.append(int(valid_frames.min()))
                except Exception as exc:
                    print(f"  Warning: could not pre-scan {mouse_name}: {exc}")
            if not min_frames_list:
                raise RuntimeError("Could not determine global clip_frames from any selected mouse.")
            global_clip = min(min_frames_list)
            print(f"  global_clip_frames={global_clip} (min across {len(min_frames_list)} mice)")

        print("\nLoading mouse datasets ...")
        mouse_data: Dict[str, Dict[str, Any]] = {}
        for mouse_name in selected_mice:
            print(f"  {mouse_name} ...", end=" ")
            try:
                data = _load_mouse_data(state, mouse_name, global_clip)
                if data is None:
                    print("skipped (single label, no grid, or no overlapping trials)")
                    continue
                mouse_data[mouse_name] = data
                print(f"ok (trials={data['n_trials']}, source={data['vec_source']})")
            except Exception as exc:
                print(f"FAILED: {exc}")
                traceback.print_exc()

        eligible_mice = sorted(mouse_data.keys())
        if len(eligible_mice) < 2:
            raise RuntimeError(f"Need at least 2 eligible mice for LOMO, got {len(eligible_mice)}")

        print(f"\nEligible mice for LOMO: {len(eligible_mice)}")

        model_order = ["logreg", "mlp", "cnn1d", "cnn3d"]
        model_titles = {
            "logreg": "LogReg (vector)",
            "mlp": "MLP (vector)",
            "cnn1d": "1D-CNN (vector)",
            "cnn3d": "3D-CNN (grid)",
        }
        per_fold: Dict[str, Dict[str, Any]] = {}
        confusion_payload: Dict[str, Dict[str, Any]] = {}

        for test_mouse in eligible_mice:
            candidate_train_mice = [m for m in eligible_mice if m != test_mouse]
            print(f"\n## Test mouse: {test_mouse}")
            print(f"  Candidate train pool: {len(candidate_train_mice)} mice")
            try:
                test_data = mouse_data[test_mouse]
                test_label_set = set(np.unique(test_data["labels"]).tolist())
                train_label_union: set = set()
                for mouse_name in candidate_train_mice:
                    train_label_union.update(np.unique(mouse_data[mouse_name]["labels"]).tolist())

                shared_labels = sorted(test_label_set & train_label_union)
                if len(shared_labels) < 2:
                    print(
                        f"  Skipped: fewer than 2 shared labels "
                        f"(test={sorted(test_label_set)}, train={sorted(train_label_union)})"
                    )
                    continue

                encoder = LabelEncoder().fit(shared_labels)

                test_mask = np.isin(test_data["labels"], shared_labels)
                if not np.any(test_mask):
                    print("  Skipped: no test samples after shared-label filtering")
                    continue

                test_x_vec = np.asarray(test_data["x_vec"][test_mask], dtype=np.float64)
                test_labels = np.asarray(test_data["labels"][test_mask])
                test_y = encoder.transform(test_labels)
                test_grid_paths = [test_data["grid_paths"][i] for i in np.where(test_mask)[0]]
                test_grid_frames = np.asarray(test_data["grid_frames"][test_mask], dtype=np.int64)

                active_train_mice: List[str] = []
                train_x_parts: List[np.ndarray] = []
                train_y_parts: List[np.ndarray] = []
                train_grid_paths: List[Path] = []
                train_grid_frames_list: List[int] = []
                vec_sources_by_mouse: Dict[str, str] = {}
                cache_paths_by_mouse: Dict[str, str] = {test_mouse: test_data["cache_path"]}
                vec_sources_by_mouse[test_mouse] = test_data["vec_source"]

                for mouse_name in candidate_train_mice:
                    train_data = mouse_data[mouse_name]
                    mask = np.isin(train_data["labels"], shared_labels)
                    if not np.any(mask):
                        continue
                    active_train_mice.append(mouse_name)
                    train_x_parts.append(np.asarray(train_data["x_vec"][mask], dtype=np.float64))
                    train_y_parts.append(encoder.transform(np.asarray(train_data["labels"][mask])))
                    idxs = np.where(mask)[0]
                    train_grid_paths.extend([train_data["grid_paths"][i] for i in idxs])
                    train_grid_frames_list.extend(np.asarray(train_data["grid_frames"][mask], dtype=np.int64).tolist())
                    vec_sources_by_mouse[mouse_name] = train_data["vec_source"]
                    cache_paths_by_mouse[mouse_name] = train_data["cache_path"]

                if not train_x_parts:
                    print("  Skipped: no training samples after shared-label filtering")
                    continue

                train_x_vec = np.concatenate(train_x_parts, axis=0)
                train_y = np.concatenate(train_y_parts, axis=0)
                train_grid_frames = np.asarray(train_grid_frames_list, dtype=np.int64)
                if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
                    print("  Skipped: train/test data collapsed to a single class after shared-label filtering")
                    continue

                scaler = StandardScaler().fit(train_x_vec)
                x_train_scaled = scaler.transform(train_x_vec)
                x_test_scaled = scaler.transform(test_x_vec)
                cnn1d_channels, cnn1d_seq_len = _infer_cnn1d_shape(train_x_vec.shape[1], global_clip)
                vec_source_summary = (
                    next(iter(set(vec_sources_by_mouse.values())))
                    if len(set(vec_sources_by_mouse.values())) == 1
                    else "mixed"
                )
                cache_path_summary = (
                    next(iter(set(cache_paths_by_mouse.values())))
                    if len(set(cache_paths_by_mouse.values())) == 1
                    else "multiple"
                )

                print(
                    f"  Shared labels={shared_labels}, train_trials={len(train_y)}, "
                    f"test_trials={len(test_y)}, train_mice={len(active_train_mice)}"
                )

                logreg_metrics, logreg_pred = _train_eval_logreg(
                    x_train_scaled,
                    train_y,
                    x_test_scaled,
                    test_y,
                )

                mlp_metrics, mlp_pred = _train_eval_nn(
                    make_model=lambda: MLP(n_classes=len(shared_labels), input_dim=train_x_vec.shape[1]),
                    train_ds=VectorDataset(x_train_scaled, train_y),
                    y_train=train_y,
                    test_ds=VectorDataset(x_test_scaled, test_y),
                    test_y=test_y,
                    epochs=state.epochs_mlp,
                    lr=state.lr_vec,
                    batch_size=state.batch_size_vec,
                    patience=state.early_stop_patience,
                    weight_decay=state.weight_decay,
                    device=device,
                    num_workers=state.num_workers_dl,
                    seed=state.seed,
                )

                cnn1d_metrics, cnn1d_pred = _train_eval_nn(
                    make_model=lambda: CNN1D(
                        n_classes=len(shared_labels),
                        in_channels=cnn1d_channels,
                        seq_len=cnn1d_seq_len,
                    ),
                    train_ds=VectorDataset(x_train_scaled, train_y),
                    y_train=train_y,
                    test_ds=VectorDataset(x_test_scaled, test_y),
                    test_y=test_y,
                    epochs=state.epochs_cnn1d,
                    lr=state.lr_vec,
                    batch_size=state.batch_size_vec,
                    patience=state.early_stop_patience,
                    weight_decay=state.weight_decay,
                    device=device,
                    num_workers=state.num_workers_dl,
                    seed=state.seed,
                )

                train_grid_ds = GridTrialDataset(
                    grid_paths=train_grid_paths,
                    y=train_y,
                    valid_frames=train_grid_frames,
                    clip_frames=global_clip,
                )
                test_grid_ds = GridTrialDataset(
                    grid_paths=test_grid_paths,
                    y=test_y,
                    valid_frames=test_grid_frames,
                    clip_frames=global_clip,
                )
                cnn3d_metrics, cnn3d_pred = _train_eval_nn(
                    make_model=lambda: CNN3D(
                        n_classes=len(shared_labels),
                        in_channels=test_data["in_channels"],
                    ),
                    train_ds=train_grid_ds,
                    y_train=train_y,
                    test_ds=test_grid_ds,
                    test_y=test_y,
                    epochs=state.epochs_cnn3d,
                    lr=state.lr_cnn3d,
                    batch_size=state.batch_size_grid,
                    patience=state.early_stop_patience,
                    weight_decay=state.weight_decay,
                    device=device,
                    num_workers=state.num_workers_dl,
                    seed=state.seed,
                )

                model_metrics = {
                    "logreg": logreg_metrics,
                    "mlp": mlp_metrics,
                    "cnn1d": cnn1d_metrics,
                    "cnn3d": cnn3d_metrics,
                }
                model_preds = {
                    "logreg": logreg_pred,
                    "mlp": mlp_pred,
                    "cnn1d": cnn1d_pred,
                    "cnn3d": cnn3d_pred,
                }
                best_model = max(model_order, key=lambda mk: model_metrics[mk]["macro_f1"])

                per_fold[test_mouse] = {
                    "test_mouse": test_mouse,
                    "train_mice": active_train_mice,
                    "n_train_mice": len(active_train_mice),
                    "n_train_trials": int(len(train_y)),
                    "n_test_trials": int(len(test_y)),
                    "n_features": int(train_x_vec.shape[1]),
                    "clip_frames": int(global_clip),
                    "shared_labels": shared_labels,
                    "n_classes": int(len(shared_labels)),
                    "cnn1d_channels": int(cnn1d_channels),
                    "cnn1d_seq_len": int(cnn1d_seq_len),
                    "best_model": best_model,
                    "vec_source": vec_source_summary,
                    "cache_path": cache_path_summary,
                    "cache_paths_by_mouse": cache_paths_by_mouse,
                    "vec_sources_by_mouse": vec_sources_by_mouse,
                    "models": model_metrics,
                }
                model_cms: Dict[str, np.ndarray] = {
                    mk: confusion_matrix(test_y, model_preds[mk], labels=np.arange(len(shared_labels)))
                    for mk in model_order
                }
                confusion_payload[test_mouse] = {
                    "labels": shared_labels,
                    "cms": model_cms,
                    "best_model": best_model,
                }

                for mk in model_order:
                    mm = model_metrics[mk]
                    print(f"    {mk:6s} acc={mm['accuracy']:.3f} f1={mm['macro_f1']:.3f}")
                print(f"  best={best_model}:{model_metrics[best_model]['macro_f1']:.3f}")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                traceback.print_exc()

        if not per_fold:
            raise RuntimeError("No LOMO fold produced results.")

        fold_order = sorted(per_fold.keys())

        fig, ax = plt.subplots(figsize=(max(9, len(fold_order) * 1.3), 5.0))
        x = np.arange(len(fold_order))
        width = 0.18
        offsets = {
            "logreg": -1.5 * width,
            "mlp": -0.5 * width,
            "cnn1d": 0.5 * width,
            "cnn3d": 1.5 * width,
        }
        colors = {
            "logreg": "#4C72B0",
            "mlp": "#55A868",
            "cnn1d": "#8172B2",
            "cnn3d": "#DD8452",
        }
        for mk in model_order:
            vals = [float(per_fold[m]["models"][mk]["macro_f1"]) for m in fold_order]
            ax.bar(x + offsets[mk], vals, width, label=model_titles[mk], alpha=0.85, color=colors[mk])
        ax.set_xticks(x)
        ax.set_xticklabels(fold_order, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Cross-mouse LOMO macro-F1 by test mouse ({state.method})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        fig1 = figures_dir / "01_lomo_macro_f1_by_test_mouse.png"
        fig.savefig(fig1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig1}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        mean_acc = [np.mean([per_fold[m]["models"][mk]["accuracy"] for m in fold_order]) for mk in model_order]
        mean_f1 = [np.mean([per_fold[m]["models"][mk]["macro_f1"] for m in fold_order]) for mk in model_order]
        std_acc = [np.std([per_fold[m]["models"][mk]["accuracy"] for m in fold_order]) for mk in model_order]
        std_f1 = [np.std([per_fold[m]["models"][mk]["macro_f1"] for m in fold_order]) for mk in model_order]
        xi = np.arange(len(model_order))
        color_list = [colors[m] for m in model_order]
        axes[0].bar(xi, mean_acc, yerr=std_acc, capsize=4, color=color_list, alpha=0.85)
        axes[0].set_xticks(xi)
        axes[0].set_xticklabels([model_titles[m] for m in model_order], rotation=25, ha="right", fontsize=8)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Mean accuracy across test mice")
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(xi, mean_f1, yerr=std_f1, capsize=4, color=color_list, alpha=0.85)
        axes[1].set_xticks(xi)
        axes[1].set_xticklabels([model_titles[m] for m in model_order], rotation=25, ha="right", fontsize=8)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("Macro-F1")
        axes[1].set_title("Mean macro-F1 across test mice")
        axes[1].grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig2 = figures_dir / "02_lomo_mean_scores.png"
        fig.savefig(fig2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig2}")

        # Figure 3: Confusion matrices for all 4 classifiers per test mouse.
        # Layout: rows = test mice, columns = 4 models.
        n_mice = len(fold_order)
        n_models = len(model_order)
        fig, axes = plt.subplots(
            n_mice,
            n_models,
            figsize=(4.4 * n_models, 3.8 * n_mice),
            squeeze=False,
        )
        for row_idx, mouse_name in enumerate(fold_order):
            payload = confusion_payload[mouse_name]
            labels_order = payload["labels"]
            best_model = payload["best_model"]
            for col_idx, mk in enumerate(model_order):
                ax = axes[row_idx][col_idx]
                cm_arr = np.asarray(payload["cms"][mk], dtype=float)
                row_sums = cm_arr.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)
                ConfusionMatrixDisplay(cm_norm, display_labels=labels_order).plot(
                    ax=ax,
                    cmap="Blues",
                    colorbar=False,
                    values_format=".2f",
                )
                title = f"{mouse_name}\n{model_titles[mk]}"
                if mk == best_model:
                    title += " ★"
                ax.set_title(title, fontsize=7)
        fig.suptitle("Normalized confusion matrices — all classifiers per test mouse (★ = best)", fontsize=11)
        fig.tight_layout()
        fig3 = figures_dir / "03_all_classifier_confusion_matrices.png"
        fig.savefig(fig3, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig3}")

        summary_json_path = output_folder / "cross_mouse_metrics.json"
        summary_csv_path = output_folder / "cross_mouse_metrics.csv"
        payload = {
            "method": state.method,
            "p_active": state.p_active,
            "per_trial_thresh": state.per_trial_thresh,
            "zz_folder": state.zz_folder,
            "grid_subdir": state.grid_subdir,
            "global_clip_frames": int(global_clip),
            "eligible_mice": eligible_mice,
            "n_lomo_folds": len(fold_order),
            "models": model_order,
            "results": per_fold,
            "figures": [str(fig1), str(fig2), str(fig3)],
            "log_path": str(log_path),
            "cache_dir": (
                str(state.cache_dir)
                if state.cache_dir is not None
                else "<data_root>/<mouse>/cache"
            ),
            "device": str(device),
        }
        with open(summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"Wrote summary JSON: {summary_json_path}")

        with open(summary_csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "test_mouse",
                    "train_mice",
                    "n_train_mice",
                    "method",
                    "model",
                    "input",
                    "n_train_trials",
                    "n_test_trials",
                    "n_features",
                    "clip_frames",
                    "n_classes",
                    "shared_labels",
                    "accuracy",
                    "macro_f1",
                    "vec_source",
                    "cache_path",
                    "best_model",
                ]
            )
            for mouse_name in fold_order:
                row = per_fold[mouse_name]
                train_mice_str = "|".join(row["train_mice"])
                shared_label_str = "|".join(row["shared_labels"])
                for model_name in model_order:
                    model_metrics = row["models"][model_name]
                    writer.writerow(
                        [
                            row["test_mouse"],
                            train_mice_str,
                            row["n_train_mice"],
                            state.method,
                            model_name,
                            "grid" if model_name == "cnn3d" else "vector",
                            row["n_train_trials"],
                            row["n_test_trials"],
                            row["n_features"],
                            row["clip_frames"],
                            row["n_classes"],
                            shared_label_str,
                            model_metrics["accuracy"],
                            model_metrics["macro_f1"],
                            row["vec_source"],
                            row["cache_path"],
                            row["best_model"],
                        ]
                    )
        print(f"Wrote summary CSV: {summary_csv_path}")

    return {
        "log_path": str(log_path),
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "cache_dir": (
            str(state.cache_dir)
            if state.cache_dir is not None
            else "<data_root>/<mouse>/cache"
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-mouse leave-one-mouse-out classification using selected zigzag vectorization "
            "(LogReg/MLP/1D-CNN) and 3D-CNN on raw grids."
        )
    )
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)
    parser.add_argument("--p-active", required=True, type=int)
    parser.add_argument("--per-trial-thresh", required=True, type=_str2bool)
    parser.add_argument("--method", required=True)

    parser.add_argument("--mice", default=None, type=_opt_csv_list)
    parser.add_argument("--clip-frames", default=None, type=_opt_int)
    parser.add_argument("--grid-subdir", default="trials_grid")
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Directory for .npz vectorization caches. Default: <data-root>/<mouse>/cache",
    )
    parser.add_argument("--force-recompute", action="store_true")
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
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    state = RunState(
        output_folder=output_folder,
        data_root=args.data_root,
        meta_root=args.meta_root,
        p_active=args.p_active,
        per_trial_thresh=args.per_trial_thresh,
        zz_folder=_build_zz_folder(args.p_active, args.per_trial_thresh),
        method=args.method,
        mice=args.mice,
        clip_frames=args.clip_frames,
        grid_subdir=args.grid_subdir,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
        max_trials=args.max_trials,
        batch_size_vec=args.batch_size_vec,
        batch_size_grid=args.batch_size_grid,
        epochs_mlp=args.epochs_mlp,
        epochs_cnn1d=args.epochs_cnn1d,
        epochs_cnn3d=args.epochs_cnn3d,
        lr_vec=args.lr_vec,
        lr_cnn3d=args.lr_cnn3d,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        device=args.device,
        num_workers_dl=args.num_workers_dl,
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
    print(f"Vectorization cache dir: {artifacts['cache_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())