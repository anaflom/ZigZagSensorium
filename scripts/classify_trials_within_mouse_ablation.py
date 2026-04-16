#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Within-mouse ablation: vectorization models vs 3D-CNN on raw grids."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# Force a non-interactive backend for cluster/headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
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
    n_splits: int
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


def _short_mouse_name(name: str) -> str:
    m = re.match(r"dynamic(\d+)-(\d+)-(\d+)", name)
    if m:
        return f"rec-{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return name


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

        # Input file convention from compute_grid_activation.py is (Nx, Ny, Nz, T).
        x = arr.transpose(2, 3, 0, 1).astype(np.float32, copy=False)  # (C=Nz, T, H, W)

        # Clip to the smallest reliable temporal support and zero-pad to fixed clip.
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


def _run_nn_cv(
    make_model,
    train_dataset_builder,
    y_int: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    weight_decay: float,
    device: torch.device,
    num_workers: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    fold_acc: List[float] = []
    fold_f1: List[float] = []
    oof_pred = np.full_like(y_int, fill_value=-1, dtype=np.int64)

    for train_idx, val_idx in splits:
        train_ds, val_ds = train_dataset_builder(train_idx, val_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = make_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_f1 = -1.0
        best_acc = -1.0
        best_pred: Optional[np.ndarray] = None
        wait = 0

        for _epoch in range(int(epochs)):
            _train_epoch(model, train_loader, optimizer, criterion, device)
            val_pred = _predict(model, val_loader, device)
            val_true = y_int[val_idx]
            val_acc = float(accuracy_score(val_true, val_pred))
            val_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
            scheduler.step(1.0 - val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                best_pred = val_pred.copy()
                wait = 0
            else:
                wait += 1
                if wait >= int(patience):
                    break

        if best_pred is None:
            raise RuntimeError("Training failed to produce predictions")

        oof_pred[val_idx] = best_pred
        fold_acc.append(best_acc)
        fold_f1.append(best_f1)

    if np.any(oof_pred < 0):
        raise RuntimeError("OOF prediction array contains unset indices")

    metrics = {
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
    }
    return metrics, oof_pred


def _run_logreg_cv(
    x: np.ndarray,
    y_int: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, float], np.ndarray]:
    fold_acc: List[float] = []
    fold_f1: List[float] = []
    oof_pred = np.full_like(y_int, fill_value=-1, dtype=np.int64)

    for train_idx, val_idx in splits:
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
        pipe.fit(x[train_idx], y_int[train_idx])
        pred = pipe.predict(x[val_idx])

        oof_pred[val_idx] = pred
        fold_acc.append(float(accuracy_score(y_int[val_idx], pred)))
        fold_f1.append(float(f1_score(y_int[val_idx], pred, average="macro", zero_division=0)))

    if np.any(oof_pred < 0):
        raise RuntimeError("OOF prediction array contains unset indices")

    metrics = {
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
    }
    return metrics, oof_pred


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
        print("Within-Mouse Ablation: LogReg/MLP/1D-CNN vs 3D-CNN")
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
        print(f"  n_splits:          {state.n_splits}")
        print(f"  max_trials:        {state.max_trials}")
        print(f"  device:            {device}")

        discovered_mice = _discover_mice(state.data_root)
        selected_mice = state.mice if state.mice is not None else discovered_mice
        selected_mice = [m for m in selected_mice if m in discovered_mice]

        if not selected_mice:
            raise RuntimeError("No valid mice selected for ablation.")

        print(f"\nDiscovered mice: {len(discovered_mice)}")
        print(f"Selected mice: {len(selected_mice)}")

        model_order = ["logreg", "mlp", "cnn1d", "cnn3d"]
        model_titles = {
            "logreg": "LogReg (vector)",
            "mlp": "MLP (vector)",
            "cnn1d": "1D-CNN (vector)",
            "cnn3d": "3D-CNN (grid)",
        }

        per_mouse: Dict[str, Dict[str, object]] = {}
        confusion_payload: Dict[str, Dict[str, object]] = {}

        for mouse_name in selected_mice:
            print(f"\n## Mouse: {mouse_name}")
            try:
                barcodes, labels, trial_ids, valid_frames = load_labelled_barcodes(
                    state.data_root,
                    state.meta_root,
                    mouse_name,
                    state.zz_folder,
                    max_trials=state.max_trials,
                )
                if len(barcodes) == 0:
                    print("  No labelled barcodes found; skipping.")
                    continue

                clip_used = state.clip_frames
                if clip_used is None:
                    clip_used = int(valid_frames.min())

                cache_stem = build_vectorization_cache_stem(
                    mouse_name=mouse_name,
                    method=state.method,
                    p_active=state.p_active,
                    per_trial_thresh=state.per_trial_thresh,
                    clip_frames=clip_used,
                )
                mouse_cache_dir = _resolve_mouse_cache_dir(state, mouse_name)
                mouse_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = mouse_cache_dir / f"{cache_stem}.npz"

                vec_trial_ids = [int(t) for t in trial_ids]
                use_cache = False
                if cache_path.exists() and not state.force_recompute:
                    cache = load_vectorization_cache(cache_path)
                    if "features" in cache:
                        xmat = np.asarray(cache["features"])
                    elif "X" in cache:
                        xmat = np.asarray(cache["X"])
                    else:
                        raise RuntimeError(f"Cache missing feature matrix: {cache_path}")
                    xmat = np.nan_to_num(xmat)

                    cache_trial_ids = None
                    if "trial_ids" in cache:
                        cache_trial_ids = [int(t) for t in np.asarray(cache["trial_ids"]).tolist()]

                    if cache_trial_ids is not None:
                        if len(cache_trial_ids) != int(xmat.shape[0]):
                            print(
                                "  Cache mismatch: trial_ids length differs from feature rows; "
                                "recomputing vectorization."
                            )
                        elif cache_trial_ids != vec_trial_ids:
                            print(
                                "  Cache mismatch: cached trial_ids differ from current trial_ids; "
                                "recomputing vectorization."
                            )
                        else:
                            use_cache = True
                    else:
                        if int(xmat.shape[0]) != len(vec_trial_ids):
                            print(
                                "  Cache mismatch: feature rows differ from current trials and "
                                "cache has no trial_ids; recomputing vectorization."
                            )
                        else:
                            use_cache = True

                    if use_cache:
                        vec_source = "cache"

                if not use_cache:
                    vec_out = create_vectorization(
                        barcodes,
                        state.method,
                        clip_frames=clip_used,
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
                    print(f"  No grid files found in {state.grid_subdir}; skipping mouse.")
                    continue

                vec_idx = {tid: i for i, tid in enumerate(vec_trial_ids)}
                grid_idx = {int(tid): i for i, tid in enumerate(grid_trial_ids)}

                common_trial_ids = [tid for tid in vec_trial_ids if tid in grid_idx]
                if len(common_trial_ids) == 0:
                    print("  No overlapping trial ids between vectorization and grid data; skipping mouse.")
                    continue

                vec_take = np.array([vec_idx[tid] for tid in common_trial_ids], dtype=np.int64)
                grid_take = np.array([grid_idx[tid] for tid in common_trial_ids], dtype=np.int64)

                x_common = np.asarray(xmat[vec_take], dtype=np.float64)
                labels_common = np.asarray(labels[vec_take])
                grid_paths_common = [grid_paths[i] for i in grid_take]
                grid_frames_common = np.asarray(grid_valid_frames[grid_take], dtype=np.int64)

                grid_labels_common = np.asarray(grid_labels[grid_take])
                if not np.all(labels_common == grid_labels_common):
                    raise RuntimeError("Label mismatch between vectorization and grid trial alignment")

                le = LabelEncoder().fit(labels_common)
                y_int = le.transform(labels_common)
                class_labels = list(le.classes_)
                class_counts = np.bincount(y_int, minlength=len(class_labels))
                min_count = int(class_counts.min())
                folds = min(int(state.n_splits), min_count)
                if len(class_labels) < 2:
                    print("  Skipping mouse: only one class after trial intersection.")
                    continue
                if folds < 2:
                    print(
                        f"  Skipping mouse: not enough samples per class for CV "
                        f"(min_count={min_count}, requested={state.n_splits})."
                    )
                    continue

                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=state.seed)
                splits = list(cv.split(np.zeros(len(y_int)), y_int))

                model_results: Dict[str, Dict[str, float]] = {}
                model_preds: Dict[str, np.ndarray] = {}

                # LogReg
                logreg_metrics, logreg_pred = _run_logreg_cv(x_common, y_int, splits)
                model_results["logreg"] = logreg_metrics
                model_preds["logreg"] = logreg_pred

                # MLP
                def build_mlp_dataset(train_idx: np.ndarray, val_idx: np.ndarray):
                    scaler = StandardScaler().fit(x_common[train_idx])
                    x_tr = scaler.transform(x_common[train_idx])
                    x_va = scaler.transform(x_common[val_idx])
                    return VectorDataset(x_tr, y_int[train_idx]), VectorDataset(x_va, y_int[val_idx])

                mlp_metrics, mlp_pred = _run_nn_cv(
                    make_model=lambda: MLP(n_classes=len(class_labels), input_dim=x_common.shape[1]),
                    train_dataset_builder=build_mlp_dataset,
                    y_int=y_int,
                    splits=splits,
                    epochs=state.epochs_mlp,
                    lr=state.lr_vec,
                    batch_size=state.batch_size_vec,
                    patience=state.early_stop_patience,
                    weight_decay=state.weight_decay,
                    device=device,
                    num_workers=state.num_workers_dl,
                )
                model_results["mlp"] = mlp_metrics
                model_preds["mlp"] = mlp_pred

                # 1D-CNN
                if clip_used > 0 and x_common.shape[1] % int(clip_used) == 0:
                    cnn1d_channels = int(x_common.shape[1] // int(clip_used))
                    cnn1d_seq_len = int(clip_used)
                else:
                    cnn1d_channels = 1
                    cnn1d_seq_len = int(x_common.shape[1])
                    print(
                        "  Warning: feature length is not divisible by clip_frames; "
                        "1D-CNN uses a single channel over full feature length."
                    )

                def build_cnn1d_dataset(train_idx: np.ndarray, val_idx: np.ndarray):
                    scaler = StandardScaler().fit(x_common[train_idx])
                    x_tr = scaler.transform(x_common[train_idx])
                    x_va = scaler.transform(x_common[val_idx])
                    return VectorDataset(x_tr, y_int[train_idx]), VectorDataset(x_va, y_int[val_idx])

                cnn1d_metrics, cnn1d_pred = _run_nn_cv(
                    make_model=lambda: CNN1D(
                        n_classes=len(class_labels),
                        in_channels=cnn1d_channels,
                        seq_len=cnn1d_seq_len,
                    ),
                    train_dataset_builder=build_cnn1d_dataset,
                    y_int=y_int,
                    splits=splits,
                    epochs=state.epochs_cnn1d,
                    lr=state.lr_vec,
                    batch_size=state.batch_size_vec,
                    patience=state.early_stop_patience,
                    weight_decay=state.weight_decay,
                    device=device,
                    num_workers=state.num_workers_dl,
                )
                model_results["cnn1d"] = cnn1d_metrics
                model_preds["cnn1d"] = cnn1d_pred

                # 3D-CNN
                grid_dataset = GridTrialDataset(
                    grid_paths=grid_paths_common,
                    y=y_int,
                    valid_frames=grid_frames_common,
                    clip_frames=int(clip_used),
                )

                def build_grid_dataset(train_idx: np.ndarray, val_idx: np.ndarray):
                    return Subset(grid_dataset, train_idx), Subset(grid_dataset, val_idx)

                cnn3d_metrics, cnn3d_pred = _run_nn_cv(
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
                model_results["cnn3d"] = cnn3d_metrics
                model_preds["cnn3d"] = cnn3d_pred

                best_model = max(model_order, key=lambda m: model_results[m]["mean_f1"])

                per_mouse[mouse_name] = {
                    "n_trials": int(len(y_int)),
                    "n_features": int(x_common.shape[1]),
                    "clip_frames": int(clip_used),
                    "cv_folds": int(folds),
                    "class_labels": class_labels,
                    "class_counts": {
                        class_labels[i]: int(class_counts[i]) for i in range(len(class_labels))
                    },
                    "source": vec_source,
                    "cache_path": str(cache_path),
                    "best_model": best_model,
                    "cnn1d_channels": int(cnn1d_channels),
                    "cnn1d_seq_len": int(cnn1d_seq_len),
                    "models": model_results,
                }
                model_cms: Dict[str, np.ndarray] = {}
                for mk in model_order:
                    mk_cm = confusion_matrix(y_int, model_preds[mk], labels=np.arange(len(class_labels)))
                    model_cms[mk] = mk_cm
                confusion_payload[mouse_name] = {
                    "labels": class_labels,
                    "cms": model_cms,
                    "best_model": best_model,
                }

                print(
                    f"  source={vec_source}, trials={len(y_int)}, feat={x_common.shape[1]}, folds={folds}, "
                    f"best={best_model}:{model_results[best_model]['mean_f1']:.3f}"
                )
                for mk in model_order:
                    mr = model_results[mk]
                    print(
                        f"    {mk:6s} acc={mr['mean_acc']:.3f}+/-{mr['std_acc']:.3f} "
                        f"f1={mr['mean_f1']:.3f}+/-{mr['std_f1']:.3f}"
                    )
            except Exception as exc:
                print(f"  FAILED: {exc}")
                traceback.print_exc()

        if not per_mouse:
            raise RuntimeError("No mouse produced ablation results.")

        mice_order = sorted(per_mouse.keys())

        # Figure 1: Macro-F1 and accuracy grouped bars by model per mouse.
        x = np.arange(len(mice_order))
        width = 0.18
        fig, (ax_f1, ax_acc) = plt.subplots(2, 1, figsize=(max(9, len(mice_order) * 1.3), 9.0), sharex=True)
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
            vals_f1 = [float(per_mouse[m]["models"][mk]["mean_f1"]) for m in mice_order]
            errs_f1 = [float(per_mouse[m]["models"][mk]["std_f1"]) for m in mice_order]
            vals_acc = [float(per_mouse[m]["models"][mk]["mean_acc"]) for m in mice_order]
            errs_acc = [float(per_mouse[m]["models"][mk]["std_acc"]) for m in mice_order]
            ax_f1.bar(
                x + offsets[mk], vals_f1, width, yerr=errs_f1, capsize=3,
                label=model_titles[mk], alpha=0.85, color=colors[mk],
            )
            ax_acc.bar(
                x + offsets[mk], vals_acc, width, yerr=errs_acc, capsize=3,
                label=model_titles[mk], alpha=0.85, color=colors[mk],
            )

        ax_f1.set_ylim(0, 1.05)
        ax_f1.set_ylabel("Macro-F1")
        ax_f1.set_title(f"Within-mouse ablation by mouse ({state.method})")
        ax_f1.grid(axis="y", alpha=0.25)
        ax_f1.legend(loc="upper right", ncol=2, fontsize=8)
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels([_short_mouse_name(m) for m in mice_order], rotation=30, ha="right", fontsize=8)
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"Within-mouse ablation accuracy by mouse ({state.method})")
        ax_acc.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig1 = figures_dir / "01_ablation_macro_f1_by_mouse.png"
        fig.savefig(fig1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig1}")

        # Figure 2: Mean performance across mice.
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        mean_acc = [np.mean([per_mouse[m]["models"][mk]["mean_acc"] for m in mice_order]) for mk in model_order]
        mean_f1 = [np.mean([per_mouse[m]["models"][mk]["mean_f1"] for m in mice_order]) for mk in model_order]
        std_acc = [np.std([per_mouse[m]["models"][mk]["mean_acc"] for m in mice_order]) for mk in model_order]
        std_f1 = [np.std([per_mouse[m]["models"][mk]["mean_f1"] for m in mice_order]) for mk in model_order]

        axes[0].bar(np.arange(len(model_order)), mean_acc, yerr=std_acc, capsize=4, color=[colors[m] for m in model_order], alpha=0.85)
        axes[0].set_xticks(np.arange(len(model_order)))
        axes[0].set_xticklabels([model_titles[m] for m in model_order], rotation=25, ha="right", fontsize=8)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Mean accuracy across mice")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(np.arange(len(model_order)), mean_f1, yerr=std_f1, capsize=4, color=[colors[m] for m in model_order], alpha=0.85)
        axes[1].set_xticks(np.arange(len(model_order)))
        axes[1].set_xticklabels([model_titles[m] for m in model_order], rotation=25, ha="right", fontsize=8)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("Macro-F1")
        axes[1].set_title("Mean macro-F1 across mice")
        axes[1].grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig2 = figures_dir / "02_ablation_mean_scores.png"
        fig.savefig(fig2, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig2}")

        # Figure 3: Confusion matrices for all 4 classifiers per mouse.
        # Layout: rows = mice, columns = 4 models.
        n_mice = len(mice_order)
        n_models = len(model_order)
        fig, axes = plt.subplots(
            n_mice,
            n_models,
            figsize=(4.4 * n_models, 3.8 * n_mice),
            squeeze=False,
        )

        for row_idx, mouse_name in enumerate(mice_order):
            payload = confusion_payload[mouse_name]
            labels_order = payload["labels"]
            best_model = payload["best_model"]
            for col_idx, mk in enumerate(model_order):
                ax = axes[row_idx][col_idx]
                cm = np.asarray(payload["cms"][mk], dtype=float)
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
                ConfusionMatrixDisplay(cm_norm, display_labels=labels_order).plot(
                    ax=ax,
                    cmap="Blues",
                    colorbar=False,
                    values_format=".2f",
                )
                title = f"{_short_mouse_name(mouse_name)}\n{model_titles[mk]}"
                if mk == best_model:
                    title += " ★"
                ax.set_title(title, fontsize=7)

        fig.suptitle("Normalized confusion matrices — all classifiers per mouse (★ = best)", fontsize=11)
        fig.tight_layout()
        fig3 = figures_dir / "03_all_classifier_confusion_matrices.png"
        fig.savefig(fig3, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig3}")

        summary_json_path = output_folder / "within_mouse_ablation_metrics.json"
        summary_csv_path = output_folder / "within_mouse_ablation_metrics.csv"

        payload = {
            "method": state.method,
            "p_active": state.p_active,
            "per_trial_thresh": state.per_trial_thresh,
            "zz_folder": state.zz_folder,
            "grid_subdir": state.grid_subdir,
            "mice": mice_order,
            "results": per_mouse,
            "models": model_order,
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
                    "mouse",
                    "method",
                    "model",
                    "input",
                    "n_trials",
                    "n_features",
                    "clip_frames",
                    "cv_folds",
                    "mean_acc",
                    "std_acc",
                    "mean_f1",
                    "std_f1",
                    "source",
                    "cache_path",
                    "best_model",
                ]
            )
            for mouse_name in mice_order:
                row = per_mouse[mouse_name]
                for mk in model_order:
                    mr = row["models"][mk]
                    writer.writerow(
                        [
                            mouse_name,
                            state.method,
                            mk,
                            "grid" if mk == "cnn3d" else "vector",
                            row["n_trials"],
                            row["n_features"],
                            row["clip_frames"],
                            row["cv_folds"],
                            mr["mean_acc"],
                            mr["std_acc"],
                            mr["mean_f1"],
                            mr["std_f1"],
                            row["source"],
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
            "Within-mouse ablation using selected zigzag vectorization (LogReg/MLP/1D-CNN) "
            "and 3D-CNN on raw grid activity."
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
    parser.add_argument("--n-splits", default=5, type=int)
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
        n_splits=args.n_splits,
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
