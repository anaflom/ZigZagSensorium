#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared classification models and training utilities."""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset


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
    """Lazy grid loader from file paths."""

    def __init__(
        self,
        grid_paths: Sequence,
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


def _infer_cnn1d_shape(feat_dim: int, clip_frames: int) -> Tuple[int, int]:
    """Infer (channels, seq_len) for 1D CNN from feature dimension and clip frames."""
    if clip_frames > 0 and feat_dim % clip_frames == 0:
        return int(feat_dim // clip_frames), int(clip_frames)
    return 1, int(feat_dim)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    """Train for one epoch."""
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
    """Generate predictions from a model."""
    model.eval()
    preds: List[np.ndarray] = []
    for xb, _yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _build_train_val_indices(y_train: np.ndarray, seed: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build train/val split indices for validation during training."""
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
    make_model: Callable,
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
    """Train and evaluate a neural network on train/test split."""
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
    """Train and evaluate logistic regression on train/test split."""
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


def _run_nn_cv(
    make_model: Callable,
    train_dataset_builder: Callable,
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
    """Run k-fold cross-validation for a neural network."""
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

        best_state = None
        best_f1 = -1.0
        wait = 0

        for _epoch in range(int(epochs)):
            _train_epoch(model, train_loader, optimizer, criterion, device)
            val_pred = _predict(model, val_loader, device)
            val_y = np.asarray([val_ds[i][1] for i in range(len(val_ds))], dtype=np.int64)
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

        if best_state is not None:
            model.load_state_dict(best_state)

        fold_pred = _predict(model, val_loader, device)
        oof_pred[val_idx] = fold_pred
        val_y = np.asarray([val_ds[i][1] for i in range(len(val_ds))], dtype=np.int64)
        fold_acc.append(float(accuracy_score(val_y, fold_pred)))
        fold_f1.append(float(f1_score(val_y, fold_pred, average="macro", zero_division=0)))

    metrics = {
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
    }
    return metrics, oof_pred


def _run_logreg_cv(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[str, float], np.ndarray]:
    """Run k-fold cross-validation for logistic regression."""
    fold_acc: List[float] = []
    fold_f1: List[float] = []
    oof_pred = np.full_like(y, fill_value=-1, dtype=np.int64)

    for train_idx, val_idx in splits:
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

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
        fold_pred = pipe.predict(x_val)
        oof_pred[val_idx] = fold_pred

        fold_acc.append(float(accuracy_score(y_val, fold_pred)))
        fold_f1.append(float(f1_score(y_val, fold_pred, average="macro", zero_division=0)))

    metrics = {
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
    }
    return metrics, oof_pred
