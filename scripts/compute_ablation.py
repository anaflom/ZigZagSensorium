#!/usr/bin/env python3
"""
Ablation study: Neural Network on Raw 3-D Grids vs TurnoverRate.

Runs all 4 models (LogReg, MLP, 1D-CNN, 3D-CNN) on all mice with 5-fold
stratified CV. Saves results to JSON for later plotting in the notebook.

Usage:
    python scripts/run_ablation.py              # all mice
    python scripts/run_ablation.py --mouse 0    # single mouse by index
    python scripts/run_ablation.py --mouse 0 1 2  # subset of mice
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

from utils import (
    build_vectorization_cache_stem,
    load_labelled_barcodes,
    load_vectorization_cache,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/"
                 "grid-15x15x10_norm-by_minmax")
GRID_ROOT = Path("/orfeo/scratch/area/ygardinazzi/sensorium_2026/derivatives/"
                 "grid-15x15x10_norm-by_minmax")
META_ROOT = Path("/u/mdmc/anaflom/projects_mdmc/sensorium/metadata")
CACHE_DIR = Path("/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium/notebooks/cache")
OUT_DIR   = Path("/u/mdmc/anaflom/projects_mdmc/ZigZagSensorium/results")

P_ACTIVE  = 30
ZZ_FOLDER = f"trials_zz-thresh-{P_ACTIVE}"
PER_TRIAL_THRESH = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_turnover_cache(mouse_name):
    cache_stem = build_vectorization_cache_stem(
        mouse_name=mouse_name,
        method="Turnover",
        p_active=P_ACTIVE,
        per_trial_thresh=PER_TRIAL_THRESH,
        clip_frames=None,
    )
    candidate_paths = [
        CACHE_DIR / f"{cache_stem}.npz",
        CACHE_DIR / f"turnover_{mouse_name}.npz",  # legacy notebook cache naming
    ]

    cache_file = next((p for p in candidate_paths if p.exists()), None)
    if cache_file is None:
        raise FileNotFoundError(
            "No turnover cache found. Expected one of: "
            + ", ".join(str(p) for p in candidate_paths)
        )

    data = load_vectorization_cache(cache_file)
    x_key = "X" if "X" in data else "features"
    if x_key not in data or "labels" not in data:
        raise KeyError(f"Cache missing required keys in {cache_file}")

    clip_raw = data.get("clip_frames", -1)
    clip_val = int(np.asarray(clip_raw).item())
    if clip_val < 0:
        raise ValueError(
            f"Cache {cache_file} has invalid clip_frames={clip_val}; regenerate cache with clipping."
        )
    return np.asarray(data[x_key]), np.asarray(data["labels"]), clip_val


def get_trial_info(mouse_name):
    _, labels_arr, trial_ids, frames_arr = load_labelled_barcodes(
        DATA_ROOT,
        META_ROOT,
        mouse_name,
        ZZ_FOLDER,
        max_trials=None,
    )
    trials = [
        (int(tid), str(label), int(frames))
        for tid, label, frames in zip(trial_ids, labels_arr, frames_arr)
    ]
    return trials


# ══════════════════════════════════════════════════════════════════════════════
#  Datasets
# ══════════════════════════════════════════════════════════════════════════════

class GridDataset(Dataset):
    """Lazily loads 4-D grid volumes (15x15x10xT) from disk per trial."""

    def __init__(self, mouse_name, trial_ids, labels, clip_frames):
        self.trial_ids = trial_ids
        self.clip_frames = clip_frames
        self.le = LabelEncoder().fit(sorted(set(labels)))
        self.labels = self.le.transform(labels)
        self.n_classes = len(self.le.classes_)

        grid_dir = GRID_ROOT / mouse_name / "trials"
        sample = next(grid_dir.glob("grid-*.npy"))
        self.template = str(grid_dir / sample.name.split("_trial-")[0]) + "_trial-{}.npy"

    def __len__(self):
        return len(self.trial_ids)

    def __getitem__(self, idx):
        tid = self.trial_ids[idx]
        grid = np.load(self.template.format(tid))  # (15, 15, 10, T)
        grid = grid[:, :, :, :self.clip_frames]
        grid = grid.transpose(2, 3, 0, 1)  # (10, T, 15, 15)
        return torch.tensor(grid, dtype=torch.float32), self.labels[idx]


class TurnoverDataset(Dataset):
    """Wraps pre-computed TurnoverRate feature vectors."""

    def __init__(self, X, labels):
        self.le = LabelEncoder().fit(sorted(set(labels)))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.labels = self.le.transform(labels)
        self.n_classes = len(self.le.classes_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Models
# ══════════════════════════════════════════════════════════════════════════════

class CNN3D(nn.Module):
    """3-D CNN on raw neural activity grids: (B, 10, T, 15, 15)."""

    def __init__(self, n_classes, clip_frames, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(10, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.bn1   = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.bn2   = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3   = nn.BatchNorm3d(128)
        self.pool  = nn.AdaptiveAvgPool3d(1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), ceil_mode=True)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), ceil_mode=True)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class CNN1D(nn.Module):
    """1-D CNN on TurnoverRate: (B, 720) -> (B, 3, clip)."""

    def __init__(self, n_classes, clip_frames, dropout=0.3):
        super().__init__()
        self.clip = clip_frames
        self.n_dim = 3
        self.conv1 = nn.Conv1d(3, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(64, n_classes)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.n_dim, self.clip)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2, ceil_mode=True)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2, ceil_mode=True)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


class MLP(nn.Module):
    """MLP on TurnoverRate vectors."""

    def __init__(self, n_classes, input_dim, dropout=0.3):
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

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  Training utilities
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = torch.tensor(y_batch, dtype=torch.long).to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(np.array(y_batch))
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1, preds, labels


def run_nn_cv(make_model_fn, dataset, labels_int, n_splits=5, epochs=40,
              lr=1e-3, batch_size=32, patience=8, num_workers=4):
    """Run stratified K-fold CV for a PyTorch model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs, fold_f1s = [], []
    pin = DEVICE.type == "cuda"

    for fold, (train_idx, val_idx) in enumerate(skf.split(
            np.zeros(len(labels_int)), labels_int)):
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin)
        val_loader   = DataLoader(Subset(dataset, val_idx),
                                  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin)

        model = make_model_fn().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_f1, best_acc, wait = 0.0, 0.0, 0
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                                criterion)
            val_acc, val_f1, _, _ = evaluate(model, val_loader)
            scheduler.step(1 - val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        fold_accs.append(best_acc)
        fold_f1s.append(best_f1)
        print(f"    Fold {fold+1}: acc={best_acc:.3f}  F1={best_f1:.3f}  "
              f"(stopped at epoch {epoch+1})")

    return {
        "acc": float(np.mean(fold_accs)), "acc_std": float(np.std(fold_accs)),
        "f1":  float(np.mean(fold_f1s)),  "f1_std":  float(np.std(fold_f1s)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Per-mouse ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_mouse(mname):
    """Run full ablation for one mouse, return dict of results."""
    short = mname.split("-")[0].replace("dynamic", "")
    print(f"\n{'='*70}")
    print(f"Mouse {short}: {mname}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}")

    # Load turnover cache
    X_turn, labels_str, clip_frames = load_turnover_cache(mname)
    le = LabelEncoder().fit(sorted(set(labels_str)))
    labels_int = le.transform(labels_str)
    nc = len(le.classes_)

    print(f"  Trials: {len(labels_str)}, Clip: {clip_frames}, Classes: {nc}")
    results = {"mouse": mname, "short": short, "n_trials": len(labels_str),
               "clip_frames": clip_frames, "n_classes": nc,
               "classes": list(le.classes_)}

    # ── LogReg ────────────────────────────────────────────────────────────
    print("\n  [LogReg on TurnoverRate]")
    t0 = time.perf_counter()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"acc": "accuracy", "f1": "f1_macro"}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42,
                                   class_weight="balanced")),
    ])
    cv_res = cross_validate(pipe, X_turn, labels_str, cv=cv, scoring=scoring)
    lr_res = {
        "acc": float(cv_res["test_acc"].mean()),
        "acc_std": float(cv_res["test_acc"].std()),
        "f1":  float(cv_res["test_f1"].mean()),
        "f1_std":  float(cv_res["test_f1"].std()),
        "time": time.perf_counter() - t0,
    }
    results["LogReg"] = lr_res
    print(f"    acc={lr_res['acc']:.3f}  F1={lr_res['f1']:.3f}  "
          f"({lr_res['time']:.1f}s)")

    # ── MLP on TurnoverRate ───────────────────────────────────────────────
    print("\n  [MLP on TurnoverRate]")
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(X_turn)
    turn_ds = TurnoverDataset(scaler.transform(X_turn), labels_str)
    mlp_res = run_nn_cv(
        lambda: MLP(nc, input_dim=X_turn.shape[1]),
        turn_ds, labels_int, epochs=60, lr=1e-3, batch_size=64,
        patience=10, num_workers=0)
    mlp_res["time"] = time.perf_counter() - t0
    results["MLP"] = mlp_res
    print(f"    acc={mlp_res['acc']:.3f}  F1={mlp_res['f1']:.3f}  "
          f"({mlp_res['time']:.1f}s)")

    # ── 1D-CNN on TurnoverRate ────────────────────────────────────────────
    print("\n  [1D-CNN on TurnoverRate]")
    t0 = time.perf_counter()
    cnn1d_res = run_nn_cv(
        lambda: CNN1D(nc, clip_frames=clip_frames),
        turn_ds, labels_int, epochs=60, lr=1e-3, batch_size=64,
        patience=10, num_workers=0)
    cnn1d_res["time"] = time.perf_counter() - t0
    results["CNN1D"] = cnn1d_res
    print(f"    acc={cnn1d_res['acc']:.3f}  F1={cnn1d_res['f1']:.3f}  "
          f"({cnn1d_res['time']:.1f}s)")

    # ── 3D-CNN on raw grids ──────────────────────────────────────────────
    print("\n  [3D-CNN on raw grids]")
    trial_info = get_trial_info(mname)
    tids = [t[0] for t in trial_info]
    tlabels = [t[1] for t in trial_info]

    grid_dir = GRID_ROOT / mname / "trials"
    if not grid_dir.exists():
        print(f"    SKIPPED — no grid data")
        cnn3d_res = {"acc": None, "f1": None, "acc_std": None, "f1_std": None,
                     "time": 0, "skipped": "no grid data"}
    elif len(tids) != len(labels_str):
        print(f"    SKIPPED — trial mismatch ({len(tids)} vs {len(labels_str)})")
        cnn3d_res = {"acc": None, "f1": None, "acc_std": None, "f1_std": None,
                     "time": 0, "skipped": "trial mismatch"}
    else:
        t0 = time.perf_counter()
        grid_ds = GridDataset(mname, tids, tlabels, clip_frames)
        nw = 4 if DEVICE.type == "cuda" else 0
        cnn3d_res = run_nn_cv(
            lambda: CNN3D(nc, clip_frames=clip_frames),
            grid_ds, labels_int, epochs=40, lr=5e-4, batch_size=16,
            patience=8, num_workers=nw)
        cnn3d_res["time"] = time.perf_counter() - t0
        print(f"    acc={cnn3d_res['acc']:.3f}  F1={cnn3d_res['f1']:.3f}  "
              f"({cnn3d_res['time']:.1f}s)")
    results["CNN3D"] = cnn3d_res

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ablation: NN vs TurnoverRate")
    parser.add_argument("--mouse", type=int, nargs="*", default=None,
                        help="Mouse index(es) to run. Default: all.")
    args = parser.parse_args()

    mice = sorted(
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("dynamic")
    )
    print(f"Found {len(mice)} mice total")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    if args.mouse is not None:
        selected = [mice[i] for i in args.mouse]
    else:
        selected = mice

    print(f"Running ablation on {len(selected)} mice\n")

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    all_results = {}

    for mname in selected:
        try:
            res = run_mouse(mname)
            all_results[mname] = res

            # Save incrementally
            out_file = OUT_DIR / "ablation_results.json"
            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  -> Saved to {out_file}")
        except Exception as e:
            print(f"\n  ERROR on {mname}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"{'Mouse':>8s} | {'LogReg F1':>10s} | {'MLP F1':>10s} | "
          f"{'1D-CNN F1':>10s} | {'3D-CNN F1':>10s}")
    print("-" * 62)
    for mname, res in all_results.items():
        s = res["short"]
        lr = res.get("LogReg", {}).get("f1")
        mlp = res.get("MLP", {}).get("f1")
        c1 = res.get("CNN1D", {}).get("f1")
        c3 = res.get("CNN3D", {}).get("f1")
        print(f"{s:>8s} | {_fmt(lr):>10s} | {_fmt(mlp):>10s} | "
              f"{_fmt(c1):>10s} | {_fmt(c3):>10s}")

    print("\nDone!")


def _fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"


if __name__ == "__main__":
    main()
