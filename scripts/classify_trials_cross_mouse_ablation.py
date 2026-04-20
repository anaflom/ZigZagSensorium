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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from classification_models import (
    CNN1D,
    CNN3D,
    MLP,
    GridTrialDataset,
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
    _str2bool,
    load_labelled_barcodes,
    load_labelled_grid_paths,
    load_or_compute_vectorization_features,
    _short_mouse_name,
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

    vec_trial_ids = [int(t) for t in trial_ids]
    xmat, vec_source, cache_path = load_or_compute_vectorization_features(
        data_root=state.data_root,
        mouse_name=mouse_name,
        method=state.method,
        p_active=state.p_active,
        per_trial_thresh=state.per_trial_thresh,
        clip_frames=int(global_clip),
        barcodes=barcodes,
        labels=labels,
        trial_ids=trial_ids,
        valid_frames=valid_frames,
        cache_dir=_resolve_mouse_cache_dir(state, mouse_name),
        force_recompute=state.force_recompute,
        expected_trial_ids=vec_trial_ids,
        message_prefix=f"  [{mouse_name}] ",
    )

    grid_paths, grid_labels, grid_trial_ids, grid_valid_frames = load_labelled_grid_paths(
        state.data_root,
        state.meta_root,
        mouse_name,
        grid_subdir=state.grid_subdir,
    )
    if len(grid_paths) == 0:
        return None

    vec_tid_to_i = {int(tid): i for i, tid in enumerate(vec_trial_ids)}
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
        prediction_payload: Dict[str, Dict[str, Any]] = {}

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
                cnn1d_channels, cnn1d_seq_len = infer_cnn1d_shape(train_x_vec.shape[1], global_clip)
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

                logreg_metrics, logreg_pred = train_eval_logreg(
                    x_train_scaled,
                    train_y,
                    x_test_scaled,
                    test_y,
                )

                mlp_metrics, mlp_pred = train_eval_nn(
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

                cnn1d_metrics, cnn1d_pred = train_eval_nn(
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
                cnn3d_metrics, cnn3d_pred = train_eval_nn(
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
                prediction_payload[test_mouse] = {
                    "labels": [str(v) for v in shared_labels],
                    "y_true": [int(v) for v in test_y.tolist()],
                    "predictions": {
                        mk: [int(v) for v in np.asarray(model_preds[mk]).tolist()]
                        for mk in model_order
                    },
                    "best_model": best_model,
                    "test_mouse": test_mouse,
                    "train_mice": active_train_mice,
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

        fig, (ax_f1, ax_acc) = plt.subplots(2, 1, figsize=(max(9, len(fold_order) * 1.3), 9.0), sharex=True)
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
            vals_f1 = [float(per_fold[m]["models"][mk]["macro_f1"]) for m in fold_order]
            vals_acc = [float(per_fold[m]["models"][mk]["accuracy"]) for m in fold_order]
            ax_f1.bar(x + offsets[mk], vals_f1, width, label=model_titles[mk], alpha=0.85, color=colors[mk])
            ax_acc.bar(x + offsets[mk], vals_acc, width, label=model_titles[mk], alpha=0.85, color=colors[mk])
        ax_f1.set_ylim(0, 1.05)
        ax_f1.set_ylabel("Macro-F1")
        ax_f1.set_title(f"Cross-mouse LOMO macro-F1 by test mouse ({state.method})")
        ax_f1.grid(axis="y", alpha=0.25)
        ax_f1.legend(loc="upper right", ncol=2, fontsize=8)
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels([_short_mouse_name(m) for m in fold_order], rotation=30, ha="right", fontsize=8)
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"Cross-mouse LOMO accuracy by test mouse ({state.method})")
        ax_acc.grid(axis="y", alpha=0.25)
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
                title = f"{_short_mouse_name(mouse_name)}\n{model_titles[mk]}"
                if mk == best_model:
                    title += " ★"
                ax.set_title(title, fontsize=7)
        fig.suptitle("Normalized confusion matrices — all classifiers per test mouse (★ = best)", fontsize=11)
        fig.tight_layout()
        fig3 = figures_dir / "03_all_classifier_confusion_matrices.png"
        fig.savefig(fig3, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {fig3}")

        confusion_json_path = output_folder / "cross_mouse_confusion_matrices.json"
        confusion_json_payload: Dict[str, Dict[str, Any]] = {}
        for mouse_name in fold_order:
            payload = confusion_payload[mouse_name]
            labels_order = [str(v) for v in payload["labels"]]
            cms_counts: Dict[str, List[List[int]]] = {}
            cms_norm: Dict[str, List[List[float]]] = {}
            for mk in model_order:
                cm = np.asarray(payload["cms"][mk], dtype=np.int64)
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
                cms_counts[mk] = cm.tolist()
                cms_norm[mk] = cm_norm.tolist()
            confusion_json_payload[mouse_name] = {
                "labels": labels_order,
                "best_model": payload["best_model"],
                "cms_counts": cms_counts,
                "cms_normalized": cms_norm,
            }
        with open(confusion_json_path, "w", encoding="utf-8") as fp:
            json.dump(confusion_json_payload, fp, indent=2)
        print(f"Wrote confusion JSON: {confusion_json_path}")

        predictions_json_path = output_folder / "cross_mouse_prediction_outputs.json"
        predictions_json_payload = {
            mouse_name: prediction_payload[mouse_name]
            for mouse_name in fold_order
        }
        with open(predictions_json_path, "w", encoding="utf-8") as fp:
            json.dump(predictions_json_payload, fp, indent=2)
        print(f"Wrote prediction JSON: {predictions_json_path}")

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
            "confusion_matrices_path": str(confusion_json_path),
            "prediction_outputs_path": str(predictions_json_path),
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
        "confusion_json_path": str(confusion_json_path),
        "predictions_json_path": str(predictions_json_path),
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