#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Within-mouse video-ID decoding per label with selectable vector/grid models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import traceback
from collections import Counter
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
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset

from classification_models import (
    CNN1D,
    CNN3D,
    MLP,
    GridTrialDataset,
    VectorDataset,
    infer_cnn1d_shape,
    run_logreg_cv,
    run_nn_cv,
)
from utils import (
    _build_zz_folder,
    _discover_mice,
    _eligible_trials,
    _opt_csv_list,
    _opt_int,
    _resolve_mouse_cache_dir,
    _short_mouse_name,
    _str2bool,
    load_labelled_barcodes,
    load_labelled_grid_paths,
    load_or_compute_vectorization_features,
    load_trial_metadata,
)


TARGET_LABELS: Tuple[str, ...] = (
    "NaturalVideo",
    "NaturalImages",
    "PinkNoise",
    "RandomDots",
    "Gabor",
    "GaussianDot",
)

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


@dataclass
class RunState:
    output_folder: Path
    data_root: Path
    meta_root: Path
    p_active: int
    per_trial_thresh: bool
    zz_folder: str
    vectorization_method: str
    mice: Optional[List[str]]
    models: List[str]
    clip_frames: Optional[int]
    grid_subdir: str
    cache_dir: Optional[Path]
    max_trials: Optional[int]
    min_id_repetitions: int
    cv_scheme: str
    cv_n_splits: int
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


def _normalize_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if text.endswith(".0"):
        try:
            as_float = float(text)
            if as_float.is_integer():
                return str(int(as_float))
        except ValueError:
            pass
    return text


def _safe_token(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _build_cv_splits(
    n_samples: int,
    *,
    scheme: str,
    n_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    scheme_norm = str(scheme).strip().lower()
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for CV, got {n_samples}")

    if scheme_norm == "loso":
        idx_all = np.arange(n_samples, dtype=np.int64)
        return [
            (idx_all[idx_all != i], np.asarray([i], dtype=np.int64))
            for i in range(n_samples)
        ]

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


def _plot_mouse_figures(
    *,
    mouse_name: str,
    per_label_results: Dict[str, Dict[str, Any]],
    model_order: Sequence[str],
    figures_dir: Path,
) -> Optional[Path]:
    labels = [lab for lab in TARGET_LABELS if lab in per_label_results]
    if len(labels) == 0:
        return None

    x = np.arange(len(labels))
    width = min(0.82 / len(model_order), 0.22)

    fig, (ax_acc, ax_f1) = plt.subplots(2, 1, figsize=(max(10, len(labels) * 2.0), 8.2), sharex=True)
    chance = [float(per_label_results[l]["chance_level"]) for l in labels]

    for idx, mk in enumerate(model_order):
        offset = (idx - (len(model_order) - 1) / 2.0) * width
        acc_vals = [float(per_label_results[l][mk]["accuracy"]) for l in labels]
        f1_vals = [float(per_label_results[l][mk]["macro_f1"]) for l in labels]
        ax_acc.bar(x + offset, acc_vals, width=width, alpha=0.88, color=MODEL_COLORS[mk], label=MODEL_TITLES[mk])
        ax_f1.bar(x + offset, f1_vals, width=width, alpha=0.88, color=MODEL_COLORS[mk], label=MODEL_TITLES[mk])

    ax_acc.plot(x, chance, "k--", linewidth=1.3, marker="o", markersize=4, label="Chance")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{_short_mouse_name(mouse_name)} - video-ID decoding (accuracy)")
    ax_acc.grid(axis="y", alpha=0.25)
    ax_acc.legend(loc="upper right", fontsize=8)

    ax_f1.plot(x, chance, "k--", linewidth=1.3, marker="o", markersize=4, label="Chance")
    ax_f1.set_ylim(0.0, 1.05)
    ax_f1.set_ylabel("Macro-F1")
    ax_f1.set_title(f"{_short_mouse_name(mouse_name)} - video-ID decoding (macro-F1)")
    ax_f1.grid(axis="y", alpha=0.25)
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(labels, rotation=25, ha="right")

    fig.tight_layout()
    out_path = figures_dir / f"mouse_{_safe_token(mouse_name)}_video_id_scores.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_pipeline(state: RunState) -> Dict[str, Any]:
    _set_seed(state.seed)
    device = _resolve_device(state.device)

    state.output_folder.mkdir(parents=True, exist_ok=True)
    figures_dir = state.output_folder / "figures"
    logs_dir = state.output_folder / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_order = list(state.models)
    log_path = logs_dir / "run.log"

    with open(log_path, "w", encoding="utf-8") as log_fp, redirect_stdout(log_fp), redirect_stderr(log_fp):
        print("=" * 90)
        print("Within-Mouse Video-ID Decoding: selectable vector/grid models")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)
        print(f"models={model_order}")

        discovered_mice = _discover_mice(state.data_root)
        selected_mice = state.mice if state.mice is not None else discovered_mice
        selected_mice = [m for m in selected_mice if m in discovered_mice]
        if not selected_mice:
            raise RuntimeError("No valid mice selected for within-mouse video-ID decoding")

        summary_rows: List[Dict[str, Any]] = []
        skip_rows: List[Dict[str, Any]] = []
        summary_payload: Dict[str, Any] = {
            "method": state.vectorization_method,
            "models": model_order,
            "target_labels": list(TARGET_LABELS),
            "results": {},
            "figures": [],
            "log_path": str(log_path),
            "device": str(device),
        }
        confusion_payload: Dict[str, Any] = {}
        prediction_payload: Dict[str, Any] = {}

        for mouse_name in selected_mice:
            print(f"\n## Mouse: {mouse_name}")
            df_trials = load_trial_metadata(state.meta_root, mouse_name)
            df_eligible = _eligible_trials(df_trials)
            if "video_ID" not in df_eligible.columns:
                skip_rows.append({"mouse": mouse_name, "label": "ALL", "reason": "missing_video_ID", "detail": ""})
                continue

            trial_to_label: Dict[int, str] = {}
            trial_to_id: Dict[int, str] = {}
            for row in df_eligible.itertuples(index=False):
                tid = int(getattr(row, "trial"))
                vid = _normalize_id(getattr(row, "video_ID"))
                if vid is None:
                    continue
                trial_to_label[tid] = str(getattr(row, "label"))
                trial_to_id[tid] = vid

            barcodes, labels_raw, trial_ids, valid_frames = load_labelled_barcodes(
                state.data_root,
                state.meta_root,
                mouse_name,
                state.zz_folder,
                max_trials=state.max_trials,
            )
            if len(barcodes) == 0:
                skip_rows.append({"mouse": mouse_name, "label": "ALL", "reason": "no_labelled_barcodes", "detail": ""})
                continue

            clip_arg = state.clip_frames
            vec_trial_ids = [int(t) for t in trial_ids]
            xmat, vec_source, cache_path = load_or_compute_vectorization_features(
                data_root=state.data_root,
                mouse_name=mouse_name,
                method=state.vectorization_method,
                p_active=state.p_active,
                per_trial_thresh=state.per_trial_thresh,
                clip_frames=clip_arg,
                barcodes=barcodes,
                labels=labels_raw,
                trial_ids=trial_ids,
                valid_frames=valid_frames,
                cache_dir=_resolve_mouse_cache_dir(state, mouse_name),
                expected_trial_ids=vec_trial_ids,
                message_prefix="  ",
            )
            if state.clip_frames is None:
                clip_used = int(xmat.shape[1] // 3) if state.vectorization_method == "Turnover" and xmat.shape[1] % 3 == 0 else int(xmat.shape[1])
            else:
                clip_used = int(state.clip_frames)

            grid_paths, grid_labels, grid_trial_ids, grid_valid_frames = load_labelled_grid_paths(
                state.data_root,
                state.meta_root,
                mouse_name,
                grid_subdir=state.grid_subdir,
            )
            if len(grid_paths) == 0:
                skip_rows.append({"mouse": mouse_name, "label": "ALL", "reason": "no_grid_paths", "detail": state.grid_subdir})
                continue

            vec_idx = {int(tid): i for i, tid in enumerate(vec_trial_ids)}
            grid_idx = {int(tid): i for i, tid in enumerate(grid_trial_ids)}
            common_trials = [
                tid
                for tid in vec_trial_ids
                if tid in grid_idx and tid in trial_to_label and tid in trial_to_id
            ]
            if len(common_trials) == 0:
                skip_rows.append({"mouse": mouse_name, "label": "ALL", "reason": "no_aligned_trials", "detail": ""})
                continue

            vec_take = np.asarray([vec_idx[t] for t in common_trials], dtype=np.int64)
            grid_take = np.asarray([grid_idx[t] for t in common_trials], dtype=np.int64)
            labels_meta = np.asarray([trial_to_label[t] for t in common_trials])
            ids_meta = np.asarray([trial_to_id[t] for t in common_trials])
            trials_meta = np.asarray(common_trials, dtype=np.int64)

            x_vec = np.asarray(xmat[vec_take], dtype=np.float64)
            grid_paths_sel = [grid_paths[i] for i in grid_take]
            grid_frames_sel = np.asarray(grid_valid_frames[grid_take], dtype=np.int64)

            per_label_results: Dict[str, Dict[str, Any]] = {}
            per_label_confusions: Dict[str, Any] = {}
            per_label_predictions: Dict[str, Any] = {}

            for label_name in TARGET_LABELS:
                label_mask = labels_meta == label_name
                if not np.any(label_mask):
                    skip_rows.append({"mouse": mouse_name, "label": label_name, "reason": "missing_label", "detail": ""})
                    continue

                ids_label = ids_meta[label_mask]
                id_counts = Counter(ids_label.tolist())
                keep_ids = sorted([k for k, v in id_counts.items() if int(v) >= int(state.min_id_repetitions)])
                if len(keep_ids) < 2:
                    skip_rows.append({
                        "mouse": mouse_name,
                        "label": label_name,
                        "reason": "insufficient_repeated_ids",
                        "detail": str(len(keep_ids)),
                    })
                    continue

                id_mask = np.isin(ids_label, np.asarray(keep_ids))
                trial_ids_label = trials_meta[label_mask][id_mask]
                y_tokens = ids_label[id_mask]
                x_label = x_vec[label_mask][id_mask]
                grid_paths_label = np.asarray(grid_paths_sel, dtype=object)[label_mask][id_mask].tolist()
                grid_frames_label = grid_frames_sel[label_mask][id_mask]

                if len(y_tokens) < 2:
                    skip_rows.append({"mouse": mouse_name, "label": label_name, "reason": "insufficient_samples", "detail": str(len(y_tokens))})
                    continue

                le = LabelEncoder().fit(keep_ids)
                y_int = le.transform(y_tokens)
                n_classes = int(len(le.classes_))
                if n_classes < 2:
                    skip_rows.append({"mouse": mouse_name, "label": label_name, "reason": "insufficient_classes", "detail": str(n_classes)})
                    continue

                try:
                    splits = _build_cv_splits(
                        len(y_int),
                        scheme=state.cv_scheme,
                        n_splits=state.cv_n_splits,
                        seed=state.seed,
                        groups=np.asarray(trial_ids_label, dtype=np.int64),
                    )
                except Exception as exc:
                    skip_rows.append({"mouse": mouse_name, "label": label_name, "reason": "cv_split_error", "detail": str(exc)})
                    continue

                vec_ds = VectorDataset(x_label, y_int)
                need_grid_raw = "cnn3d_raw" in model_order
                need_grid_norm = "cnn3d_norm" in model_order
                grid_ds_raw = None
                grid_ds_norm = None
                if need_grid_raw:
                    grid_ds_raw = GridTrialDataset(
                        grid_paths=grid_paths_label,
                        y=y_int,
                        valid_frames=grid_frames_label,
                        clip_frames=int(clip_used),
                        normalize_by_trial=False,
                    )
                if need_grid_norm:
                    grid_ds_norm = GridTrialDataset(
                        grid_paths=grid_paths_label,
                        y=y_int,
                        valid_frames=grid_frames_label,
                        clip_frames=int(clip_used),
                        normalize_by_trial=True,
                    )

                model_metrics: Dict[str, Dict[str, float]] = {}
                model_preds: Dict[str, np.ndarray] = {}
                for mk in model_order:
                    if mk == "logreg":
                        mm, pred = run_logreg_cv(x_label, y_int, splits)
                        pred_use = pred
                    elif mk == "mlp":
                        mm, pred = run_nn_cv(
                            make_model=lambda n_classes=n_classes, dim=int(x_label.shape[1]): MLP(n_classes=n_classes, input_dim=dim),
                            train_dataset_builder=lambda tr, va, ds=vec_ds: (Subset(ds, tr), Subset(ds, va)),
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
                        pred_use = pred
                    elif mk == "cnn1d":
                        in_ch, seq_len = infer_cnn1d_shape(int(x_label.shape[1]), int(clip_used))
                        mm, pred = run_nn_cv(
                            make_model=lambda n_classes=n_classes, ic=in_ch, sl=seq_len: CNN1D(n_classes=n_classes, in_channels=ic, seq_len=sl),
                            train_dataset_builder=lambda tr, va, ds=vec_ds: (Subset(ds, tr), Subset(ds, va)),
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
                        pred_use = pred
                    elif mk == "cnn3d_raw":
                        if grid_ds_raw is None:
                            raise RuntimeError("cnn3d_raw selected but raw grid dataset unavailable")
                        mm, pred = run_nn_cv(
                            make_model=lambda n_classes=n_classes, in_ch=int(grid_ds_raw.in_channels): CNN3D(n_classes=n_classes, in_channels=in_ch),
                            train_dataset_builder=lambda tr, va, ds=grid_ds_raw: (Subset(ds, tr), Subset(ds, va)),
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
                        pred_use = pred
                    elif mk == "cnn3d_norm":
                        if grid_ds_norm is None:
                            raise RuntimeError("cnn3d_norm selected but normalized grid dataset unavailable")
                        mm, pred = run_nn_cv(
                            make_model=lambda n_classes=n_classes, in_ch=int(grid_ds_norm.in_channels): CNN3D(n_classes=n_classes, in_channels=in_ch),
                            train_dataset_builder=lambda tr, va, ds=grid_ds_norm: (Subset(ds, tr), Subset(ds, va)),
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
                        pred_use = pred
                    else:
                        raise ValueError(f"Unsupported model key: {mk}")

                    if "mean_acc" in mm:
                        model_metrics[mk] = {
                            "accuracy": float(mm["mean_acc"]),
                            "macro_f1": float(mm["mean_f1"]),
                            "std_acc": float(mm["std_acc"]),
                            "std_f1": float(mm["std_f1"]),
                        }
                    else:
                        model_metrics[mk] = {
                            "accuracy": float(mm["accuracy"]),
                            "macro_f1": float(mm["macro_f1"]),
                            "std_acc": 0.0,
                            "std_f1": 0.0,
                        }
                    model_preds[mk] = np.asarray(pred_use, dtype=np.int64)

                chance = 1.0 / float(n_classes)
                cms = {
                    mk: confusion_matrix(y_int, model_preds[mk], labels=np.arange(n_classes))
                    for mk in model_order
                }

                per_label_results[label_name] = {
                    "n_samples": int(len(y_int)),
                    "n_classes": int(n_classes),
                    "class_labels": [str(v) for v in le.classes_],
                    "chance_level": float(chance),
                    "cv": {"scheme": state.cv_scheme, "n_folds": int(len(splits))},
                    **model_metrics,
                }
                per_label_confusions[label_name] = {
                    "labels": [str(v) for v in le.classes_],
                    "cms_counts": {mk: cms[mk].tolist() for mk in model_order},
                    "cms_normalized": {
                        mk: np.divide(
                            cms[mk],
                            cms[mk].sum(axis=1, keepdims=True),
                            out=np.zeros_like(cms[mk], dtype=float),
                            where=cms[mk].sum(axis=1, keepdims=True) != 0,
                        ).tolist()
                        for mk in model_order
                    },
                }
                per_label_predictions[label_name] = {
                    "labels": [str(v) for v in le.classes_],
                    "y_true": [int(v) for v in y_int.tolist()],
                    "predictions": {mk: [int(v) for v in model_preds[mk].tolist()] for mk in model_order},
                    "trial_ids": [int(v) for v in np.asarray(trial_ids_label).tolist()],
                    "id_tokens": [str(v) for v in y_tokens.tolist()],
                }

                for mk in model_order:
                    summary_rows.append(
                        {
                            "mouse": mouse_name,
                            "label": label_name,
                            "model": mk,
                            "input": "grid" if mk.startswith("cnn3d") else "vector",
                            "method": state.vectorization_method,
                            "source": vec_source,
                            "cache_path": str(cache_path),
                            "clip_frames": int(clip_used),
                            "n_samples": int(len(y_int)),
                            "n_classes": int(n_classes),
                            "cv_scheme": state.cv_scheme,
                            "cv_folds": int(len(splits)),
                            "chance_level": float(chance),
                            "accuracy": float(model_metrics[mk]["accuracy"]),
                            "macro_f1": float(model_metrics[mk]["macro_f1"]),
                            "std_acc": float(model_metrics[mk]["std_acc"]),
                            "std_f1": float(model_metrics[mk]["std_f1"]),
                        }
                    )

            if per_label_results:
                fig = _plot_mouse_figures(
                    mouse_name=mouse_name,
                    per_label_results=per_label_results,
                    model_order=model_order,
                    figures_dir=figures_dir,
                )
                if fig is not None:
                    summary_payload["figures"].append(str(fig))
                summary_payload["results"][mouse_name] = {
                    "source": vec_source,
                    "cache_path": str(cache_path),
                    "clip_frames": int(clip_used),
                    "labels": per_label_results,
                }
                confusion_payload[mouse_name] = per_label_confusions
                prediction_payload[mouse_name] = per_label_predictions

        if not summary_rows:
            raise RuntimeError("No within-mouse video-ID results were produced")

        summary_csv_path = state.output_folder / "within_mouse_video_id_metrics.csv"
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
                    "n_samples",
                    "n_classes",
                    "cv_scheme",
                    "cv_folds",
                    "chance_level",
                    "accuracy",
                    "macro_f1",
                    "std_acc",
                    "std_f1",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        skip_csv_path = state.output_folder / "within_mouse_video_id_skips.csv"
        with open(skip_csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["mouse", "label", "reason", "detail"])
            writer.writeheader()
            for row in skip_rows:
                writer.writerow(row)

        confusion_json_path = state.output_folder / "within_mouse_video_id_confusion_matrices.json"
        with open(confusion_json_path, "w", encoding="utf-8") as fp:
            json.dump(confusion_payload, fp, indent=2)

        predictions_json_path = state.output_folder / "within_mouse_video_id_prediction_outputs.json"
        with open(predictions_json_path, "w", encoding="utf-8") as fp:
            json.dump(prediction_payload, fp, indent=2)

        summary_json_path = state.output_folder / "within_mouse_video_id_metrics.json"
        summary_payload["summary_csv_path"] = str(summary_csv_path)
        summary_payload["skip_csv_path"] = str(skip_csv_path)
        summary_payload["confusion_json_path"] = str(confusion_json_path)
        summary_payload["predictions_json_path"] = str(predictions_json_path)
        with open(summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(summary_payload, fp, indent=2)

    return {
        "log_path": str(log_path),
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "skip_csv_path": str(skip_csv_path),
        "confusion_json_path": str(confusion_json_path),
        "predictions_json_path": str(predictions_json_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Within-mouse video-ID decoding by label with selectable vector/grid models."
    )
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)

    parser.add_argument("--p-active", required=True, type=int)
    parser.add_argument("--per-trial-thresh", required=True, type=_str2bool)
    parser.add_argument("--vectorization-method", required=True, type=str)

    parser.add_argument("--mice", default=None, type=_opt_csv_list)
    parser.add_argument("--models", default=None, type=_opt_csv_list)
    parser.add_argument("--clip-frames", default=None, type=_opt_int)
    parser.add_argument("--grid-subdir", default="trials_grid")
    parser.add_argument("--cache-dir", default=None, type=Path)
    parser.add_argument("--max-trials", default=None, type=_opt_int)

    parser.add_argument("--min-id-repetitions", default=7, type=int)
    parser.add_argument("--cv-scheme", default="logo", choices=["loso", "logo", "groupkfold", "kfold"])
    parser.add_argument("--cv-n-splits", default=5, type=int)

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

    if int(args.min_id_repetitions) < 1:
        print("ERROR: --min-id-repetitions must be >= 1", file=sys.stderr)
        return 2

    try:
        model_order = _resolve_models(args.models)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    state = RunState(
        output_folder=args.output_folder,
        data_root=args.data_root,
        meta_root=args.meta_root,
        p_active=args.p_active,
        per_trial_thresh=args.per_trial_thresh,
        zz_folder=_build_zz_folder(args.p_active, args.per_trial_thresh),
        vectorization_method=args.vectorization_method,
        mice=args.mice,
        models=model_order,
        clip_frames=args.clip_frames,
        grid_subdir=args.grid_subdir,
        cache_dir=args.cache_dir,
        max_trials=args.max_trials,
        min_id_repetitions=args.min_id_repetitions,
        cv_scheme=args.cv_scheme,
        cv_n_splits=args.cv_n_splits,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
