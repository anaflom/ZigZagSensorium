#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-mouse video ID decoding (train on one mouse, test on another).

This script builds directed mouse-pair experiments (A->B and B->A) and runs
ID classification per label class using:
1. Zigzag vectorization + LogReg / MLP / 1D-CNN
2. Grid activations + 3D-CNN raw / 3D-CNN normalized

Pair eligibility is based on metadata at:
  <meta_root>/<mouse_name>/trials/meta-trials_<mouse_name>.csv
using only trials where valid_trial and valid_response are true. A mouse pair
is kept when both mice have common ID values repeated at least
--min-id-repetitions times.

Outputs follow the schema used by other classification scripts:
- cross_mouse_id_decoding_metrics.json
- cross_mouse_id_decoding_metrics.csv
- cross_mouse_id_decoding_confusion_matrices.json
- cross_mouse_id_decoding_prediction_outputs.json
- figures/
- logs/run.log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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
MODEL_CHOICES = ["logreg", "mlp", "cnn1d", "cnn3d_raw", "cnn3d_norm"]
DEFAULT_MODELS = ["logreg", "mlp", "cnn1d", "cnn3d_raw", "cnn3d_norm"]
MODEL_TITLES = {
    "logreg": "LogReg (vector)",
    "mlp": "MLP (vector)",
    "cnn1d": "1D-CNN (vector)",
    "cnn3d_raw": "3D-CNN (raw grid)",
    "cnn3d_norm": "3D-CNN (normalized grid)",
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
    batch_size_vec: int
    epochs_mlp: int
    epochs_cnn1d: int
    lr_vec: float
    batch_size_grid: int
    epochs_cnn3d: int
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


def _stable_seed(base_seed: int, token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    delta = int(digest[:8], 16)
    return int((base_seed + delta) % (2**31 - 1))


def _filter_to_repeated_eligible(df_trials: Any, min_id_repetitions: int) -> Any:
    df_eligible = _eligible_trials(df_trials)
    if "video_ID" not in df_eligible.columns:
        raise ValueError("Metadata must include an video_ID column")

    out = df_eligible.copy()
    out["id_norm"] = out["video_ID"].map(_normalize_id)
    out = out.loc[out["id_norm"].notna()].copy()
    if out.empty:
        return out

    counts = out["id_norm"].value_counts()
    keep_ids = set(counts[counts >= int(min_id_repetitions)].index.tolist())
    return out.loc[out["id_norm"].isin(keep_ids)].copy()


def _select_candidate_pairs(
    meta_root: Path,
    selected_mice: Sequence[str],
    min_id_repetitions: int,
) -> Tuple[List[Tuple[str, str]], Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    id_set_by_mouse: Dict[str, Set[str]] = {}

    for mouse_name in selected_mice:
        df = load_trial_metadata(meta_root, mouse_name)
        df_rep = _filter_to_repeated_eligible(df, min_id_repetitions)
        id_set_by_mouse[mouse_name] = set(df_rep["id_norm"].astype(str).unique().tolist())

    pairs: List[Tuple[str, str]] = []
    pair_common_ids: Dict[Tuple[str, str], Set[str]] = {}

    for m1, m2 in combinations(sorted(selected_mice), 2):
        common_ids = id_set_by_mouse.get(m1, set()) & id_set_by_mouse.get(m2, set())
        if len(common_ids) == 0:
            continue
        pairs.append((m1, m2))
        pair_common_ids[(m1, m2)] = common_ids

    return pairs, id_set_by_mouse, pair_common_ids


def _load_mouse_data(
    state: RunState,
    mouse_name: str,
    global_clip: int,
) -> Optional[Dict[str, Any]]:
    df_trials = load_trial_metadata(state.meta_root, mouse_name)
    df_rep = _filter_to_repeated_eligible(df_trials, state.min_id_repetitions)
    if df_rep.empty:
        return None

    if "trial" not in df_rep.columns or "label" not in df_rep.columns:
        raise ValueError(f"Metadata for {mouse_name} must include trial and label columns")

    trial_to_label: Dict[int, str] = {}
    trial_to_id: Dict[int, str] = {}
    for row in df_rep.itertuples(index=False):
        tid = int(getattr(row, "trial"))
        lid = _normalize_id(getattr(row, "video_ID"))
        if lid is None:
            continue
        trial_to_label[tid] = str(getattr(row, "label"))
        trial_to_id[tid] = lid

    barcodes, labels_raw, trial_ids, valid_frames = load_labelled_barcodes(
        state.data_root,
        state.meta_root,
        mouse_name,
        state.zz_folder,
        max_trials=state.max_trials,
    )
    if len(barcodes) == 0:
        return None

    vec_trial_ids = [int(t) for t in trial_ids]
    xmat, vec_source, cache_path = load_or_compute_vectorization_features(
        data_root=state.data_root,
        mouse_name=mouse_name,
        method=state.vectorization_method,
        p_active=state.p_active,
        per_trial_thresh=state.per_trial_thresh,
        clip_frames=int(global_clip),
        barcodes=barcodes,
        labels=labels_raw,
        trial_ids=trial_ids,
        valid_frames=valid_frames,
        cache_dir=_resolve_mouse_cache_dir(state, mouse_name),
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

    vec_idx = {int(tid): i for i, tid in enumerate(vec_trial_ids)}
    grid_idx = {int(tid): i for i, tid in enumerate(grid_trial_ids)}

    common_trials = [
        tid for tid in vec_trial_ids if tid in grid_idx and tid in trial_to_label and tid in trial_to_id
    ]
    if len(common_trials) == 0:
        return None

    vec_take = np.array([vec_idx[tid] for tid in common_trials], dtype=np.int64)
    grid_take = np.array([grid_idx[tid] for tid in common_trials], dtype=np.int64)

    labels_meta = np.asarray([trial_to_label[tid] for tid in common_trials])
    ids_meta = np.asarray([trial_to_id[tid] for tid in common_trials])

    labels_grid = np.asarray(grid_labels[grid_take])
    if not np.all(labels_meta == labels_grid):
        raise RuntimeError(f"Label mismatch between metadata and grid labels for mouse {mouse_name}")

    x_vec = np.asarray(xmat[vec_take], dtype=np.float64)
    grid_paths_common = [grid_paths[i] for i in grid_take]
    grid_frames_common = np.asarray(grid_valid_frames[grid_take], dtype=np.int64)

    return {
        "mouse": mouse_name,
        "trial_ids": np.asarray(common_trials, dtype=np.int64),
        "labels": labels_meta,
        "id_tokens": ids_meta,
        "x_vec": x_vec,
        "grid_paths": grid_paths_common,
        "grid_frames": grid_frames_common,
        "cache_path": str(cache_path),
        "vec_source": vec_source,
        "id_set": set(ids_meta.tolist()),
    }


def _evaluate_direction_label(
    *,
    source_mouse: str,
    target_mouse: str,
    label_name: str,
    source_data: Dict[str, Any],
    target_data: Dict[str, Any],
    pair_common_ids: Set[str],
    min_id_repetitions: int,
    clip_frames: int,
    state: RunState,
    device: torch.device,
    model_order: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    src_label_mask = source_data["labels"] == label_name
    tgt_label_mask = target_data["labels"] == label_name

    if not np.any(src_label_mask) or not np.any(tgt_label_mask):
        return None, None, None, "label missing in source or target"

    src_id_mask = np.isin(source_data["id_tokens"], np.asarray(sorted(pair_common_ids)))
    tgt_id_mask = np.isin(target_data["id_tokens"], np.asarray(sorted(pair_common_ids)))
    src_mask = src_label_mask & src_id_mask
    tgt_mask = tgt_label_mask & tgt_id_mask

    if not np.any(src_mask) or not np.any(tgt_mask):
        return None, None, None, "no label samples for pair common IDs"

    src_ids = source_data["id_tokens"][src_mask]
    tgt_ids = target_data["id_tokens"][tgt_mask]

    src_classes, src_counts = np.unique(src_ids, return_counts=True)
    tgt_classes, tgt_counts = np.unique(tgt_ids, return_counts=True)
    src_count_map = {str(k): int(v) for k, v in zip(src_classes.tolist(), src_counts.tolist())}
    tgt_count_map = {str(k): int(v) for k, v in zip(tgt_classes.tolist(), tgt_counts.tolist())}

    common_ids_label = sorted(
        [
            cid
            for cid in sorted(pair_common_ids)
            if src_count_map.get(cid, 0) >= int(min_id_repetitions)
            and tgt_count_map.get(cid, 0) >= int(min_id_repetitions)
        ]
    )

    if len(common_ids_label) < 2:
        return None, None, None, "fewer than 2 shared IDs at per-label repetition threshold"

    src_final_mask = src_mask & np.isin(source_data["id_tokens"], np.asarray(common_ids_label))
    tgt_final_mask = tgt_mask & np.isin(target_data["id_tokens"], np.asarray(common_ids_label))

    x_train = np.asarray(source_data["x_vec"][src_final_mask], dtype=np.float64)
    x_test = np.asarray(target_data["x_vec"][tgt_final_mask], dtype=np.float64)
    ids_train = np.asarray(source_data["id_tokens"][src_final_mask])
    ids_test = np.asarray(target_data["id_tokens"][tgt_final_mask])

    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        return None, None, None, "empty train/test arrays after per-label filtering"

    le = LabelEncoder().fit(common_ids_label)
    y_train = le.transform(ids_train)
    y_test = le.transform(ids_test)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None, None, None, "collapsed class space after encoding"

    train_paths = [source_data["grid_paths"][i] for i in np.where(src_final_mask)[0]]
    test_paths = [target_data["grid_paths"][i] for i in np.where(tgt_final_mask)[0]]
    train_frames = np.asarray(source_data["grid_frames"][src_final_mask], dtype=np.int64)
    test_frames = np.asarray(target_data["grid_frames"][tgt_final_mask], dtype=np.int64)

    need_grid_raw = "cnn3d_raw" in model_order
    need_grid_norm = "cnn3d_norm" in model_order
    train_grid_raw = None
    test_grid_raw = None
    train_grid_norm = None
    test_grid_norm = None
    if need_grid_raw:
        train_grid_raw = GridTrialDataset(
            grid_paths=train_paths,
            y=y_train,
            valid_frames=train_frames,
            clip_frames=int(clip_frames),
            normalize_by_trial=False,
        )
        test_grid_raw = GridTrialDataset(
            grid_paths=test_paths,
            y=y_test,
            valid_frames=test_frames,
            clip_frames=int(clip_frames),
            normalize_by_trial=False,
        )
    if need_grid_norm:
        train_grid_norm = GridTrialDataset(
            grid_paths=train_paths,
            y=y_train,
            valid_frames=train_frames,
            clip_frames=int(clip_frames),
            normalize_by_trial=True,
        )
        test_grid_norm = GridTrialDataset(
            grid_paths=test_paths,
            y=y_test,
            valid_frames=test_frames,
            clip_frames=int(clip_frames),
            normalize_by_trial=True,
        )

    vec_train_ds = VectorDataset(x_train, y_train)
    vec_test_ds = VectorDataset(x_test, y_test)

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    preds_by_model: Dict[str, np.ndarray] = {}
    seed_local = _stable_seed(state.seed, f"{source_mouse}->{target_mouse}:{label_name}")

    for mk in model_order:
        _set_seed(seed_local)
        if mk == "logreg":
            mm, pred = train_eval_logreg(x_train, y_train, x_test, y_test)
        elif mk == "mlp":
            mm, pred = train_eval_nn(
                make_model=lambda n_classes=len(common_ids_label), d=int(x_train.shape[1]): MLP(
                    n_classes=n_classes,
                    input_dim=d,
                ),
                train_ds=vec_train_ds,
                y_train=y_train,
                test_ds=vec_test_ds,
                test_y=y_test,
                epochs=state.epochs_mlp,
                lr=state.lr_vec,
                batch_size=state.batch_size_vec,
                patience=state.early_stop_patience,
                weight_decay=state.weight_decay,
                device=device,
                num_workers=state.num_workers_dl,
                seed=seed_local,
            )
        elif mk == "cnn1d":
            in_channels, seq_len = infer_cnn1d_shape(int(x_train.shape[1]), int(clip_frames))
            mm, pred = train_eval_nn(
                make_model=lambda n_classes=len(common_ids_label), ic=in_channels, sl=seq_len: CNN1D(
                    n_classes=n_classes,
                    in_channels=ic,
                    seq_len=sl,
                ),
                train_ds=vec_train_ds,
                y_train=y_train,
                test_ds=vec_test_ds,
                test_y=y_test,
                epochs=state.epochs_cnn1d,
                lr=state.lr_vec,
                batch_size=state.batch_size_vec,
                patience=state.early_stop_patience,
                weight_decay=state.weight_decay,
                device=device,
                num_workers=state.num_workers_dl,
                seed=seed_local,
            )
        elif mk == "cnn3d_raw":
            if train_grid_raw is None or test_grid_raw is None:
                raise RuntimeError("cnn3d_raw selected but raw grid datasets were not prepared")
            mm, pred = train_eval_nn(
                make_model=lambda n_classes=len(common_ids_label), in_ch=int(test_grid_raw.in_channels): CNN3D(
                    n_classes=n_classes,
                    in_channels=in_ch,
                ),
                train_ds=train_grid_raw,
                y_train=y_train,
                test_ds=test_grid_raw,
                test_y=y_test,
                epochs=state.epochs_cnn3d,
                lr=state.lr_cnn3d,
                batch_size=state.batch_size_grid,
                patience=state.early_stop_patience,
                weight_decay=state.weight_decay,
                device=device,
                num_workers=state.num_workers_dl,
                seed=seed_local,
            )
        elif mk == "cnn3d_norm":
            if train_grid_norm is None or test_grid_norm is None:
                raise RuntimeError("cnn3d_norm selected but normalized grid datasets were not prepared")
            mm, pred = train_eval_nn(
                make_model=lambda n_classes=len(common_ids_label), in_ch=int(test_grid_norm.in_channels): CNN3D(
                    n_classes=n_classes,
                    in_channels=in_ch,
                ),
                train_ds=train_grid_norm,
                y_train=y_train,
                test_ds=test_grid_norm,
                test_y=y_test,
                epochs=state.epochs_cnn3d,
                lr=state.lr_cnn3d,
                batch_size=state.batch_size_grid,
                patience=state.early_stop_patience,
                weight_decay=state.weight_decay,
                device=device,
                num_workers=state.num_workers_dl,
                seed=seed_local,
            )
        else:
            raise ValueError(f"Unsupported model key: {mk}")
        metrics_by_model[mk] = mm
        preds_by_model[mk] = pred

    best_model = max(model_order, key=lambda mk: metrics_by_model[mk]["macro_f1"])

    result_row = {
        "source_mouse": source_mouse,
        "target_mouse": target_mouse,
        "pair_direction": f"{source_mouse}__to__{target_mouse}",
        "label": label_name,
        "n_train_trials": int(x_train.shape[0]),
        "n_test_trials": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]),
        "clip_frames": int(clip_frames),
        "n_classes": int(len(common_ids_label)),
        "class_labels": [str(v) for v in common_ids_label],
        "best_model": best_model,
        "models": metrics_by_model,
        "common_ids_label": [str(v) for v in common_ids_label],
        "n_common_ids_label": int(len(common_ids_label)),
        "n_common_ids_pair": int(len(pair_common_ids)),
        "common_ids_pair": [str(v) for v in sorted(pair_common_ids)],
        "vec_source_train": source_data["vec_source"],
        "vec_source_test": target_data["vec_source"],
        "cache_path_train": source_data["cache_path"],
        "cache_path_test": target_data["cache_path"],
    }

    cms = {
        mk: confusion_matrix(y_test, preds_by_model[mk], labels=np.arange(len(common_ids_label)))
        for mk in model_order
    }
    cm_payload = {
        "labels": [str(v) for v in common_ids_label],
        "cms": cms,
        "best_model": best_model,
    }

    pred_payload = {
        "labels": [str(v) for v in common_ids_label],
        "y_true": [int(v) for v in y_test.tolist()],
        "predictions": {
            mk: [int(v) for v in np.asarray(preds_by_model[mk]).tolist()] for mk in model_order
        },
        "best_model": best_model,
        "source_mouse": source_mouse,
        "target_mouse": target_mouse,
        "label": label_name,
    }

    return result_row, cm_payload, pred_payload, None


def _compute_per_label_summary(records: Sequence[Dict[str, Any]], model_order: Sequence[str]) -> Dict[str, Any]:
    """Compute compact per-label summary aggregating across directions and models.
    
    Returns dict with per-label keys containing:
    - mean_accuracy: {model_name: {mean, std}}
    - mean_macro_f1: {model_name: {mean, std}}
    - model_ranking: sorted model names by mean macro_f1 (descending)
    - n_directions: count of unique directions
    - n_experiments: total (direction, model) pairs for this label
    """
    if not records:
        return {}
    
    labels = sorted({str(r["label"]) for r in records})
    summary: Dict[str, Any] = {}
    
    for label_name in labels:
        label_records = [r for r in records if str(r["label"]) == label_name]
        directions = sorted({str(r["pair_direction"]) for r in label_records})
        
        # Aggregate by model
        stats_by_model: Dict[str, Dict[str, Any]] = {}
        for model_name in model_order:
            model_records = [r for r in label_records if str(r["model"]) == model_name]
            if not model_records:
                continue
            
            accuracies = [float(r["accuracy"]) for r in model_records]
            macro_f1s = [float(r["macro_f1"]) for r in model_records]
            
            stats_by_model[model_name] = {
                "accuracy": {
                    "mean": float(np.mean(accuracies)),
                    "std": float(np.std(accuracies)),
                    "n_samples": len(accuracies),
                },
                "macro_f1": {
                    "mean": float(np.mean(macro_f1s)),
                    "std": float(np.std(macro_f1s)),
                    "n_samples": len(macro_f1s),
                },
            }
        
        # Rank models by mean macro_f1 (descending)
        model_ranking = sorted(
            stats_by_model.keys(),
            key=lambda m: stats_by_model[m]["macro_f1"]["mean"],
            reverse=True,
        )
        
        summary[label_name] = {
            "mean_accuracy": {m: stats_by_model[m]["accuracy"] for m in stats_by_model},
            "mean_macro_f1": {m: stats_by_model[m]["macro_f1"] for m in stats_by_model},
            "model_ranking": model_ranking,
            "n_directions": len(directions),
            "n_experiments": len(label_records),
        }
    
    return summary


def _plot_metric_heatmaps(
    *,
    records: Sequence[Dict[str, Any]],
    model_order: Sequence[str],
    value_key: str,
    figures_dir: Path,
    filename: str,
    title: str,
) -> Optional[Path]:
    if not records:
        return None

    direction_order = sorted({str(r["pair_direction"]) for r in records})
    label_order = sorted({str(r["label"]) for r in records})
    models = list(model_order)
    if len(models) == 0:
        return None

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(max(6 + len(models) * 2.5, len(label_order) * 1.2 * len(models)), max(6, len(direction_order) * 0.45)),
        squeeze=False,
    )

    for col_idx, model_name in enumerate(models):
        ax = axes[0][col_idx]
        mat = np.full((len(direction_order), len(label_order)), np.nan, dtype=float)

        for rec in records:
            if rec["model"] != model_name:
                continue
            i = direction_order.index(str(rec["pair_direction"]))
            j = label_order.index(str(rec["label"]))
            mat[i, j] = float(rec[value_key])

        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{model_name} {value_key}")
        ax.set_xticks(np.arange(len(label_order)))
        ax.set_xticklabels(label_order, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(direction_order)))
        ax.set_yticklabels([
            f"{_short_mouse_name(d.split('__to__')[0])}->{_short_mouse_name(d.split('__to__')[1])}"
            for d in direction_order
        ], fontsize=7)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = figures_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_label_aggregate_bars(
    *,
    records: Sequence[Dict[str, Any]],
    model_order: Sequence[str],
    figures_dir: Path,
) -> Optional[Path]:
    if not records:
        return None

    labels = sorted({str(r["label"]) for r in records})
    models = list(model_order)
    if len(models) == 0:
        return None
    colors = {
        "logreg": "#4C72B0",
        "mlp": "#55A868",
        "cnn1d": "#8172B2",
        "cnn3d_raw": "#DD8452",
        "cnn3d_norm": "#937860",
    }

    width = min(0.82 / len(models), 0.22)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.1), 4.8))

    for axis, metric in zip(axes, ["accuracy", "macro_f1"]):
        for idx, model_name in enumerate(models):
            vals_by_label: List[float] = []
            err_by_label: List[float] = []
            for label in labels:
                vals = [
                    float(r[metric])
                    for r in records
                    if str(r["label"]) == label and str(r["model"]) == model_name
                ]
                vals_by_label.append(float(np.mean(vals)) if vals else np.nan)
                err_by_label.append(float(np.std(vals)) if vals else 0.0)

            axis.bar(
                x + (idx - (len(models) - 1) / 2.0) * width,
                vals_by_label,
                width,
                yerr=err_by_label,
                capsize=3,
                alpha=0.85,
                color=colors[model_name],
                label=model_name,
            )

        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel(metric)
        axis.set_title(f"Mean {metric} across directed pairs")
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = figures_dir / "03_mean_scores_by_label.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_confusion_per_direction_label(
    *,
    direction_key: str,
    label_name: str,
    payload: Dict[str, Any],
    model_order: Sequence[str],
    figures_dir: Path,
) -> Path:
    model_titles = MODEL_TITLES

    labels_order = payload["labels"]
    src, tgt = direction_key.split("__to__", 1)

    fig, axes = plt.subplots(1, len(model_order), figsize=(max(4.0 * len(model_order), 9.0), 3.8), squeeze=False)
    for col_idx, mk in enumerate(model_order):
        ax = axes[0][col_idx]
        cm_arr = np.asarray(payload["cms"][mk], dtype=float)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)
        ConfusionMatrixDisplay(cm_norm, display_labels=labels_order).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format=".2f",
        )
        title = model_titles[mk]
        if mk == payload.get("best_model"):
            title += " ★"
        ax.set_title(title, fontsize=8)

    fig.suptitle(
        f"{_short_mouse_name(src)} -> {_short_mouse_name(tgt)} | {label_name}",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = figures_dir / (
        f"cm_{_safe_token(_short_mouse_name(src))}_to_{_safe_token(_short_mouse_name(tgt))}"
        f"_{_safe_token(label_name)}.png"
    )
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

    log_path = logs_dir / "run.log"

    with open(log_path, "w", encoding="utf-8") as log_fp, redirect_stdout(log_fp), redirect_stderr(log_fp):
        print("=" * 90)
        print("Cross-Mouse Video ID Decoding: selectable vector/grid models")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)

        print("\nResolved arguments:")
        print(f"  output_folder:       {state.output_folder}")
        print(f"  data_root:           {state.data_root}")
        print(f"  meta_root:           {state.meta_root}")
        print(f"  p_active:            {state.p_active}")
        print(f"  per_trial_thresh:    {state.per_trial_thresh}")
        print(f"  zz_folder:           {state.zz_folder}")
        print(f"  vectorization_method:{state.vectorization_method}")
        print(f"  models:              {state.models}")
        print(f"  mice:                {state.mice}")
        print(f"  clip_frames:         {state.clip_frames}")
        print(f"  max_trials:          {state.max_trials}")
        print(f"  min_id_repetitions:  {state.min_id_repetitions}")
        print(f"  grid_subdir:         {state.grid_subdir}")
        print(f"  device:              {device}")

        discovered_mice = _discover_mice(state.data_root)
        selected_mice = state.mice if state.mice is not None else discovered_mice
        selected_mice = [m for m in selected_mice if m in discovered_mice]
        if len(selected_mice) < 2:
            raise RuntimeError("Need at least 2 valid mice for cross-mouse ID decoding")

        print(f"\nDiscovered mice: {len(discovered_mice)}")
        print(f"Selected mice: {len(selected_mice)}")

        model_order = list(state.models)

        # Determine global clip
        if state.clip_frames is not None:
            global_clip = int(state.clip_frames)
            print(f"\nUsing user clip_frames={global_clip}")
        else:
            print("\nPre-scanning valid_frames to determine global clip_frames ...")
            mins: List[int] = []
            for mouse_name in selected_mice:
                try:
                    _b, _l, _t, vf = load_labelled_barcodes(
                        state.data_root,
                        state.meta_root,
                        mouse_name,
                        state.zz_folder,
                        max_trials=state.max_trials,
                    )
                    if len(vf) > 0:
                        mins.append(int(vf.min()))
                except Exception as exc:
                    print(f"  Warning: pre-scan failed for {mouse_name}: {exc}")
            if not mins:
                raise RuntimeError("Could not determine clip_frames from selected mice")
            global_clip = int(min(mins))
            print(f"  global_clip_frames={global_clip}")

        # Candidate pair selection from metadata repeated-ID filter
        print("\nSelecting candidate pairs from metadata ...")
        candidate_pairs, id_set_by_mouse, pair_common_ids = _select_candidate_pairs(
            state.meta_root,
            selected_mice,
            state.min_id_repetitions,
        )
        print(f"  candidate unordered pairs: {len(candidate_pairs)}")

        # Load mouse datasets
        print("\nLoading per-mouse aligned data (vector + grid + IDs) ...")
        mouse_data: Dict[str, Dict[str, Any]] = {}
        for mouse_name in sorted({m for p in candidate_pairs for m in p}):
            print(f"  {mouse_name} ...", end=" ")
            try:
                data = _load_mouse_data(state, mouse_name, global_clip)
                if data is None:
                    print("skipped")
                    continue
                mouse_data[mouse_name] = data
                print(f"ok (trials={len(data['trial_ids'])}, ids={len(data['id_set'])}, source={data['vec_source']})")
            except Exception as exc:
                print(f"FAILED: {exc}")
                traceback.print_exc()

        if len(mouse_data) < 2:
            raise RuntimeError("Fewer than 2 mice have usable aligned data")

        # Containers
        per_direction: Dict[str, Dict[str, Any]] = {}
        confusion_payload: Dict[str, Dict[str, Any]] = {}
        prediction_payload: Dict[str, Dict[str, Any]] = {}
        csv_records: List[Dict[str, Any]] = []
        skipped_units: List[Dict[str, Any]] = []

        print("\nRunning directional pair decoding by label ...")
        for m1, m2 in candidate_pairs:
            if m1 not in mouse_data or m2 not in mouse_data:
                continue

            pair_ids = pair_common_ids[(m1, m2)]
            if len(pair_ids) == 0:
                continue

            for source_mouse, target_mouse in [(m1, m2), (m2, m1)]:
                source_data = mouse_data[source_mouse]
                target_data = mouse_data[target_mouse]

                direction_key = f"{source_mouse}__to__{target_mouse}"
                per_direction[direction_key] = {
                    "source_mouse": source_mouse,
                    "target_mouse": target_mouse,
                    "pair_common_ids": sorted(pair_ids),
                    "n_pair_common_ids": int(len(pair_ids)),
                    "labels": {},
                }
                confusion_payload[direction_key] = {}
                prediction_payload[direction_key] = {}

                shared_labels = sorted(
                    set(source_data["labels"].tolist()) & set(target_data["labels"].tolist())
                )
                print(
                    f"  direction {source_mouse[-20:]} -> {target_mouse[-20:]}: "
                    f"shared_labels={len(shared_labels)}, pair_common_ids={len(pair_ids)}"
                )

                for label_name in shared_labels:
                    result_row, cm_row, pred_row, skip_reason = _evaluate_direction_label(
                        source_mouse=source_mouse,
                        target_mouse=target_mouse,
                        label_name=label_name,
                        source_data=source_data,
                        target_data=target_data,
                        pair_common_ids=pair_ids,
                        min_id_repetitions=state.min_id_repetitions,
                        clip_frames=global_clip,
                        state=state,
                        device=device,
                        model_order=model_order,
                    )

                    if result_row is None or cm_row is None or pred_row is None:
                        skipped_units.append(
                            {
                                "direction": direction_key,
                                "label": label_name,
                                "reason": skip_reason or "unknown",
                            }
                        )
                        continue

                    per_direction[direction_key]["labels"][label_name] = result_row
                    confusion_payload[direction_key][label_name] = cm_row
                    prediction_payload[direction_key][label_name] = pred_row

                    best = result_row["best_model"]
                    best_f1 = result_row["models"][best]["macro_f1"]
                    print(
                        f"    label={label_name}: n_train={result_row['n_train_trials']}, "
                        f"n_test={result_row['n_test_trials']}, classes={result_row['n_classes']}, "
                        f"best={best}:{best_f1:.3f}"
                    )

                    for model_name in model_order:
                        mm = result_row["models"][model_name]
                        csv_records.append(
                            {
                                "source_mouse": source_mouse,
                                "target_mouse": target_mouse,
                                "pair_direction": direction_key,
                                "label": label_name,
                                "model": model_name,
                                "input": "grid" if model_name.startswith("cnn3d") else "vector",
                                "n_train_trials": result_row["n_train_trials"],
                                "n_test_trials": result_row["n_test_trials"],
                                "n_features": result_row["n_features"],
                                "clip_frames": result_row["clip_frames"],
                                "n_classes": result_row["n_classes"],
                                "class_labels": "|".join(result_row["class_labels"]),
                                "accuracy": float(mm["accuracy"]),
                                "macro_f1": float(mm["macro_f1"]),
                                "best_model": result_row["best_model"],
                                "n_common_ids_pair": result_row["n_common_ids_pair"],
                                "common_ids_pair": "|".join(result_row["common_ids_pair"]),
                                "n_common_ids_label": result_row["n_common_ids_label"],
                                "common_ids_label": "|".join(result_row["common_ids_label"]),
                                "vec_source_train": result_row["vec_source_train"],
                                "vec_source_test": result_row["vec_source_test"],
                                "cache_path_train": result_row["cache_path_train"],
                                "cache_path_test": result_row["cache_path_test"],
                            }
                        )

                if not per_direction[direction_key]["labels"]:
                    # Keep output focused on valid experiments only.
                    del per_direction[direction_key]
                    del confusion_payload[direction_key]
                    del prediction_payload[direction_key]

        if not csv_records:
            raise RuntimeError("No valid direction/label experiments were produced")

        # Compute per-label summary for downstream figure regeneration
        print("\nComputing per-label summary ...")
        per_label_summary = _compute_per_label_summary(csv_records, model_order=model_order)

        # Figures
        print("\nSaving figures ...")
        fig_paths: List[Path] = []
        fig_acc = _plot_metric_heatmaps(
            records=csv_records,
            model_order=model_order,
            value_key="accuracy",
            figures_dir=figures_dir,
            filename="01_accuracy_by_pair_label.png",
            title="Accuracy by directed mouse pair and label",
        )
        if fig_acc is not None:
            fig_paths.append(fig_acc)
            print(f"  Saved: {fig_acc}")

        fig_f1 = _plot_metric_heatmaps(
            records=csv_records,
            model_order=model_order,
            value_key="macro_f1",
            figures_dir=figures_dir,
            filename="02_macro_f1_by_pair_label.png",
            title="Macro-F1 by directed mouse pair and label",
        )
        if fig_f1 is not None:
            fig_paths.append(fig_f1)
            print(f"  Saved: {fig_f1}")

        fig_mean = _plot_label_aggregate_bars(records=csv_records, model_order=model_order, figures_dir=figures_dir)
        if fig_mean is not None:
            fig_paths.append(fig_mean)
            print(f"  Saved: {fig_mean}")

        for direction_key, label_payloads in confusion_payload.items():
            for label_name, payload in label_payloads.items():
                out = _plot_confusion_per_direction_label(
                    direction_key=direction_key,
                    label_name=label_name,
                    payload=payload,
                    model_order=model_order,
                    figures_dir=figures_dir,
                )
                fig_paths.append(out)

        # Output JSON/CSV
        summary_json_path = state.output_folder / "cross_mouse_id_decoding_metrics.json"
        summary_csv_path = state.output_folder / "cross_mouse_id_decoding_metrics.csv"
        confusion_json_path = state.output_folder / "cross_mouse_id_decoding_confusion_matrices.json"
        predictions_json_path = state.output_folder / "cross_mouse_id_decoding_prediction_outputs.json"

        confusion_out: Dict[str, Dict[str, Any]] = {}
        for direction_key, labels_map in confusion_payload.items():
            confusion_out[direction_key] = {}
            for label_name, payload in labels_map.items():
                cms_counts: Dict[str, List[List[int]]] = {}
                cms_norm: Dict[str, List[List[float]]] = {}
                for mk in model_order:
                    cm = np.asarray(payload["cms"][mk], dtype=np.int64)
                    row_sums = cm.sum(axis=1, keepdims=True)
                    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
                    cms_counts[mk] = cm.tolist()
                    cms_norm[mk] = cm_norm.tolist()

                confusion_out[direction_key][label_name] = {
                    "labels": [str(v) for v in payload["labels"]],
                    "best_model": payload["best_model"],
                    "cms_counts": cms_counts,
                    "cms_normalized": cms_norm,
                }

        with open(confusion_json_path, "w", encoding="utf-8") as fp:
            json.dump(confusion_out, fp, indent=2)
        print(f"Wrote confusion JSON: {confusion_json_path}")

        with open(predictions_json_path, "w", encoding="utf-8") as fp:
            json.dump(prediction_payload, fp, indent=2)
        print(f"Wrote prediction JSON: {predictions_json_path}")

        with open(summary_csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "source_mouse",
                    "target_mouse",
                    "pair_direction",
                    "label",
                    "model",
                    "input",
                    "n_train_trials",
                    "n_test_trials",
                    "n_features",
                    "clip_frames",
                    "n_classes",
                    "class_labels",
                    "accuracy",
                    "macro_f1",
                    "best_model",
                    "n_common_ids_pair",
                    "common_ids_pair",
                    "n_common_ids_label",
                    "common_ids_label",
                    "vec_source_train",
                    "vec_source_test",
                    "cache_path_train",
                    "cache_path_test",
                ],
            )
            writer.writeheader()
            for row in csv_records:
                writer.writerow(row)
        print(f"Wrote summary CSV: {summary_csv_path}")

        payload = {
            "method": state.vectorization_method,
            "p_active": state.p_active,
            "per_trial_thresh": state.per_trial_thresh,
            "zz_folder": state.zz_folder,
            "grid_subdir": state.grid_subdir,
            "global_clip_frames": int(global_clip),
            "min_id_repetitions": int(state.min_id_repetitions),
            "eligible_mice": sorted(mouse_data.keys()),
            "selected_candidate_pairs": [[m1, m2] for (m1, m2) in candidate_pairs],
            "id_set_by_mouse": {k: sorted(list(v)) for k, v in id_set_by_mouse.items()},
            "results": per_direction,
            "per_label_summary": per_label_summary,
            "n_direction_label_units": int(len(csv_records) // max(1, len(model_order))),
            "models": model_order,
            "figures": [str(p) for p in fig_paths],
            "confusion_matrices_path": str(confusion_json_path),
            "prediction_outputs_path": str(predictions_json_path),
            "log_path": str(log_path),
            "cache_dir": (
                str(state.cache_dir)
                if state.cache_dir is not None
                else "<data_root>/<mouse>/cache"
            ),
            "device": str(device),
            "skipped_units": skipped_units,
        }
        with open(summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"Wrote summary JSON: {summary_json_path}")

    return {
        "log_path": str(log_path),
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "confusion_json_path": str(confusion_json_path),
        "predictions_json_path": str(predictions_json_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-mouse video ID decoding by label: train on one mouse, test on another, "
            "using selectable vector/grid models."
        )
    )
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)

    parser.add_argument("--p-active", required=True, type=int)
    parser.add_argument("--per-trial-thresh", required=True, type=_str2bool)
    parser.add_argument("--vectorization-method", required=True, type=str)

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

    parser.add_argument("--min-id-repetitions", default=5, type=int)

    parser.add_argument("--batch-size-vec", default=64, type=int)
    parser.add_argument("--batch-size-grid", default=16, type=int)
    parser.add_argument("--epochs-mlp", default=60, type=int)
    parser.add_argument("--epochs-cnn1d", default=60, type=int)
    parser.add_argument("--lr-vec", default=1e-3, type=float)
    parser.add_argument("--epochs-cnn3d", default=40, type=int)
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

    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    state = RunState(
        output_folder=output_folder,
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
        batch_size_vec=args.batch_size_vec,
        epochs_mlp=args.epochs_mlp,
        epochs_cnn1d=args.epochs_cnn1d,
        lr_vec=args.lr_vec,
        batch_size_grid=args.batch_size_grid,
        epochs_cnn3d=args.epochs_cnn3d,
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
