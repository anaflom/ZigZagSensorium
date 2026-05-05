#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-mouse segment-ID decoding with leave-one-mouse-out evaluation."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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


def _extract_turnover_segment_vector(
    full_feature: np.ndarray,
    *,
    clip_frames: int,
    start_frame: int,
    seg_length: int,
) -> np.ndarray:
    feat = np.asarray(full_feature, dtype=np.float64)
    if clip_frames <= 0:
        raise ValueError(f"Invalid clip_frames={clip_frames}")
    if feat.ndim != 1:
        raise ValueError(f"Expected 1D feature row, got shape {feat.shape}")
    if feat.shape[0] % clip_frames != 0:
        raise ValueError(f"Feature length {feat.shape[0]} not divisible by clip_frames={clip_frames}")
    if start_frame < 0 or seg_length <= 0:
        raise ValueError(f"Invalid segment window start={start_frame}, len={seg_length}")
    end = int(start_frame + seg_length)
    if end > clip_frames:
        raise ValueError(f"Segment end {end} exceeds clip_frames={clip_frames}")

    blocks: List[np.ndarray] = []
    for dim in (0, 1, 2):
        base = dim * clip_frames
        blocks.append(feat[base + start_frame : base + end])
    out = np.concatenate(blocks, axis=0)
    return np.asarray(out, dtype=np.float64)


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
    min_segment_repetitions: int
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
    segment_lengths: Dict[str, int]


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


def _safe_token(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _load_mouse_segment_samples(state: RunState, mouse_name: str) -> Optional[Dict[str, Any]]:
    seg_df, seg_counters = build_segment_sample_records(
        meta_root=state.meta_root,
        mouse_name=mouse_name,
        target_labels=TARGET_LABELS,
        segment_length_by_label=state.segment_lengths,
        videos_subdir="global_meta/videos",
    )
    if len(seg_df) == 0:
        return None

    barcodes, vec_labels, vec_trial_ids, vec_valid_frames = load_labelled_barcodes(
        state.data_root,
        state.meta_root,
        mouse_name,
        state.zz_folder,
        max_trials=state.max_trials,
    )
    if len(barcodes) == 0:
        return None

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
        message_prefix=f"  [{mouse_name}] ",
    )

    if xmat.ndim != 2 or xmat.shape[1] % 3 != 0:
        raise RuntimeError(f"Unexpected Turnover feature matrix shape for {mouse_name}: {xmat.shape}")
    clip_used = int(xmat.shape[1] // 3)

    grid_paths, grid_labels, grid_trial_ids, grid_valid_frames = load_labelled_grid_paths(
        state.data_root,
        state.meta_root,
        mouse_name,
        grid_subdir=state.grid_subdir,
    )
    if len(grid_paths) == 0:
        return None

    vec_idx = {int(tid): i for i, tid in enumerate(vec_trial_ids_int)}
    grid_idx = {int(tid): i for i, tid in enumerate(grid_trial_ids)}

    records: List[Dict[str, Any]] = []
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

        records.append(
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

    return {
        "mouse": mouse_name,
        "records": records,
        "clip_used": clip_used,
        "vec_source": vec_source,
        "cache_path": str(cache_path),
        "seg_counters": seg_counters,
        "align_counters": dict(align_counters),
    }


def _aggregate_plot(rows: Sequence[Dict[str, Any]], model_order: Sequence[str], out_path: Path) -> Optional[Path]:
    if not rows:
        return None
    labels = sorted({str(r["label"]) for r in rows})
    if not labels:
        return None
    x = np.arange(len(labels))
    width = min(0.82 / len(model_order), 0.22)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 4.5))
    y_values: List[float] = []
    for idx, mk in enumerate(model_order):
        vals = []
        errs = []
        for label in labels:
            vv = [float(r["macro_f1"]) for r in rows if str(r["label"]) == label and str(r["model"]) == mk]
            vals.append(float(np.mean(vv)) if vv else np.nan)
            errs.append(float(np.std(vv)) if vv else 0.0)
            y_values.extend([float(v) for v in vv if np.isfinite(v)])
        ax.bar(
            x + (idx - (len(model_order) - 1) / 2.0) * width,
            vals,
            width,
            yerr=errs,
            capsize=3,
            alpha=0.88,
            color=MODEL_COLORS.get(mk, "#4C72B0"),
            label=mk,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    if y_values:
        y_min = float(min(y_values))
        y_max = float(max(y_values))
        if np.isclose(y_min, y_max):
            pad = max(0.01, 0.2 * max(y_max, 1e-3))
        else:
            pad = max(0.01, 0.15 * (y_max - y_min))
        lower = max(0.0, y_min - pad)
        upper = min(1.0, y_max + pad)
        if upper - lower < 0.03:
            center = 0.5 * (lower + upper)
            lower = max(0.0, center - 0.015)
            upper = min(1.0, center + 0.015)
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-mouse segment-ID decoding (LOMO): mean macro-F1 by label")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
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

    model_order = list(state.models)

    with open(log_path, "w", encoding="utf-8") as log_fp, redirect_stdout(log_fp), redirect_stderr(log_fp):
        print("=" * 90)
        print("Cross-Mouse Segment-ID Decoding: global leave-one-mouse-out")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)
        print(f"models={model_order}")
        print(f"min_segment_repetitions={state.min_segment_repetitions}")

        discovered_mice = _discover_mice(state.data_root)
        selected_mice = state.mice if state.mice is not None else discovered_mice
        selected_mice = [m for m in selected_mice if m in discovered_mice]
        if len(selected_mice) < 2:
            raise RuntimeError("Need at least 2 valid mice for cross-mouse segment-ID decoding")

        mouse_payload: Dict[str, Dict[str, Any]] = {}
        for mouse_name in selected_mice:
            try:
                payload = _load_mouse_segment_samples(state, mouse_name)
            except Exception as exc:
                print(f"  load failed for {mouse_name}: {exc}")
                traceback.print_exc()
                continue
            if payload is None or len(payload["records"]) == 0:
                continue
            mouse_payload[mouse_name] = payload
            print(
                f"  loaded {mouse_name}: records={len(payload['records'])}, "
                f"clip_used={payload['clip_used']}"
            )

        eligible_mice = sorted(mouse_payload.keys())
        if len(eligible_mice) < 2:
            raise RuntimeError("Fewer than 2 mice with usable aligned segment data")

        rows: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        confusion_payload: Dict[str, Any] = {}
        prediction_payload: Dict[str, Any] = {}

        for test_mouse in eligible_mice:
            train_mice = [m for m in eligible_mice if m != test_mouse]
            if not train_mice:
                continue

            fold_key = f"holdout_{test_mouse}"
            confusion_payload[fold_key] = {}
            prediction_payload[fold_key] = {}

            test_records_all = mouse_payload[test_mouse]["records"]
            train_records_all = [r for m in train_mice for r in mouse_payload[m]["records"]]

            for label_name in TARGET_LABELS:
                test_group = [r for r in test_records_all if str(r["label"]) == label_name]
                train_group = [r for r in train_records_all if str(r["label"]) == label_name]
                if len(test_group) == 0 or len(train_group) == 0:
                    skipped.append({"fold": fold_key, "label": label_name, "reason": "missing_label_samples", "detail": ""})
                    continue

                counts_test = Counter([str(r["segment_id"]) for r in test_group])
                counts_train = Counter([str(r["segment_id"]) for r in train_group])
                shared_segment_ids = sorted(
                    [
                        sid
                        for sid in sorted(set(counts_test.keys()) & set(counts_train.keys()))
                        if int(counts_test[sid]) >= int(state.min_segment_repetitions)
                        and int(counts_train[sid]) >= int(state.min_segment_repetitions)
                    ]
                )

                if len(shared_segment_ids) < 2:
                    skipped.append(
                        {
                            "fold": fold_key,
                            "label": label_name,
                            "reason": "insufficient_shared_segment_ids",
                            "detail": str(len(shared_segment_ids)),
                        }
                    )
                    continue

                train_keep = [r for r in train_group if str(r["segment_id"]) in shared_segment_ids]
                test_keep = [r for r in test_group if str(r["segment_id"]) in shared_segment_ids]
                if len(train_keep) == 0 or len(test_keep) == 0:
                    skipped.append({"fold": fold_key, "label": label_name, "reason": "empty_after_shared_filter", "detail": ""})
                    continue

                # Count only mice that contribute label-specific training samples.
                n_train_mice_label = int(len({str(r["mouse"]) for r in train_keep}))

                train_seglen = sorted({int(r["seg_length"]) for r in train_keep})
                test_seglen = sorted({int(r["seg_length"]) for r in test_keep})
                if len(train_seglen) != 1 or len(test_seglen) != 1 or train_seglen[0] != test_seglen[0]:
                    skipped.append(
                        {
                            "fold": fold_key,
                            "label": label_name,
                            "reason": "inconsistent_segment_lengths",
                            "detail": f"train={train_seglen};test={test_seglen}",
                        }
                    )
                    continue
                seg_len = int(train_seglen[0])

                le = LabelEncoder().fit(shared_segment_ids)
                y_train = le.transform(np.asarray([str(r["segment_id"]) for r in train_keep]))
                y_test = le.transform(np.asarray([str(r["segment_id"]) for r in test_keep]))
                n_classes = int(len(le.classes_))
                if n_classes < 2:
                    skipped.append({"fold": fold_key, "label": label_name, "reason": "insufficient_classes", "detail": str(n_classes)})
                    continue

                x_train = np.asarray([r["vector"] for r in train_keep], dtype=np.float64)
                x_test = np.asarray([r["vector"] for r in test_keep], dtype=np.float64)

                train_vec_ds = VectorDataset(x_train, y_train)
                test_vec_ds = VectorDataset(x_test, y_test)

                need_raw = "cnn3d_raw" in model_order
                need_norm = "cnn3d_norm" in model_order
                train_grid_raw = None
                test_grid_raw = None
                train_grid_norm = None
                test_grid_norm = None
                if need_raw:
                    train_grid_raw = SegmentGridDataset(
                        grid_paths=[r["grid_path"] for r in train_keep],
                        y=y_train,
                        valid_frames=np.asarray([r["grid_valid_frames"] for r in train_keep], dtype=np.int64),
                        start_frames=np.asarray([r["start_frame"] for r in train_keep], dtype=np.int64),
                        seg_lengths=np.asarray([r["seg_length"] for r in train_keep], dtype=np.int64),
                        normalize_by_trial=False,
                    )
                    test_grid_raw = SegmentGridDataset(
                        grid_paths=[r["grid_path"] for r in test_keep],
                        y=y_test,
                        valid_frames=np.asarray([r["grid_valid_frames"] for r in test_keep], dtype=np.int64),
                        start_frames=np.asarray([r["start_frame"] for r in test_keep], dtype=np.int64),
                        seg_lengths=np.asarray([r["seg_length"] for r in test_keep], dtype=np.int64),
                        normalize_by_trial=False,
                    )
                if need_norm:
                    train_grid_norm = SegmentGridDataset(
                        grid_paths=[r["grid_path"] for r in train_keep],
                        y=y_train,
                        valid_frames=np.asarray([r["grid_valid_frames"] for r in train_keep], dtype=np.int64),
                        start_frames=np.asarray([r["start_frame"] for r in train_keep], dtype=np.int64),
                        seg_lengths=np.asarray([r["seg_length"] for r in train_keep], dtype=np.int64),
                        normalize_by_trial=True,
                    )
                    test_grid_norm = SegmentGridDataset(
                        grid_paths=[r["grid_path"] for r in test_keep],
                        y=y_test,
                        valid_frames=np.asarray([r["grid_valid_frames"] for r in test_keep], dtype=np.int64),
                        start_frames=np.asarray([r["start_frame"] for r in test_keep], dtype=np.int64),
                        seg_lengths=np.asarray([r["seg_length"] for r in test_keep], dtype=np.int64),
                        normalize_by_trial=True,
                    )

                model_metrics: Dict[str, Dict[str, float]] = {}
                model_preds: Dict[str, np.ndarray] = {}

                for mk in model_order:
                    _set_seed(state.seed)
                    if mk == "logreg":
                        mm, pred = train_eval_logreg(x_train, y_train, x_test, y_test)
                    elif mk == "mlp":
                        mm, pred = train_eval_nn(
                            make_model=lambda n_classes=n_classes, dim=int(x_train.shape[1]): MLP(n_classes=n_classes, input_dim=dim),
                            train_ds=train_vec_ds,
                            y_train=y_train,
                            test_ds=test_vec_ds,
                            test_y=y_test,
                            epochs=state.epochs_mlp,
                            lr=state.lr_vec,
                            batch_size=state.batch_size_vec,
                            patience=state.early_stop_patience,
                            weight_decay=state.weight_decay,
                            device=device,
                            num_workers=state.num_workers_dl,
                            seed=state.seed,
                        )
                    elif mk == "cnn1d":
                        in_channels, seq_len = infer_cnn1d_shape(int(x_train.shape[1]), int(seg_len))
                        mm, pred = train_eval_nn(
                            make_model=lambda n_classes=n_classes, ic=in_channels, sl=seq_len: CNN1D(
                                n_classes=n_classes,
                                in_channels=ic,
                                seq_len=sl,
                            ),
                            train_ds=train_vec_ds,
                            y_train=y_train,
                            test_ds=test_vec_ds,
                            test_y=y_test,
                            epochs=state.epochs_cnn1d,
                            lr=state.lr_vec,
                            batch_size=state.batch_size_vec,
                            patience=state.early_stop_patience,
                            weight_decay=state.weight_decay,
                            device=device,
                            num_workers=state.num_workers_dl,
                            seed=state.seed,
                        )
                    elif mk == "cnn3d_raw":
                        if train_grid_raw is None or test_grid_raw is None:
                            raise RuntimeError("cnn3d_raw selected but raw grid datasets unavailable")
                        mm, pred = train_eval_nn(
                            make_model=lambda n_classes=n_classes, in_ch=int(test_grid_raw.in_channels): CNN3D(
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
                            seed=state.seed,
                        )
                    elif mk == "cnn3d_norm":
                        if train_grid_norm is None or test_grid_norm is None:
                            raise RuntimeError("cnn3d_norm selected but normalized grid datasets unavailable")
                        mm, pred = train_eval_nn(
                            make_model=lambda n_classes=n_classes, in_ch=int(test_grid_norm.in_channels): CNN3D(
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
                            seed=state.seed,
                        )
                    else:
                        raise ValueError(f"Unsupported model key: {mk}")

                    model_metrics[mk] = {
                        "accuracy": float(mm["accuracy"]),
                        "macro_f1": float(mm["macro_f1"]),
                    }
                    model_preds[mk] = np.asarray(pred, dtype=np.int64)

                best_model = max(model_order, key=lambda mk: model_metrics[mk]["macro_f1"])
                cms = {
                    mk: confusion_matrix(y_test, model_preds[mk], labels=np.arange(n_classes))
                    for mk in model_order
                }

                fold_label_key = f"{fold_key}::{label_name}"
                confusion_payload[fold_key][label_name] = {
                    "labels": [str(v) for v in le.classes_],
                    "best_model": best_model,
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
                prediction_payload[fold_key][label_name] = {
                    "labels": [str(v) for v in le.classes_],
                    "y_true": [int(v) for v in y_test.tolist()],
                    "predictions": {mk: [int(v) for v in model_preds[mk].tolist()] for mk in model_order},
                    "samples": [
                        {
                            "trial_id": int(r["trial_id"]),
                            "segment_id": str(r["segment_id"]),
                            "segment_index": int(r["segment_index"]),
                            "start_frame": int(r["start_frame"]),
                            "seg_length": int(r["seg_length"]),
                            "video_id": str(r["video_id"]),
                        }
                        for r in test_keep
                    ],
                }

                for mk in model_order:
                    rows.append(
                        {
                            "heldout_mouse": test_mouse,
                            "n_train_mice": n_train_mice_label,
                            "n_train_mice_pool": int(len(train_mice)),
                            "label": label_name,
                            "model": mk,
                            "input": "grid" if mk.startswith("cnn3d") else "vector",
                            "n_train_segments": int(len(train_keep)),
                            "n_test_segments": int(len(test_keep)),
                            "n_classes": int(n_classes),
                            "class_labels": "|".join([str(v) for v in le.classes_]),
                            "min_segment_repetitions": int(state.min_segment_repetitions),
                            "shared_segment_ids": "|".join(shared_segment_ids),
                            "accuracy": float(model_metrics[mk]["accuracy"]),
                            "macro_f1": float(model_metrics[mk]["macro_f1"]),
                            "best_model": best_model,
                        }
                    )

                print(
                    f"  {fold_label_key}: train={len(train_keep)}, test={len(test_keep)}, "
                    f"classes={n_classes}, best={best_model}:{model_metrics[best_model]['macro_f1']:.3f}"
                )

        if not rows:
            raise RuntimeError("No valid cross-mouse segment-ID experiments were produced")

        summary_csv_path = state.output_folder / "cross_mouse_segment_id_metrics.csv"
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "heldout_mouse",
                    "n_train_mice",
                    "n_train_mice_pool",
                    "label",
                    "model",
                    "input",
                    "n_train_segments",
                    "n_test_segments",
                    "n_classes",
                    "class_labels",
                    "min_segment_repetitions",
                    "shared_segment_ids",
                    "accuracy",
                    "macro_f1",
                    "best_model",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        skip_csv_path = state.output_folder / "cross_mouse_segment_id_skips.csv"
        with open(skip_csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["fold", "label", "reason", "detail"])
            writer.writeheader()
            for row in skipped:
                writer.writerow(row)

        confusion_json_path = state.output_folder / "cross_mouse_segment_id_confusion_matrices.json"
        with open(confusion_json_path, "w", encoding="utf-8") as fp:
            json.dump(confusion_payload, fp, indent=2)

        predictions_json_path = state.output_folder / "cross_mouse_segment_id_prediction_outputs.json"
        with open(predictions_json_path, "w", encoding="utf-8") as fp:
            json.dump(prediction_payload, fp, indent=2)

        fig_path = _aggregate_plot(rows, model_order=model_order, out_path=figures_dir / "01_mean_macro_f1_by_label.png")

        summary_json_path = state.output_folder / "cross_mouse_segment_id_metrics.json"
        payload = {
            "method": "Turnover",
            "models": model_order,
            "target_labels": list(TARGET_LABELS),
            "eligible_mice": eligible_mice,
            "min_segment_repetitions": int(state.min_segment_repetitions),
            "results_csv_path": str(summary_csv_path),
            "skips_csv_path": str(skip_csv_path),
            "confusion_json_path": str(confusion_json_path),
            "predictions_json_path": str(predictions_json_path),
            "figure_paths": [str(fig_path)] if fig_path is not None else [],
            "log_path": str(log_path),
            "device": str(device),
        }
        with open(summary_json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    return {
        "log_path": str(log_path),
        "summary_json_path": str(summary_json_path),
        "summary_csv_path": str(summary_csv_path),
        "skip_csv_path": str(skip_csv_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-mouse segment-ID decoding with global leave-one-mouse-out and selectable models."
    )

    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)
    parser.add_argument("--p-active", required=True, type=int)
    parser.add_argument("--per-trial-thresh", required=True, type=_str2bool)

    parser.add_argument("--mice", default=None, type=_opt_csv_list)
    parser.add_argument("--models", default=None, type=_opt_csv_list)
    parser.add_argument("--clip-frames", default=None, type=_opt_int)
    parser.add_argument("--grid-subdir", default="trials_grid")
    parser.add_argument("--cache-dir", default=None, type=Path)
    parser.add_argument("--max-trials", default=None, type=_opt_int)

    parser.add_argument("--min-segment-repetitions", default=7, type=int)

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

    parser.add_argument("--seg-len-naturalimages", default=12, type=int)
    parser.add_argument("--seg-len-pinknoise", default=27, type=int)
    parser.add_argument("--seg-len-randomdots", default=60, type=int)
    parser.add_argument("--seg-len-gabor", default=25, type=int)
    parser.add_argument("--seg-len-gaussiandot", default=9, type=int)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if int(args.min_segment_repetitions) < 1:
        print("ERROR: --min-segment-repetitions must be >= 1", file=sys.stderr)
        return 2

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
        output_folder=args.output_folder,
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
        min_segment_repetitions=args.min_segment_repetitions,
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
        segment_lengths=seg_lengths,
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
