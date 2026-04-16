#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Within-mouse trial classification using configurable zigzag vectorizations."""

from __future__ import annotations

import argparse
import csv
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    build_vectorization_cache_stem,
    create_vectorization,
    load_labelled_barcodes,
    load_vectorization_cache,
    _discover_mice,
    _eligible_trials,
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
    cache_dir: Optional[Path]
    force_recompute: bool
    n_splits: int
    max_trials: Optional[int]


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


def _fit_within_mouse_classifier(
    xmat: np.ndarray, labels: np.ndarray, n_splits: int
) -> Tuple[Dict[str, float], np.ndarray, List[str], int]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min())
    folds = min(n_splits, min_count)
    if folds < 2:
        # Edge case handling: Mouse has insufficient class samples for stratified CV.
        # This typically occurs when a mouse has data from only one stimulus class.
        # The exception will be caught in run_pipeline() and logged as "FAILED: <message>";
        # the pipeline will continue processing other mice gracefully.
        raise RuntimeError(
            f"Not enough samples per class for CV. min_count={min_count}, requested={n_splits}. "
            f"(Mouse may have only one stimulus class.)"
        )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
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

    scores = cross_validate(
        pipe,
        xmat,
        labels,
        cv=cv,
        scoring={"acc": "accuracy", "f1": "f1_macro"},
    )
    y_pred = cross_val_predict(pipe, xmat, labels, cv=cv)
    label_order = sorted(unique_labels.tolist())
    cm = confusion_matrix(labels, y_pred, labels=label_order)

    metrics = {
        "mean_acc": float(scores["test_acc"].mean()),
        "std_acc": float(scores["test_acc"].std()),
        "mean_f1": float(scores["test_f1"].mean()),
        "std_f1": float(scores["test_f1"].std()),
    }
    return metrics, cm, label_order, folds


def run_pipeline(state: RunState) -> Dict[str, object]:
    output_folder = state.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    figures_dir = output_folder / "figures"
    logs_dir = output_folder / "logs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "run.log"

    with open(log_path, "w", encoding="utf-8") as log_fp, redirect_stdout(log_fp), redirect_stderr(log_fp):
        print("=" * 90)
        print("Within-Mouse Trial Classification")
        print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        print("=" * 90)

        print("\nResolved arguments:")
        print(f"  output_folder:    {state.output_folder}")
        print(f"  data_root:        {state.data_root}")
        print(f"  meta_root:        {state.meta_root}")
        print(f"  p_active:         {state.p_active}")
        print(f"  per_trial_thresh: {state.per_trial_thresh}")
        print(f"  zz_folder:        {state.zz_folder}")
        print(f"  method:           {state.method}")
        print(f"  mice:             {state.mice}")
        print(f"  clip_frames:      {state.clip_frames}")
        if state.cache_dir is None:
            print("  cache_dir:        <data_root>/<mouse>/cache")
        else:
            print(f"  cache_dir:        {state.cache_dir}")
        print(f"  force_recompute:  {state.force_recompute}")
        print(f"  n_splits:         {state.n_splits}")
        print(f"  max_trials:       {state.max_trials}")

        discovered_mice = _discover_mice(state.data_root)
        selected_mice = state.mice if state.mice is not None else discovered_mice
        selected_mice = [m for m in selected_mice if m in discovered_mice]

        if not selected_mice:
            raise RuntimeError("No valid mice selected for classification.")

        print(f"\nDiscovered mice: {len(discovered_mice)}")
        print(f"Selected mice: {len(selected_mice)}")

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
                    print("  No labelled barcode matches found; skipping.")
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

                if cache_path.exists() and not state.force_recompute:
                    cache = load_vectorization_cache(cache_path)
                    if "features" in cache:
                        xmat = np.asarray(cache["features"])
                    elif "X" in cache:
                        xmat = np.asarray(cache["X"])
                    else:
                        raise RuntimeError(f"Cache missing feature matrix: {cache_path}")
                    xmat = np.nan_to_num(xmat)
                    source = "cache"
                else:
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
                    source = "computed"

                metrics, cm, label_order, folds = _fit_within_mouse_classifier(
                    xmat,
                    labels,
                    state.n_splits,
                )

                per_mouse[mouse_name] = {
                    "n_trials": int(len(labels)),
                    "n_features": int(xmat.shape[1]),
                    "clip_frames": int(clip_used),
                    "cv_folds": int(folds),
                    "class_labels": label_order,
                    "class_counts": {
                        label: int((labels == label).sum()) for label in label_order
                    },
                    "source": source,
                    "cache_path": str(cache_path),
                    **metrics,
                }
                confusion_payload[mouse_name] = {
                    "labels": label_order,
                    "cm": cm,
                }

                print(
                    f"  source={source}, trials={len(labels)}, feat={xmat.shape[1]}, "
                    f"acc={metrics['mean_acc']:.3f}+/-{metrics['std_acc']:.3f}, "
                    f"f1={metrics['mean_f1']:.3f}+/-{metrics['std_f1']:.3f}, folds={folds}"
                )
            except Exception as exc:
                print(f"  FAILED: {exc}")
                traceback.print_exc()

        if not per_mouse:
            raise RuntimeError("No mouse produced classification results.")

        mice_order = sorted(per_mouse.keys())

        # Accuracy / F1 overview.
        fig, ax = plt.subplots(figsize=(max(8, len(mice_order) * 1.2), 4.8))
        x = np.arange(len(mice_order))
        width = 0.36
        acc = [float(per_mouse[m]["mean_acc"]) for m in mice_order]
        acc_err = [float(per_mouse[m]["std_acc"]) for m in mice_order]
        f1 = [float(per_mouse[m]["mean_f1"]) for m in mice_order]
        f1_err = [float(per_mouse[m]["std_f1"]) for m in mice_order]

        ax.bar(x - width / 2, acc, width, yerr=acc_err, capsize=3, alpha=0.85, label="Accuracy")
        ax.bar(
            x + width / 2,
            f1,
            width,
            yerr=f1_err,
            capsize=3,
            alpha=0.5,
            hatch="//",
            label="Macro F1",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(mice_order, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Within-mouse classification by mouse ({state.method})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        summary_fig_path = figures_dir / "01_within_mouse_scores.png"
        fig.savefig(summary_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {summary_fig_path}")

        # Confusion matrix panels.
        n = len(mice_order)
        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))
        if hasattr(axes, "flat"):
            axes_flat = list(axes.flat)
        else:
            axes_flat = [axes]

        for ax, mouse_name in zip(axes_flat, mice_order):
            cm = np.asarray(confusion_payload[mouse_name]["cm"], dtype=float)
            labels_order = confusion_payload[mouse_name]["labels"]
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
            ConfusionMatrixDisplay(cm_norm, display_labels=labels_order).plot(
                ax=ax,
                cmap="Blues",
                colorbar=False,
                values_format=".2f",
            )
            ax.set_title(mouse_name, fontsize=9)

        for ax in axes_flat[len(mice_order):]:
            ax.set_visible(False)

        fig.suptitle(f"Normalized confusion matrices ({state.method})", fontsize=12)
        fig.tight_layout()
        cm_fig_path = figures_dir / "02_confusion_matrices.png"
        fig.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {cm_fig_path}")

        summary_json_path = output_folder / "within_mouse_metrics.json"
        summary_csv_path = output_folder / "within_mouse_metrics.csv"

        payload = {
            "method": state.method,
            "p_active": state.p_active,
            "per_trial_thresh": state.per_trial_thresh,
            "zz_folder": state.zz_folder,
            "mice": mice_order,
            "results": per_mouse,
            "figures": [str(summary_fig_path), str(cm_fig_path)],
            "log_path": str(log_path),
            "cache_dir": (
                str(state.cache_dir)
                if state.cache_dir is not None
                else "<data_root>/<mouse>/cache"
            ),
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
                ]
            )
            for mouse_name in mice_order:
                row = per_mouse[mouse_name]
                writer.writerow(
                    [
                        mouse_name,
                        state.method,
                        row["n_trials"],
                        row["n_features"],
                        row["clip_frames"],
                        row["cv_folds"],
                        row["mean_acc"],
                        row["std_acc"],
                        row["mean_f1"],
                        row["std_f1"],
                        row["source"],
                        row["cache_path"],
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
            "Classify trials within each mouse using a selected vectorization method. "
            "Loads cached vectorizations when available, otherwise computes and saves them."
        )
    )
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)
    parser.add_argument("--p-active", required=True, type=int)
    parser.add_argument("--per-trial-thresh", required=True, type=_str2bool)
    parser.add_argument("--method", default="Turnover")
    parser.add_argument(
        "--mice",
        default=None,
        type=_opt_csv_list,
        help="Optional comma-separated mouse names. Default: all discovered mice.",
    )
    parser.add_argument(
        "--clip-frames",
        default=None,
        type=_opt_int,
        help="Optional clip length. Default: min(valid_frames) per mouse.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Directory for .npz vectorization caches. Default: <data-root>/<mouse>/cache",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute vectorizations even if cache files already exist.",
    )
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--max-trials", default=None, type=_opt_int)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir

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
        cache_dir=cache_dir,
        force_recompute=args.force_recompute,
        n_splits=args.n_splits,
        max_trials=args.max_trials,
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
