#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild ablation confusion-matrix figure from saved JSON artifacts.

This utility regenerates the Figure 3 confusion-matrix panel saved by:
- scripts/classify_trials_within_mouse_ablation.py
- scripts/classify_trials_cross_mouse_ablation.py

It reads either:
1) a summary JSON containing `confusion_matrices_path`, or
2) a confusion JSON directly.

Usage examples:

Within-mouse (from summary JSON)
python rebuild_confusion_matrix_figure.py --summary-json results/within_mouse_classification_ablation/p30-per-trial/within_mouse_ablation_metrics.json

Cross-mouse (from summary JSON)
python rebuild_confusion_matrix_figure.py --summary-json results/cross_mouse_classification_ablation/p30-per-trial/cross_mouse_metrics.json

Direct confusion JSON + explicit output
python rebuild_confusion_matrix_figure.py --confusion-json results/within_mouse_classification_ablation/p30-per-trial/within_mouse_confusion_matrices.json --output results/within_mouse_classification_ablation/p30-per-trial/figures/03_all_classifier_confusion_matrices_regenerated.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


MODEL_ORDER = ["logreg", "mlp", "cnn1d", "cnn3d"]
MODEL_TITLES = {
    "logreg": "LogReg (vector)",
    "mlp": "MLP (vector)",
    "cnn1d": "1D-CNN (vector)",
    "cnn3d": "3D-CNN (grid)",
}


def _short_mouse_name(name: str) -> str:
    m = re.match(r"dynamic(\d+)-(\d+)-(\d+)", name)
    if m:
        return f"rec-{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return name


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _resolve_confusion_json(summary_json: Path | None, confusion_json: Path | None) -> Path:
    if confusion_json is not None:
        if not confusion_json.exists():
            raise FileNotFoundError(f"Confusion JSON not found: {confusion_json}")
        return confusion_json

    if summary_json is None:
        raise ValueError("Provide either --summary-json or --confusion-json")
    if not summary_json.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_json}")

    summary_payload = _load_json(summary_json)
    rel_or_abs = summary_payload.get("confusion_matrices_path")
    if not isinstance(rel_or_abs, str) or not rel_or_abs:
        raise KeyError(
            f"Summary JSON has no valid confusion_matrices_path: {summary_json}"
        )

    candidate = Path(rel_or_abs)
    if not candidate.is_absolute():
        candidate = (summary_json.parent / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Resolved confusion JSON not found: {candidate}")
    return candidate


def _to_normalized_cm(mouse_payload: Dict[str, object], model_name: str) -> np.ndarray:
    if "cms_normalized" in mouse_payload:
        cms_norm = mouse_payload["cms_normalized"]
        if isinstance(cms_norm, dict) and model_name in cms_norm:
            return np.asarray(cms_norm[model_name], dtype=float)

    if "cms_counts" in mouse_payload:
        cms_counts = mouse_payload["cms_counts"]
        if isinstance(cms_counts, dict) and model_name in cms_counts:
            cm = np.asarray(cms_counts[model_name], dtype=np.int64)
            row_sums = cm.sum(axis=1, keepdims=True)
            return np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Backward-compatibility with in-memory-like schema: {"cms": {model: matrix}}
    if "cms" in mouse_payload:
        cms = mouse_payload["cms"]
        if isinstance(cms, dict) and model_name in cms:
            cm = np.asarray(cms[model_name], dtype=float)
            row_sums = cm.sum(axis=1, keepdims=True)
            return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    raise KeyError(f"Model '{model_name}' confusion matrix not found in payload")


def _infer_mode(mode: str, confusion_json_path: Path) -> str:
    if mode in {"within", "cross"}:
        return mode
    name = confusion_json_path.name.lower()
    if "within" in name:
        return "within"
    if "cross" in name:
        return "cross"
    return "within"


def _default_output_path(confusion_json_path: Path) -> Path:
    return confusion_json_path.parent / "03_all_classifier_confusion_matrices_regenerated.png"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the all-classifier confusion-matrix figure from saved JSON artifacts."
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--confusion-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "within", "cross"],
        default="auto",
        help="Figure title mode. 'auto' infers from input filename.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.summary_json is None and args.confusion_json is None:
        parser.error("One of --summary-json or --confusion-json is required")

    confusion_json_path = _resolve_confusion_json(args.summary_json, args.confusion_json)
    payload = _load_json(confusion_json_path)
    if not isinstance(payload, dict) or len(payload) == 0:
        raise RuntimeError(f"Confusion payload is empty: {confusion_json_path}")

    mode = _infer_mode(args.mode, confusion_json_path)
    row_keys = sorted(payload.keys())

    n_rows = len(row_keys)
    n_cols = len(MODEL_ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for row_idx, row_name in enumerate(row_keys):
        row_payload = payload[row_name]
        if not isinstance(row_payload, dict):
            raise RuntimeError(f"Invalid payload shape for key: {row_name}")

        labels = row_payload.get("labels")
        if not isinstance(labels, list) or len(labels) == 0:
            raise RuntimeError(f"Missing labels for key: {row_name}")
        labels = [str(v) for v in labels]
        best_model = str(row_payload.get("best_model", ""))

        for col_idx, model_name in enumerate(MODEL_ORDER):
            ax = axes[row_idx][col_idx]
            cm_norm = _to_normalized_cm(row_payload, model_name)
            ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
                ax=ax,
                cmap="Blues",
                colorbar=False,
                values_format=".2f",
            )
            title = f"{_short_mouse_name(row_name)}\n{MODEL_TITLES[model_name]}"
            if model_name == best_model:
                title += " ★"
            ax.set_title(title, fontsize=7)

    if mode == "cross":
        fig.suptitle("Normalized confusion matrices - all classifiers per test mouse (star = best)", fontsize=11)
    else:
        fig.suptitle("Normalized confusion matrices - all classifiers per mouse (star = best)", fontsize=11)

    fig.tight_layout()
    output_path = args.output if args.output is not None else _default_output_path(confusion_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion JSON: {confusion_json_path}")
    print(f"Mode: {mode}")
    print(f"Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
