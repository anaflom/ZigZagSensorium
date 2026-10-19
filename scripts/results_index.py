#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load classification results written by classify_zigzag.py, classify_grid.py
and the ablation_shuffle_* scripts, for plotting and comparisons.

Every repeat/shuffle folder holds a ``metrics.csv`` whose first columns
describe the run (target, scope, input, vectorization, model, p_active,
thresh_mode, repeat or shuffle_id, seeds), so results are combined by
concatenating files and filtering columns, never by parsing folder names.

Examples
--------
>>> from results_index import load_metrics, summarize_repeats
>>> df = load_metrics(target="video_label", scope="cross_mouse")           # zigzag + grid
>>> grid = load_metrics(input="grid", model="cnn3d_raw", target="video_id")
>>> summarize_repeats(grid)                                                  # mean/std over repeats
>>> load_metrics(kind="ablation_shuffle", shuffle_type="time")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "results"

KIND_DIRS = {
    "classification": ("zigzag", "grid"),
    "ablation_shuffle": ("ablation_shuffle",),
}


def find_result_dirs(results_root: Path = DEFAULT_RESULTS_ROOT, kind: str = "classification") -> List[Path]:
    """All repeat-NN / shuffle-NNNN folders that contain a metrics.csv."""
    if kind not in KIND_DIRS:
        raise ValueError(f"kind must be one of {list(KIND_DIRS)}")
    out: List[Path] = []
    for top in KIND_DIRS[kind]:
        base = Path(results_root) / top
        if base.exists():
            out.extend(sorted(p.parent for p in base.rglob("metrics.csv")))
    return out


def _read(dirs: Iterable[Path], filename: str) -> pd.DataFrame:
    frames = []
    for d in dirs:
        path = d / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df.insert(0, "run_dir", str(d.parent))
        df.insert(1, "result_dir", str(d))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _filter(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    for col, value in filters.items():
        if value is None or df.empty:
            continue
        if col not in df.columns:
            raise KeyError(f"Unknown filter column '{col}'. Available: {list(df.columns)}")
        values = value if isinstance(value, (list, tuple, set)) else [value]
        df = df[df[col].isin(values)]
    return df


def load_metrics(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    kind: str = "classification",
    latest_run_only: bool = False,
    **filters: Any,
) -> pd.DataFrame:
    """Concatenate metrics.csv files (one row per classification problem).

    ``filters`` are column=value (or list of values), e.g. ``target="video_id"``,
    ``input="grid"``, ``p_active=[30, 70]``.  With ``latest_run_only`` only the
    most recent run folder is kept for each (input, vectorization, model,
    p_active, thresh_mode, target, scope) combination.
    """
    df = _filter(_read(find_result_dirs(results_root, kind), "metrics.csv"), filters)
    if latest_run_only and not df.empty:
        keys = ["input", "vectorization", "model", "p_active", "thresh_mode", "target", "scope"]
        keys = [k for k in keys if k in df.columns]
        latest = df.groupby(keys, dropna=False)["run_dir"].transform("max")
        df = df[df["run_dir"] == latest]
    return df.reset_index(drop=True)


def load_fold_metrics(results_root: Path = DEFAULT_RESULTS_ROOT, kind: str = "classification", **filters: Any) -> pd.DataFrame:
    return _filter(_read(find_result_dirs(results_root, kind), "fold_metrics.csv"), filters).reset_index(drop=True)


def load_predictions(results_root: Path = DEFAULT_RESULTS_ROOT, kind: str = "classification", **filters: Any) -> pd.DataFrame:
    return _filter(_read(find_result_dirs(results_root, kind), "predictions.csv"), filters).reset_index(drop=True)


def load_run_config(result_dir: Path) -> Dict[str, Any]:
    with open(Path(result_dir) / "run_config.json", encoding="utf-8") as fp:
        return json.load(fp)


def summarize_repeats(
    metrics: pd.DataFrame,
    over: str = "repeat",
    values: Sequence[str] = ("accuracy", "balanced_accuracy", "macro_f1"),
) -> pd.DataFrame:
    """Mean, std and count of ``values`` over repeats (or ``over="shuffle_id"``) per problem."""
    keys = [
        c
        for c in ["input", "vectorization", "model", "p_active", "thresh_mode", "shuffle_type", "shuffle_mode",
                  "target", "scope", "run_dir", "unit_id", "label", "mouse", "chance"]
        if c in metrics.columns
    ]
    agg = metrics.groupby(keys, dropna=False)[list(values)].agg(["mean", "std", "count"])
    agg.columns = [f"{v}_{s}" for v, s in agg.columns]
    return agg.reset_index().rename(columns={f"{values[0]}_count": f"n_{over}s"}).drop(
        columns=[f"{v}_count" for v in values[1:]]
    )


def confusion_matrix(predictions: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Confusion matrix (rows = true class) from a predictions table of one problem."""
    classes = sorted(set(predictions["y_true"].astype(str)) | set(predictions["y_pred"].astype(str)))
    cm = pd.crosstab(predictions["y_true"].astype(str), predictions["y_pred"].astype(str)).reindex(
        index=classes, columns=classes, fill_value=0
    )
    if normalize:
        cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return cm


def main() -> int:
    p = argparse.ArgumentParser(description="List classification results")
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT, type=Path)
    p.add_argument("--kind", default="classification", choices=list(KIND_DIRS))
    args = p.parse_args()
    df = load_metrics(args.results_root, args.kind)
    if df.empty:
        print("No results found.")
        return 0
    cols = [c for c in ["input", "vectorization", "model", "p_active", "thresh_mode", "target", "scope", "run_dir"] if c in df.columns]
    print(df.groupby(cols, dropna=False).size().rename("n_rows").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
