#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared machinery for the classification entry points.

- command-line arguments shared by all entry points
- model registry and per-fold training (vector models and grid models)
- seeding (re-seeded before every fit) and optional deterministic GPU mode
- evaluation of :class:`classification_data.EvalUnit` problems
- results layout and output files

Results layout (``<results_root>`` defaults to ``<project>/results``)::

    zigzag/<vectorization>/<model>/p<P>-<per-trial|global>/<target>_<scope>/<run>/repeat-NN/
    grid/<model>/<target>_<scope>/<run>/repeat-NN/
    ablation_shuffle/zigzag/<vectorization>/<model>/<shuffle>-<mode>/p<P>-<...>/<target>_<scope>/<run>/shuffle-NNNN/
    ablation_shuffle/grid/<model>/<shuffle>-<mode>/<target>_<scope>/<run>/shuffle-NNNN/

Each ``repeat-NN`` / ``shuffle-NNNN`` folder contains ``run_config.json``,
``metrics.csv`` (one row per classification problem), ``fold_metrics.csv``,
``predictions.csv`` (one row per test sample), ``skips.csv``,
``figures/summary.png`` and ``run.log``.  ``results_index.py`` loads them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from classification_data import (
    CV_SCHEMES,
    DEFAULT_SEGMENT_LENGTHS,
    SCOPES,
    TARGETS,
    EvalUnit,
    ProblemConfig,
)
from utils import _opt_csv_list, _opt_int, _str2bool


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_DIR / "results"

VECTOR_MODELS: Tuple[str, ...] = ("logreg", "mlp", "cnn1d")
GRID_MODELS: Tuple[str, ...] = ("cnn3d_raw", "cnn3d_norm")
MODEL_TITLES = {
    "logreg": "LogReg",
    "mlp": "MLP",
    "cnn1d": "1D-CNN",
    "cnn3d_raw": "3D-CNN (raw grid)",
    "cnn3d_norm": "3D-CNN (normalized grid)",
}


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def add_problem_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("classification problem")
    g.add_argument("--target", required=True, choices=TARGETS)
    g.add_argument("--scope", required=True, choices=SCOPES)
    g.add_argument("--data-root", required=True, type=Path)
    g.add_argument("--meta-root", required=True, type=Path)
    g.add_argument("--grid-subdir", default="trials_grid")
    g.add_argument("--mice", default=None, type=_opt_csv_list, help="Comma-separated mice (default: all)")
    g.add_argument(
        "--clip-frames",
        default=None,
        type=_opt_int,
        help="Override the frame window of video_label/video_id (default: shortest trial rule)",
    )
    g.add_argument(
        "--cv-scheme",
        default=None,
        choices=CV_SCHEMES,
        help="Within-mouse CV (default: stratified for video_label, logo for video_id/segment_id)",
    )
    g.add_argument("--n-splits", default=5, type=int)
    g.add_argument("--split-seed", default=42, type=int, help="Seed of the CV fold assignment")
    g.add_argument(
        "--min-repetitions",
        default=None,
        type=_opt_int,
        help="Minimum repetitions per video/segment ID (default: video_id 7 within / 5 cross; "
        "segment_id 1 within / 7 cross)",
    )
    for label, length in DEFAULT_SEGMENT_LENGTHS.items():
        g.add_argument(f"--seg-len-{label.lower()}", default=length, type=int)
    g.add_argument("--max-trials-per-mouse", default=None, type=_opt_int, help="Random subset for quick tests")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("training")
    g.add_argument("--batch-size-vec", default=64, type=int)
    g.add_argument("--batch-size-grid", default=16, type=int)
    g.add_argument("--epochs-mlp", default=60, type=int)
    g.add_argument("--epochs-cnn1d", default=60, type=int)
    g.add_argument("--epochs-cnn3d", default=40, type=int)
    g.add_argument("--lr-vec", default=1e-3, type=float)
    g.add_argument("--lr-cnn3d", default=5e-4, type=float)
    g.add_argument("--weight-decay", default=1e-4, type=float)
    g.add_argument("--early-stop-patience", default=10, type=int)
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    g.add_argument("--num-workers-dl", default=0, type=int)
    g.add_argument(
        "--deterministic",
        default=False,
        type=_str2bool,
        help="Use deterministic GPU algorithms (slower, bit-reproducible on the same hardware)",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("output")
    g.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT, type=Path)
    g.add_argument(
        "--run-name",
        default=None,
        help="Run folder name (default: timestamp). Array jobs pass the same name to share one run folder.",
    )
    g.add_argument("--output-dir", default=None, type=Path, help="Write here instead of the standard layout")


def problem_config_from_args(args: argparse.Namespace) -> ProblemConfig:
    segment_lengths = {label: int(getattr(args, f"seg_len_{label.lower()}")) for label in DEFAULT_SEGMENT_LENGTHS}
    return ProblemConfig(
        target=args.target,
        scope=args.scope,
        data_root=args.data_root,
        meta_root=args.meta_root,
        grid_subdir=args.grid_subdir,
        mice=args.mice,
        clip_frames=args.clip_frames,
        cv_scheme=args.cv_scheme,
        n_splits=args.n_splits,
        split_seed=args.split_seed,
        min_repetitions=args.min_repetitions,
        segment_lengths=segment_lengths,
        max_trials_per_mouse=args.max_trials_per_mouse,
    )


@dataclass
class TrainConfig:
    batch_size_vec: int = 64
    batch_size_grid: int = 16
    epochs_mlp: int = 60
    epochs_cnn1d: int = 60
    epochs_cnn3d: int = 40
    lr_vec: float = 1e-3
    lr_cnn3d: float = 5e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    device: str = "auto"
    num_workers_dl: int = 0
    deterministic: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        return cls(**{k: getattr(args, k) for k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Seeding / device
# ---------------------------------------------------------------------------

def stable_seed(base_seed: int, token: str) -> int:
    digest = hashlib.md5(f"{int(base_seed)}|{token}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(train_cfg: TrainConfig):
    """Resolve the device and apply the determinism setting. Returns torch.device."""
    if train_cfg.deterministic:
        # Must be set before the first cuBLAS call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if train_cfg.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if train_cfg.device == "cpu":
        return torch.device("cpu")
    if train_cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _constant_prediction(y_train: np.ndarray, n: int) -> np.ndarray:
    values, counts = np.unique(y_train, return_counts=True)
    return np.full(n, int(values[np.argmax(counts)]), dtype=np.int64)


def fit_predict_vector(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_classes: int,
    window_frames: int,
    seed: int,
    train_cfg: TrainConfig,
    device: Any,
) -> np.ndarray:
    """Train ``model`` on vector features ``x[train_idx]`` and predict ``x[test_idx]``."""
    from sklearn.preprocessing import StandardScaler

    from classification_models import CNN1D, MLP, VectorDataset, infer_cnn1d_shape, train_eval_logreg, train_eval_nn

    x_tr, y_tr = x[train_idx], y[train_idx]
    x_te, y_te = x[test_idx], y[test_idx]
    if model == "logreg":
        _m, pred = train_eval_logreg(x_tr, y_tr, x_te, y_te)
        return np.asarray(pred, dtype=np.int64)

    scaler = StandardScaler().fit(x_tr)
    x_tr = scaler.transform(x_tr)
    x_te = scaler.transform(x_te)
    dim = int(x.shape[1])
    if model == "mlp":
        make_model = lambda: MLP(n_classes=n_classes, input_dim=dim)  # noqa: E731
        epochs, lr = train_cfg.epochs_mlp, train_cfg.lr_vec
    elif model == "cnn1d":
        in_ch, seq_len = infer_cnn1d_shape(dim, int(window_frames))
        make_model = lambda: CNN1D(n_classes=n_classes, in_channels=in_ch, seq_len=seq_len)  # noqa: E731
        epochs, lr = train_cfg.epochs_cnn1d, train_cfg.lr_vec
    else:
        raise ValueError(f"Unknown vector model '{model}'. Use one of {VECTOR_MODELS}")

    set_all_seeds(seed)
    _m, pred = train_eval_nn(
        make_model=make_model,
        train_ds=VectorDataset(x_tr, y_tr),
        y_train=y_tr,
        test_ds=VectorDataset(x_te, y_te),
        test_y=y_te,
        epochs=epochs,
        lr=lr,
        batch_size=train_cfg.batch_size_vec,
        patience=train_cfg.early_stop_patience,
        weight_decay=train_cfg.weight_decay,
        device=device,
        num_workers=train_cfg.num_workers_dl,
        seed=seed,
    )
    return np.asarray(pred, dtype=np.int64)


def fit_predict_grid(
    model: str,
    dataset: Any,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    train_cfg: TrainConfig,
    device: Any,
) -> np.ndarray:
    """Train a 3D-CNN on ``Subset(dataset, train_idx)`` and predict ``test_idx``.

    Early stopping and learning-rate scheduling use a validation split taken
    from the training indices only; the test indices are never seen in training.
    """
    from torch.utils.data import Subset

    from classification_models import CNN3D, train_eval_nn

    if model not in GRID_MODELS:
        raise ValueError(f"Unknown grid model '{model}'. Use one of {GRID_MODELS}")
    in_ch = int(dataset.in_channels)
    set_all_seeds(seed)
    _m, pred = train_eval_nn(
        make_model=lambda: CNN3D(n_classes=n_classes, in_channels=in_ch),
        train_ds=Subset(dataset, train_idx),
        y_train=y[train_idx],
        test_ds=Subset(dataset, test_idx),
        test_y=y[test_idx],
        epochs=train_cfg.epochs_cnn3d,
        lr=train_cfg.lr_cnn3d,
        batch_size=train_cfg.batch_size_grid,
        patience=train_cfg.early_stop_patience,
        weight_decay=train_cfg.weight_decay,
        device=device,
        num_workers=train_cfg.num_workers_dl,
        seed=seed,
    )
    return np.asarray(pred, dtype=np.int64)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

FitFn = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    import warnings

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    # Leave-one-trial-out folds hold one trial (often one class), which makes
    # sklearn warn about single labels / predicted classes absent from y_true.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


def evaluate_unit(
    unit: EvalUnit,
    fit_fn: FitFn,
    *,
    init_seed: int,
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """Run all splits of ``unit``.

    ``fit_fn(train_idx, test_idx, seed)`` returns predictions for ``test_idx``.
    The seed of each fit is derived from ``init_seed``, the unit and the fold,
    so results do not depend on which problems ran before.

    Metrics are computed on the pooled test predictions (``accuracy``,
    ``balanced_accuracy``, ``macro_f1``) and also per fold (``fold_*`` mean/std).
    """
    y = unit.y
    pred = np.full(len(y), -1, dtype=np.int64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    fold_rows: List[Dict[str, Any]] = []
    n_folds = len(unit.splits)
    for k, (tr, te) in enumerate(unit.splits):
        if len(np.unique(y[tr])) < 2:
            fold_pred = _constant_prediction(y[tr], len(te))
        else:
            fold_pred = fit_fn(tr, te, stable_seed(init_seed, f"{unit.unit_id}|fold={k}"))
        pred[te] = fold_pred
        fold_ids[te] = k
        fold_rows.append({"unit_id": unit.unit_id, "fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), **_scores(y[te], fold_pred)})
        if n_folds <= 20 or (k + 1) % max(1, n_folds // 10) == 0 or k + 1 == n_folds:
            log(f"    fold {k + 1}/{n_folds}: acc={fold_rows[-1]['accuracy']:.3f}")

    test_idx = unit.test_index
    pooled = _scores(y[test_idx], pred[test_idx])
    fold_acc = np.asarray([r["accuracy"] for r in fold_rows])
    fold_f1 = np.asarray([r["macro_f1"] for r in fold_rows])
    n_train = int(len(unit.splits[0][0])) if unit.cv_scheme == "train_test" else None

    summary = {
        "unit_id": unit.unit_id,
        "label": unit.label,
        "mouse": unit.mouse,
        "train_mice": "|".join(unit.train_mice),
        "n_samples": int(len(y)),
        "n_train": n_train,
        "n_test": int(len(test_idx)),
        "n_trials": int(unit.samples["trial_id"].nunique()) if unit.scope == "within_mouse" else None,
        "n_classes": unit.n_classes,
        "classes": "|".join(unit.classes),
        "chance": 1.0 / unit.n_classes,
        "cv_scheme": unit.cv_scheme,
        "n_folds": n_folds,
        "window_frames": unit.window_frames,
        **pooled,
        "fold_accuracy_mean": float(fold_acc.mean()),
        "fold_accuracy_std": float(fold_acc.std()),
        "fold_macro_f1_mean": float(fold_f1.mean()),
        "fold_macro_f1_std": float(fold_f1.std()),
    }

    s = unit.samples.iloc[test_idx]
    predictions = pd.DataFrame(
        {
            "unit_id": unit.unit_id,
            "fold": fold_ids[test_idx],
            "mouse": s["mouse"].to_numpy(),
            "trial_id": s["trial_id"].to_numpy(),
            "segment_id": s["segment_id"].to_numpy() if "segment_id" in s else "",
            "start_frame": s["start_frame"].to_numpy(),
            "n_frames": s["n_frames"].to_numpy(),
            "y_true": [unit.classes[i] for i in y[test_idx]],
            "y_pred": [unit.classes[i] for i in pred[test_idx]],
        }
    )
    return summary, fold_rows, predictions


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def thresh_tag(p_active: int, per_trial_thresh: bool) -> str:
    return f"p{int(p_active)}-{'per-trial' if per_trial_thresh else 'global'}"


def default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


class Tee:
    """Mirror stdout/stderr into a log file."""

    def __init__(self, path: Path) -> None:
        self._fp = open(path, "a", encoding="utf-8")
        self._stdout, self._stderr = sys.stdout, sys.stderr

    def __enter__(self) -> "Tee":
        outer = self

        class _Stream:
            def __init__(self, base):
                self.base = base

            def write(self, data):
                self.base.write(data)
                outer._fp.write(data)
                return len(data)

            def flush(self):
                self.base.flush()
                outer._fp.flush()

        sys.stdout, sys.stderr = _Stream(self._stdout), _Stream(self._stderr)
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._fp.close()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=_json_default)


def plot_summary(metrics: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metrics.empty:
        return
    names = [u.replace("mouse=", "").replace("test=", "") for u in metrics["unit_id"]]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(names) + 3), 7.5), sharex=True)
    for ax, col in zip(axes, ["accuracy", "macro_f1"]):
        ax.bar(x, metrics[col], color="#4C72B0", alpha=0.85)
        ax.scatter(x, metrics["chance"], color="k", marker="_", s=120, label="chance")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title(title, fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=60, ha="right", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    *,
    context: Dict[str, Any],
    config: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    fold_rows: List[Dict[str, Any]],
    predictions: List[pd.DataFrame],
    skips: List[Dict[str, Any]],
    title: str,
) -> None:
    """Write metrics/fold_metrics/predictions/skips CSVs, config and a summary figure.

    ``context`` holds run-level columns (target, scope, input, model, ...) that
    are prepended to every CSV so files can be concatenated without parsing paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    def _with_context(df: pd.DataFrame) -> pd.DataFrame:
        for i, (k, v) in enumerate(context.items()):
            df.insert(i, k, v)
        return df

    metrics = _with_context(pd.DataFrame(summaries))
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    _with_context(pd.DataFrame(fold_rows)).to_csv(out_dir / "fold_metrics.csv", index=False)
    preds = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    _with_context(preds).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(skips, columns=["unit_id", "mouse", "label", "reason", "detail"]).to_csv(out_dir / "skips.csv", index=False)
    write_json(out_dir / "run_config.json", config)
    plot_summary(metrics, out_dir / "figures" / "summary.png", title)


def base_config(args: argparse.Namespace, problem_cfg: ProblemConfig, train_cfg: Optional[TrainConfig]) -> Dict[str, Any]:
    return {
        "script": Path(sys.argv[0]).name,
        "argv": sys.argv[1:],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "problem": problem_cfg.to_dict(),
        "training": asdict(train_cfg) if train_cfg is not None else None,
    }


def run_vector_classification(
    args: argparse.Namespace,
    *,
    problems: Any,
    problem_cfg: ProblemConfig,
    train_cfg: TrainConfig,
    run_dir: Path,
    make_features: Callable[[], Any],
    context: Dict[str, Any],
    title: str,
    transform: Optional[Callable[[np.ndarray, EvalUnit], np.ndarray]] = None,
) -> int:
    """Repeat loop shared by the vector-feature entry points (zigzag, grid stats).

    ``make_features()`` returns an object with ``features(samples)`` and a
    ``sources`` dict; ``transform(x, unit)`` optionally post-processes the
    feature matrix of a problem (e.g. averaging over time).
    Uses ``args.model``, ``args.n_repeats``, ``args.repeat_start`` and ``args.init_seed``.
    """
    import traceback

    device = configure_torch(train_cfg) if args.model != "logreg" else None
    for repeat in range(args.repeat_start, args.repeat_start + args.n_repeats):
        out_dir = run_dir / f"repeat-{repeat:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        init_seed = args.init_seed + repeat
        with Tee(out_dir / "run.log"):
            try:
                print(f"{title} | repeat {repeat}")
                print(f"output: {out_dir}")
                print(f"problems: {len(problems.units)} (skipped {len(problems.skips)}), notes={problems.notes}")
                features = make_features()

                def fit_fn_for_unit(unit: EvalUnit) -> FitFn:
                    x = features.features(unit.samples)
                    if transform is not None:
                        x = transform(x, unit)
                    return lambda tr, te, seed: fit_predict_vector(
                        args.model,
                        x,
                        unit.y,
                        tr,
                        te,
                        n_classes=unit.n_classes,
                        window_frames=unit.window_frames,
                        seed=seed,
                        train_cfg=train_cfg,
                        device=device,
                    )

                summaries, folds, preds = run_units(problems.units, fit_fn_for_unit, init_seed=init_seed)
                ctx = {**context, "repeat": repeat, "init_seed": init_seed, "split_seed": args.split_seed}
                config = base_config(args, problem_cfg, train_cfg if args.model != "logreg" else None)
                config.update(
                    {
                        "context": ctx,
                        "notes": problems.notes,
                        "mice_used": problems.mice_used,
                        "feature_caches": features.sources,
                    }
                )
                write_outputs(
                    out_dir,
                    context=ctx,
                    config=config,
                    summaries=summaries,
                    fold_rows=folds,
                    predictions=preds,
                    skips=problems.skips,
                    title=title,
                )
                print(f"\nDone: {out_dir}")
            except Exception:
                traceback.print_exc()
                return 1
    return 0


def run_units(
    units: Sequence[EvalUnit],
    fit_fn_for_unit: Callable[[EvalUnit], FitFn],
    *,
    init_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[pd.DataFrame]]:
    summaries: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    predictions: List[pd.DataFrame] = []
    for i, unit in enumerate(units, start=1):
        print(
            f"\n[{i}/{len(units)}] {unit.unit_id}: n={len(unit.y)}, classes={unit.n_classes}, "
            f"folds={len(unit.splits)} ({unit.cv_scheme}), window={unit.window_frames}",
            flush=True,
        )
        summary, folds, preds = evaluate_unit(unit, fit_fn_for_unit(unit), init_seed=init_seed)
        print(
            f"  -> acc={summary['accuracy']:.3f} macro_f1={summary['macro_f1']:.3f} "
            f"(chance={summary['chance']:.3f})",
            flush=True,
        )
        summaries.append(summary)
        fold_rows.extend(folds)
        predictions.append(preds)
    return summaries, fold_rows, predictions
