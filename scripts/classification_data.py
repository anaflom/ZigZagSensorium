#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Input-independent definition of the classification problems.

This module decides *what* is classified: which trials or segments, which
classes, which frame window and which train/test splits.  It never touches
zigzag features or grid arrays, so the zigzag and grid entry points
(``classify_zigzag.py`` / ``classify_grid.py`` and the shuffle ablations) build
identical problems from the same arguments.

Targets
-------
video_label  Classify the stimulus label (NaturalVideo, PinkNoise, ...) of a
             trial.  Window: first ``clip`` frames, where ``clip`` is the
             shortest trial among the mice used (240 with the current data).
video_id     Classify the video ID within one stimulus label.  Window: first
             ``clip`` frames, where ``clip`` is the shortest trial among the
             trials included in that classification problem.
segment_id   Classify the segment ID within one stimulus label.  Window: the
             segment's own frames (start_frame, seg_length).

Scopes
------
within_mouse  Cross-validation inside each mouse.
cross_mouse   Train on other mice, test on a held-out mouse (leave-one-mouse-out),
              or, for video_id, on directed mouse pairs (train on A, test on B).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedKFold

from utils import (
    _discover_mice,
    _eligible_trials,
    _normalize_video_id_token,
    build_segment_sample_records,
    load_labelled_grid_paths,
    load_trial_metadata,
)


TARGETS: Tuple[str, ...] = ("video_label", "video_id", "segment_id")
SCOPES: Tuple[str, ...] = ("within_mouse", "cross_mouse")

VIDEO_ID_LABELS: Tuple[str, ...] = (
    "NaturalVideo",
    "NaturalImages",
    "PinkNoise",
    "RandomDots",
    "Gabor",
    "GaussianDot",
)
SEGMENT_ID_LABELS: Tuple[str, ...] = (
    "NaturalImages",
    "PinkNoise",
    "RandomDots",
    "Gabor",
    "GaussianDot",
)
DEFAULT_SEGMENT_LENGTHS: Dict[str, int] = {
    "NaturalImages": 15,
    "PinkNoise": 27,
    "RandomDots": 60,
    "Gabor": 25,
    "GaussianDot": 9,
}

CV_SCHEMES: Tuple[str, ...] = ("stratified", "logo", "loso", "groupkfold", "kfold")
# Default CV per target for within-mouse problems (matches the previous scripts).
DEFAULT_WITHIN_CV: Dict[str, str] = {
    "video_label": "stratified",
    "video_id": "logo",
    "segment_id": "logo",
}
# Default minimum number of repetitions of a class (video ID / segment ID).
DEFAULT_MIN_REPETITIONS: Dict[Tuple[str, str], int] = {
    ("video_id", "within_mouse"): 7,
    ("video_id", "cross_mouse"): 5,
    ("segment_id", "within_mouse"): 1,
    ("segment_id", "cross_mouse"): 7,
}


@dataclass
class ProblemConfig:
    """Arguments that define the classification problems of a run."""

    target: str
    scope: str
    data_root: Path
    meta_root: Path
    grid_subdir: str = "trials_grid"
    mice: Optional[List[str]] = None
    clip_frames: Optional[int] = None
    cv_scheme: Optional[str] = None
    n_splits: int = 5
    split_seed: int = 42
    min_repetitions: Optional[int] = None
    segment_lengths: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SEGMENT_LENGTHS))
    max_trials_per_mouse: Optional[int] = None

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"Unknown target '{self.target}'. Use one of {TARGETS}")
        if self.scope not in SCOPES:
            raise ValueError(f"Unknown scope '{self.scope}'. Use one of {SCOPES}")
        if self.cv_scheme is None and self.scope == "within_mouse":
            self.cv_scheme = DEFAULT_WITHIN_CV[self.target]
        if self.cv_scheme is not None and self.cv_scheme not in CV_SCHEMES:
            raise ValueError(f"Unknown cv scheme '{self.cv_scheme}'. Use one of {CV_SCHEMES}")
        if self.min_repetitions is None and self.target != "video_label":
            self.min_repetitions = DEFAULT_MIN_REPETITIONS[(self.target, self.scope)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "scope": self.scope,
            "data_root": str(self.data_root),
            "meta_root": str(self.meta_root),
            "grid_subdir": self.grid_subdir,
            "mice": self.mice,
            "clip_frames_override": self.clip_frames,
            "cv_scheme": self.cv_scheme if self.scope == "within_mouse" else "train_test",
            "n_splits": self.n_splits,
            "split_seed": self.split_seed,
            "min_repetitions": self.min_repetitions,
            "segment_lengths": dict(self.segment_lengths) if self.target == "segment_id" else None,
            "max_trials_per_mouse": self.max_trials_per_mouse,
        }


@dataclass
class EvalUnit:
    """One classification problem: samples, classes and train/test splits.

    ``samples`` has one row per sample with at least the columns
    ``mouse, trial_id, label, class_token, start_frame, n_frames,
    valid_frames, grid_path`` (plus ``segment_id``/``segment_index`` for
    segments).  ``splits`` index rows of ``samples``.
    """

    unit_id: str
    target: str
    scope: str
    label: str
    mouse: str
    train_mice: List[str]
    samples: pd.DataFrame
    classes: List[str]
    y: np.ndarray
    splits: List[Tuple[np.ndarray, np.ndarray]]
    cv_scheme: str
    window_frames: int

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def test_index(self) -> np.ndarray:
        return np.concatenate([te for _tr, te in self.splits]) if self.splits else np.zeros(0, dtype=np.int64)


@dataclass
class ProblemSet:
    units: List[EvalUnit]
    skips: List[Dict[str, Any]]
    mice_used: List[str]
    notes: Dict[str, Any]


# ---------------------------------------------------------------------------
# Trial tables
# ---------------------------------------------------------------------------

def load_mouse_trials(
    data_root: Path,
    meta_root: Path,
    mouse_name: str,
    grid_subdir: str = "trials_grid",
) -> pd.DataFrame:
    """Eligible trials of one mouse that have a grid file, in grid-file order.

    The order (sorted grid filenames) is the same as the order of trials in the
    vectorization caches, which keeps fold assignments identical to the
    previous scripts.
    """
    paths, labels, trial_ids, valid_frames = load_labelled_grid_paths(
        data_root, meta_root, mouse_name, grid_subdir=grid_subdir
    )
    columns = ["mouse", "trial_id", "label", "video_id", "valid_frames", "grid_path"]
    if len(paths) == 0:
        return pd.DataFrame(columns=columns)

    df_meta = _eligible_trials(load_trial_metadata(meta_root, mouse_name))
    video_ids: Dict[int, Optional[str]] = {}
    if "video_ID" in df_meta.columns:
        for tid, vid in zip(df_meta["trial"].astype(int), df_meta["video_ID"]):
            video_ids[int(tid)] = _normalize_video_id_token(vid)

    return pd.DataFrame(
        {
            "mouse": mouse_name,
            "trial_id": [int(t) for t in trial_ids],
            "label": [str(v) for v in labels],
            "video_id": [video_ids.get(int(t)) for t in trial_ids],
            "valid_frames": np.asarray(valid_frames, dtype=np.int64),
            "grid_path": [str(p) for p in paths],
        },
        columns=columns,
    )


def resolve_mice(data_root: Path, mice: Optional[Sequence[str]]) -> List[str]:
    discovered = _discover_mice(Path(data_root))
    selected = list(mice) if mice is not None else discovered
    unknown = [m for m in selected if m not in discovered]
    if unknown:
        raise ValueError(f"Unknown mice (not found under data root): {unknown}")
    return selected


def _subsample_trials(df: pd.DataFrame, max_trials: Optional[int], seed: int) -> pd.DataFrame:
    if max_trials is None or len(df) <= max_trials:
        return df
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(df), size=int(max_trials), replace=False))
    return df.iloc[keep].reset_index(drop=True)


def load_trial_tables(cfg: ProblemConfig) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for mouse in resolve_mice(cfg.data_root, cfg.mice):
        df = load_mouse_trials(cfg.data_root, cfg.meta_root, mouse, cfg.grid_subdir)
        df = _subsample_trials(df, cfg.max_trials_per_mouse, cfg.split_seed)
        if len(df) > 0:
            tables[mouse] = df
    return tables


# ---------------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------------

def build_cv_splits(
    y: np.ndarray,
    *,
    scheme: str,
    n_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Within-mouse CV splits.

    stratified  StratifiedKFold(min(n_splits, smallest class), shuffled with ``seed``)
    logo        Leave-one-group-out (groups = trial IDs: one trial out per fold)
    loso        Leave-one-sample-out
    groupkfold  GroupKFold(min(n_splits, n_groups))
    kfold       KFold(min(n_splits, n_samples), shuffled with ``seed``)
    """
    y = np.asarray(y)
    n_samples = int(len(y))
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for CV, got {n_samples}")
    idx = np.arange(n_samples)

    if scheme == "stratified":
        folds = min(int(n_splits), int(np.bincount(y).min()))
        if folds < 2:
            raise ValueError(f"Not enough samples per class for stratified CV (smallest class={int(np.bincount(y).min())})")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in cv.split(np.zeros(n_samples), y)]
    if scheme == "loso":
        return [(idx[idx != i].astype(np.int64), np.asarray([i], dtype=np.int64)) for i in range(n_samples)]
    if scheme in ("logo", "groupkfold"):
        if groups is None:
            raise ValueError(f"cv scheme '{scheme}' requires groups")
        grp = np.asarray(groups)
        n_groups = int(np.unique(grp).shape[0])
        if n_groups < 2:
            raise ValueError(f"cv scheme '{scheme}' needs >=2 groups, got {n_groups}")
        if scheme == "logo":
            splitter = LeaveOneGroupOut()
        else:
            splitter = GroupKFold(n_splits=int(min(max(2, int(n_splits)), n_groups)))
        return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in splitter.split(idx, groups=grp)]
    if scheme == "kfold":
        kf = KFold(n_splits=int(min(max(2, int(n_splits)), n_samples)), shuffle=True, random_state=int(seed))
        return [(tr.astype(np.int64), te.astype(np.int64)) for tr, te in kf.split(idx)]
    raise ValueError(f"Unsupported cv scheme: {scheme}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(tokens: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes)}
    return np.asarray([lookup[str(t)] for t in tokens], dtype=np.int64)


def _with_window(df: pd.DataFrame, n_frames: int, start: Optional[np.ndarray] = None) -> pd.DataFrame:
    out = df.copy()
    out["start_frame"] = 0 if start is None else start
    out["n_frames"] = int(n_frames)
    return out


def _within_unit(
    cfg: ProblemConfig,
    *,
    unit_id: str,
    label: str,
    mouse: str,
    samples: pd.DataFrame,
    window_frames: int,
    skips: List[Dict[str, Any]],
) -> Optional[EvalUnit]:
    samples = samples.reset_index(drop=True)
    classes = sorted(samples["class_token"].astype(str).unique().tolist())
    if len(classes) < 2:
        skips.append({"unit_id": unit_id, "mouse": mouse, "label": label, "reason": "fewer_than_2_classes", "detail": str(len(classes))})
        return None
    y = _encode(samples["class_token"].astype(str), classes)
    try:
        splits = build_cv_splits(
            y,
            scheme=str(cfg.cv_scheme),
            n_splits=cfg.n_splits,
            seed=cfg.split_seed,
            groups=samples["trial_id"].to_numpy(dtype=np.int64),
        )
    except ValueError as exc:
        skips.append({"unit_id": unit_id, "mouse": mouse, "label": label, "reason": "cv_split_error", "detail": str(exc)})
        return None
    return EvalUnit(
        unit_id=unit_id,
        target=cfg.target,
        scope=cfg.scope,
        label=label,
        mouse=mouse,
        train_mice=[],
        samples=samples,
        classes=classes,
        y=y,
        splits=splits,
        cv_scheme=str(cfg.cv_scheme),
        window_frames=int(window_frames),
    )


def _train_test_unit(
    cfg: ProblemConfig,
    *,
    unit_id: str,
    label: str,
    test_mouse: str,
    train_mice: List[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    classes: List[str],
    window_frames: int,
    skips: List[Dict[str, Any]],
) -> Optional[EvalUnit]:
    samples = pd.concat([train, test], ignore_index=True)
    y = _encode(samples["class_token"].astype(str), classes)
    n_train = len(train)
    tr = np.arange(n_train, dtype=np.int64)
    te = np.arange(n_train, len(samples), dtype=np.int64)
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        skips.append({"unit_id": unit_id, "mouse": test_mouse, "label": label, "reason": "single_class_in_train_or_test", "detail": ""})
        return None
    return EvalUnit(
        unit_id=unit_id,
        target=cfg.target,
        scope=cfg.scope,
        label=label,
        mouse=test_mouse,
        train_mice=list(train_mice),
        samples=samples,
        classes=list(classes),
        y=y,
        splits=[(tr, te)],
        cv_scheme="train_test",
        window_frames=int(window_frames),
    )


# ---------------------------------------------------------------------------
# video_label
# ---------------------------------------------------------------------------

def _build_video_label(cfg: ProblemConfig, tables: Dict[str, pd.DataFrame]) -> ProblemSet:
    skips: List[Dict[str, Any]] = []
    eligible = {m: df for m, df in tables.items() if df["label"].nunique() >= 2}
    for m in sorted(set(tables) - set(eligible)):
        skips.append({"unit_id": m, "mouse": m, "label": "ALL", "reason": "fewer_than_2_labels", "detail": ""})
    if not eligible:
        return ProblemSet([], skips, [], {})

    # Shortest trial over all mice used, so every class has the same window.
    clip = int(cfg.clip_frames) if cfg.clip_frames is not None else int(
        min(int(df["valid_frames"].min()) for df in eligible.values())
    )
    samples_by_mouse = {
        m: _with_window(df.assign(class_token=df["label"].astype(str)), clip) for m, df in eligible.items()
    }

    units: List[EvalUnit] = []
    if cfg.scope == "within_mouse":
        for mouse in sorted(samples_by_mouse):
            unit = _within_unit(
                cfg,
                unit_id=f"mouse={mouse}",
                label="ALL",
                mouse=mouse,
                samples=samples_by_mouse[mouse],
                window_frames=clip,
                skips=skips,
            )
            if unit is not None:
                units.append(unit)
    else:
        mice = sorted(samples_by_mouse)
        if len(mice) < 2:
            raise RuntimeError(f"Need at least 2 eligible mice for cross-mouse classification, got {len(mice)}")
        for test_mouse in mice:
            train_pool = [m for m in mice if m != test_mouse]
            test_labels = set(samples_by_mouse[test_mouse]["label"])
            train_labels = set().union(*[set(samples_by_mouse[m]["label"]) for m in train_pool])
            shared = sorted(test_labels & train_labels)
            unit_id = f"test={test_mouse}"
            if len(shared) < 2:
                skips.append({"unit_id": unit_id, "mouse": test_mouse, "label": "ALL", "reason": "fewer_than_2_shared_labels", "detail": ""})
                continue
            train_parts = []
            train_mice = []
            for m in train_pool:
                part = samples_by_mouse[m][samples_by_mouse[m]["label"].isin(shared)]
                if len(part):
                    train_parts.append(part)
                    train_mice.append(m)
            test = samples_by_mouse[test_mouse][samples_by_mouse[test_mouse]["label"].isin(shared)]
            unit = _train_test_unit(
                cfg,
                unit_id=unit_id,
                label="ALL",
                test_mouse=test_mouse,
                train_mice=train_mice,
                train=pd.concat(train_parts, ignore_index=True),
                test=test,
                classes=shared,
                window_frames=clip,
                skips=skips,
            )
            if unit is not None:
                units.append(unit)

    return ProblemSet(units, skips, sorted(samples_by_mouse), {"clip_frames": clip})


# ---------------------------------------------------------------------------
# video_id
# ---------------------------------------------------------------------------

def _build_video_id(cfg: ProblemConfig, tables: Dict[str, pd.DataFrame]) -> ProblemSet:
    skips: List[Dict[str, Any]] = []
    min_rep = int(cfg.min_repetitions or 1)
    trials = {
        m: df[df["video_id"].notna()].assign(class_token=lambda d: d["video_id"].astype(str)).reset_index(drop=True)
        for m, df in tables.items()
    }

    def clip_for(samples: pd.DataFrame) -> int:
        # Shortest trial among the trials of this classification problem.
        return int(cfg.clip_frames) if cfg.clip_frames is not None else int(samples["valid_frames"].min())

    units: List[EvalUnit] = []
    if cfg.scope == "within_mouse":
        for mouse in sorted(trials):
            df = trials[mouse]
            for label in VIDEO_ID_LABELS:
                unit_id = f"mouse={mouse}|label={label}"
                df_label = df[df["label"] == label]
                if df_label.empty:
                    continue
                counts = Counter(df_label["class_token"])
                keep = sorted(k for k, v in counts.items() if v >= min_rep)
                if len(keep) < 2:
                    skips.append({"unit_id": unit_id, "mouse": mouse, "label": label, "reason": "fewer_than_2_repeated_ids", "detail": str(len(keep))})
                    continue
                samples = df_label[df_label["class_token"].isin(keep)]
                clip = clip_for(samples)
                unit = _within_unit(
                    cfg,
                    unit_id=unit_id,
                    label=label,
                    mouse=mouse,
                    samples=_with_window(samples, clip),
                    window_frames=clip,
                    skips=skips,
                )
                if unit is not None:
                    units.append(unit)
    else:
        mice = sorted(trials)
        for m1, m2 in combinations(mice, 2):
            for source, target in ((m1, m2), (m2, m1)):
                src, tgt = trials[source], trials[target]
                shared_labels = sorted(set(src["label"]) & set(tgt["label"]))
                for label in shared_labels:
                    unit_id = f"train={source}|test={target}|label={label}"
                    src_l = src[src["label"] == label]
                    tgt_l = tgt[tgt["label"] == label]
                    src_counts = Counter(src_l["class_token"])
                    tgt_counts = Counter(tgt_l["class_token"])
                    keep = sorted(
                        k for k in set(src_counts) & set(tgt_counts)
                        if src_counts[k] >= min_rep and tgt_counts[k] >= min_rep
                    )
                    if len(keep) < 2:
                        skips.append({"unit_id": unit_id, "mouse": target, "label": label, "reason": "fewer_than_2_shared_repeated_ids", "detail": str(len(keep))})
                        continue
                    train = src_l[src_l["class_token"].isin(keep)]
                    test = tgt_l[tgt_l["class_token"].isin(keep)]
                    clip = clip_for(pd.concat([train, test]))
                    unit = _train_test_unit(
                        cfg,
                        unit_id=unit_id,
                        label=label,
                        test_mouse=target,
                        train_mice=[source],
                        train=_with_window(train, clip),
                        test=_with_window(test, clip),
                        classes=keep,
                        window_frames=clip,
                        skips=skips,
                    )
                    if unit is not None:
                        units.append(unit)

    used = sorted({m for u in units for m in [u.mouse, *u.train_mice]})
    return ProblemSet(units, skips, used, {"clip_rule": "shortest trial in each classification problem"})


# ---------------------------------------------------------------------------
# segment_id
# ---------------------------------------------------------------------------

def load_segment_samples(cfg: ProblemConfig, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Segment samples (one row per segment occurrence) per mouse."""
    out: Dict[str, pd.DataFrame] = {}
    for mouse, trials in tables.items():
        seg_df, _counters = build_segment_sample_records(
            meta_root=cfg.meta_root,
            mouse_name=mouse,
            target_labels=SEGMENT_ID_LABELS,
            segment_length_by_label=cfg.segment_lengths,
            videos_subdir="global_meta/videos",
        )
        if seg_df.empty:
            continue
        trial_cols = trials[["trial_id", "label", "valid_frames", "grid_path"]]
        merged = seg_df.drop(columns=["valid_frames"]).merge(
            trial_cols, on=["trial_id", "label"], how="inner", validate="many_to_one"
        )
        if merged.empty:
            continue
        merged = merged.assign(
            class_token=merged["segment_id"].astype(str),
            n_frames=merged["seg_length"].astype(np.int64),
        )
        out[mouse] = merged.reset_index(drop=True)
    return out


def _build_segment_id(cfg: ProblemConfig, tables: Dict[str, pd.DataFrame]) -> ProblemSet:
    skips: List[Dict[str, Any]] = []
    min_rep = int(cfg.min_repetitions or 1)
    segs = load_segment_samples(cfg, tables)

    units: List[EvalUnit] = []
    if cfg.scope == "within_mouse":
        for mouse in sorted(segs):
            df = segs[mouse]
            for label in SEGMENT_ID_LABELS:
                unit_id = f"mouse={mouse}|label={label}"
                df_label = df[df["label"] == label]
                if df_label.empty:
                    continue
                counts = Counter(df_label["class_token"])
                keep = sorted(k for k, v in counts.items() if v >= min_rep)
                samples = df_label[df_label["class_token"].isin(keep)]
                unit = _within_unit(
                    cfg,
                    unit_id=unit_id,
                    label=label,
                    mouse=mouse,
                    samples=samples,
                    window_frames=int(cfg.segment_lengths[label]),
                    skips=skips,
                )
                if unit is not None:
                    units.append(unit)
    else:
        mice = sorted(segs)
        if len(mice) < 2:
            raise RuntimeError(f"Need at least 2 mice with segment samples, got {len(mice)}")
        for test_mouse in mice:
            train_pool = [m for m in mice if m != test_mouse]
            for label in SEGMENT_ID_LABELS:
                unit_id = f"test={test_mouse}|label={label}"
                test_l = segs[test_mouse][segs[test_mouse]["label"] == label]
                train_l = pd.concat([segs[m][segs[m]["label"] == label] for m in train_pool], ignore_index=True)
                if test_l.empty and train_l.empty:
                    continue
                if test_l.empty or train_l.empty:
                    skips.append({"unit_id": unit_id, "mouse": test_mouse, "label": label, "reason": "label_missing_in_train_or_test", "detail": ""})
                    continue
                c_test = Counter(test_l["class_token"])
                c_train = Counter(train_l["class_token"])
                keep = sorted(
                    k for k in set(c_test) & set(c_train) if c_test[k] >= min_rep and c_train[k] >= min_rep
                )
                if len(keep) < 2:
                    skips.append({"unit_id": unit_id, "mouse": test_mouse, "label": label, "reason": "fewer_than_2_shared_segment_ids", "detail": str(len(keep))})
                    continue
                train = train_l[train_l["class_token"].isin(keep)]
                test = test_l[test_l["class_token"].isin(keep)]
                unit = _train_test_unit(
                    cfg,
                    unit_id=unit_id,
                    label=label,
                    test_mouse=test_mouse,
                    train_mice=sorted(train["mouse"].unique().tolist()),
                    train=train,
                    test=test,
                    classes=keep,
                    window_frames=int(cfg.segment_lengths[label]),
                    skips=skips,
                )
                if unit is not None:
                    units.append(unit)

    used = sorted({m for u in units for m in [u.mouse, *u.train_mice]})
    return ProblemSet(units, skips, used, {"segment_lengths": dict(cfg.segment_lengths)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_problems(cfg: ProblemConfig) -> ProblemSet:
    tables = load_trial_tables(cfg)
    if not tables:
        raise RuntimeError("No mouse has eligible trials with grid files")
    if cfg.target == "video_label":
        return _build_video_label(cfg, tables)
    if cfg.target == "video_id":
        return _build_video_id(cfg, tables)
    return _build_segment_id(cfg, tables)
