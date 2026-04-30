#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for loading zigzag data and computing vectorizations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from zztop.vectorizations import (
    BettiCurve,
    BettiProfile,
    BirthFrequency,
    CumulativePersistence,
    EffectivePersistenceImage,
    PersistenceEntropy,
    PersistenceImage,
    PersistenceLandscape,
    PersistenceProfile,
    PersistenceStatistics,
    Silhouette,
    TurnoverRate,
)
from zztop.vectorizations._diagram import normalize_diagram


def _str2bool(value: str) -> bool:
    """Parse common boolean-like CLI strings into booleans."""
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _opt_int(value: str) -> Optional[int]:
    """Parse an optional integer from CLI string values."""
    if value is None:
        return None
    if value.strip().lower() in {"none", "null", ""}:
        return None
    return int(value)


def _opt_csv_list(value: str) -> Optional[List[str]]:
    """Parse comma-separated strings into a list (or None)."""
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items if items else None


def _build_zz_folder(p_active: int, per_trial_thresh: bool) -> str:
    """Build zigzag folder name from threshold options."""
    if per_trial_thresh:
        return f"trials_zz-thresh-{p_active}-per-trial"
    return f"trials_zz-thresh-{p_active}"


def _resolve_mouse_cache_dir(state: Any, mouse_name: str) -> Path:
    """Resolve cache path using state.cache_dir fallback to <data_root>/<mouse>/cache."""
    cache_dir = getattr(state, "cache_dir", None)
    if cache_dir is not None:
        return Path(cache_dir)
    data_root = getattr(state, "data_root", None)
    if data_root is None:
        raise AttributeError("state must define data_root when cache_dir is not set")
    return Path(data_root) / mouse_name / "cache"


def load_zigzag_barcodes(
    data_root: Path,
    mouse_name: str,
    zz_folder: str,
    max_trials: Optional[int] = None,
) -> Tuple[List[List[Tuple[float, float, float]]], List[str]]:
    """Load zigzag barcode files for one mouse."""
    zz_dir = data_root / mouse_name / zz_folder
    files = sorted(zz_dir.glob("zz-thresh-*.npy"))
    files = [f for f in files if "info" not in f.name]

    if max_trials is not None and files:
        indices = np.linspace(0, len(files) - 1, min(max_trials, len(files)), dtype=int)
        files = [files[i] for i in indices]

    barcodes: List[List[Tuple[float, float, float]]] = []
    trial_names: List[str] = []
    for fpath in files:
        raw = np.load(fpath, allow_pickle=True)
        bars = [tuple(row) for row in raw]
        barcodes.append(bars)
        trial_names.append(fpath.stem)

    return barcodes, trial_names


def load_trial_metadata(meta_root: Path, mouse_name: str) -> pd.DataFrame:
    """Load per-trial metadata for one mouse."""
    csv_path = meta_root / mouse_name / "trials" / f"meta-trials_{mouse_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")
    return pd.read_csv(csv_path)


def _extract_trial_id_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    m = re.search(r"trial-(\d+)$", stem)
    if m is not None:
        return int(m.group(1))
    if stem.isdigit():
        return int(stem)
    m2 = re.search(r"_(\d+)$", stem)
    if m2 is not None:
        return int(m2.group(1))
    return None


def _resolve_trial_response_file(responses_dir: Path, trial_id: int) -> Optional[Path]:
    matches: List[Path] = []
    for fpath in sorted(responses_dir.glob("*.npy")):
        tid = _extract_trial_id_from_name(fpath)
        if tid == int(trial_id):
            matches.append(fpath)
            continue
        # # Explicit fallback to satisfy requested rule where trial-<id> appears in names.
        # if f"trial-{trial_id}" in fpath.stem:
        #     matches.append(fpath)

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return None
    raise RuntimeError(
        f"Multiple response files matched trial {trial_id} in {responses_dir}: "
        f"{[str(m.name) for m in matches]}"
    )


def load_labelled_barcodes(
    data_root: Path,
    meta_root: Path,
    mouse_name: str,
    zz_folder: str,
    max_trials: Optional[int] = None,
) -> Tuple[List[List[Tuple[float, float, float]]], np.ndarray, List[int], np.ndarray]:
    """Load eligible barcodes and attach labels/valid frame counts from metadata.

    When ``max_trials`` is provided, the cap is applied *after* metadata filtering
    (valid_response & valid_trial). Candidate files are shuffled before loading.
    """
    df = load_trial_metadata(meta_root, mouse_name)
    df_eligible = _eligible_trials(df)
    trial_to_label = dict(zip(df_eligible["trial"].astype(int), df_eligible["label"]))
    trial_to_frames = dict(zip(df_eligible["trial"].astype(int), df_eligible["valid_frames"].astype(int)))

    zz_dir = data_root / mouse_name / zz_folder
    files = sorted(zz_dir.glob("zz-thresh-*.npy"))
    files = [f for f in files if "info" not in f.name]

    if max_trials is not None and files:
        # Apply max_trials after eligibility filtering by randomizing candidates
        # and early-stopping once enough valid trials are collected.
        rng = np.random.default_rng()
        order = rng.permutation(len(files))
        files = [files[i] for i in order]

    barcodes: List[List[Tuple[float, float, float]]] = []
    labels_list: List[str] = []
    trial_ids: List[int] = []
    frames_list: List[int] = []

    for fpath in files:
        trial_num = _extract_trial_id_from_name(fpath)
        if trial_num is None:
            continue
        if trial_num not in trial_to_label:
            continue

        raw = np.load(fpath, allow_pickle=True)
        bars = [tuple(row) for row in raw]
        barcodes.append(bars)
        labels_list.append(trial_to_label[trial_num])
        trial_ids.append(trial_num)
        frames_list.append(trial_to_frames[trial_num])

        if max_trials is not None and len(barcodes) >= max_trials:
            break

    return barcodes, np.array(labels_list), trial_ids, np.array(frames_list)


def load_labelled_grid_paths(
    data_root: Path,
    meta_root: Path,
    mouse_name: str,
    grid_subdir: str = "trials_grid",
) -> Tuple[List[Path], np.ndarray, List[int], np.ndarray]:
    """Load eligible grid file paths and attach labels/valid frame counts.

    Expects grid files under ``<data_root>/<mouse_name>/<grid_subdir>`` and uses
    the trial id extracted from filenames to align with eligible metadata rows.
    """
    df = load_trial_metadata(meta_root, mouse_name)
    df_eligible = _eligible_trials(df)
    trial_to_label = dict(zip(df_eligible["trial"].astype(int), df_eligible["label"]))
    trial_to_frames = dict(zip(df_eligible["trial"].astype(int), df_eligible["valid_frames"].astype(int)))

    grid_dir = data_root / mouse_name / grid_subdir
    if not grid_dir.exists():
        return [], np.array([]), [], np.array([])

    paths = sorted(grid_dir.glob("*.npy"))

    selected_paths: List[Path] = []
    labels_list: List[str] = []
    trial_ids: List[int] = []
    frames_list: List[int] = []

    for fpath in paths:
        trial_num = _extract_trial_id_from_name(fpath)
        if trial_num is None:
            continue
        if trial_num not in trial_to_label:
            continue

        selected_paths.append(fpath)
        labels_list.append(trial_to_label[trial_num])
        trial_ids.append(trial_num)
        frames_list.append(trial_to_frames[trial_num])

    return selected_paths, np.array(labels_list), trial_ids, np.array(frames_list)


def clip_barcodes(
    barcodes: Iterable[Iterable[Tuple[float, float, float]]], n_frames: int
) -> List[List[Tuple[float, float, float]]]:
    """Clip bars to a fixed frame budget using zigzag endpoint times.

    Barcode birth/death values are not zero-based frame indices. They are
    1-based endpoint times for half-open intervals, so a birth exactly equal
    to ``n_frames`` is still valid for a trial with ``n_frames`` samples.
    """
    clipped: List[List[Tuple[float, float, float]]] = []
    for bars in barcodes:
        new_bars: List[Tuple[float, float, float]] = []
        for dim, b, d in bars:
            if b > n_frames:
                continue
            new_bars.append((dim, b, min(d, n_frames)))
        clipped.append(new_bars)
    return clipped


def build_vectorizer(method: str, clip_frames: Optional[int] = None) -> object:
    """Build a vectorizer by method name."""
    base: Dict[str, object] = {
        "PersImage": PersistenceImage(resolution=(20, 20), sigma=1.0),
        "PI_20x20_s1": PersistenceImage(resolution=(20, 20), sigma=1.0),
        "PI_20x20_s2": PersistenceImage(resolution=(20, 20), sigma=2.0),
        "PI_30x30_s1": PersistenceImage(resolution=(30, 30), sigma=1.0),
        "Landscape": PersistenceLandscape(n_landscapes=5, resolution=100),
        "Landscape_5": PersistenceLandscape(n_landscapes=5, resolution=100),
        "Landscape_10": PersistenceLandscape(n_landscapes=10, resolution=100),
        "Silhouette_p1": Silhouette(resolution=100, power=1.0),
        "Silhouette_p2": Silhouette(resolution=100, power=2.0),
        "Silhouette": Silhouette(resolution=100),
        "BettiCurve_100": BettiCurve(resolution=100),
        "BettiCurve_200": BettiCurve(resolution=200),
        "BettiCurve": BettiCurve(resolution=100),
        "Entropy": PersistenceEntropy(normalize=True),
        "PersEntropy": PersistenceEntropy(),
        "Statistics": PersistenceStatistics(),
        "PersStats": PersistenceStatistics(),
    }
    if method in base:
        return base[method]

    n_frames_kw: Dict[str, int] = {}
    if clip_frames is not None:
        n_frames_kw = {"n_frames": int(clip_frames)}

    zigzag_specific: Dict[str, object] = {
        "BettiProfile": BettiProfile(**n_frames_kw),
        "BirthFreq": BirthFrequency(**n_frames_kw),
        "PersProfile": PersistenceProfile(**n_frames_kw),
        "Turnover": TurnoverRate(**n_frames_kw),
        "EffPI": EffectivePersistenceImage(**n_frames_kw),
        "CumPers": CumulativePersistence(**n_frames_kw),
    }
    if method in zigzag_specific:
        return zigzag_specific[method]

    raise ValueError(
        f"Unknown vectorization method '{method}'. Supported: {sorted(list(base.keys()) + list(zigzag_specific.keys()))}"
    )


def build_vectorization_cache_stem(
    mouse_name: str,
    method: str,
    p_active: int,
    per_trial_thresh: bool,
    clip_frames: Optional[int],
) -> str:
    """Build a deterministic cache stem for vectorized features."""
    method_safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", method).strip("_")
    clip_part = "none" if clip_frames is None else str(int(clip_frames))
    per_trial_part = "pertrial1" if per_trial_thresh else "pertrial0"
    return f"{mouse_name}_{method_safe}_p{int(p_active)}_{per_trial_part}_clip{clip_part}"


def load_vectorization_cache(cache_path: Path) -> Dict[str, Any]:
    """Load vectorization outputs from a .npz cache file."""
    data = np.load(cache_path, allow_pickle=True)
    out: Dict[str, Any] = {k: data[k] for k in data.files}
    out["cache_path"] = str(cache_path)
    return out


def build_shuffled_grid_cache_dir(cache_root: Path, cache_stem: str) -> Path:
    """Return the directory that stores shuffled grid files for one cache stem."""
    return Path(cache_root) / f"{cache_stem}_grids"


def build_shuffled_grid_cache_path(cache_root: Path, cache_stem: str, trial_id: int) -> Path:
    """Return the per-trial shuffled grid cache path for a shuffle cache stem."""
    grid_dir = build_shuffled_grid_cache_dir(cache_root, cache_stem)
    return grid_dir / f"trial_{int(trial_id):06d}.npy"


def load_shuffled_grid_cache_paths(cache_path: Path, trial_ids: Sequence[int]) -> List[Path]:
    """Resolve shuffled grid cache files for the provided trial IDs.

    Raises
    ------
    FileNotFoundError
        If the shuffled grid cache directory or any trial file is missing.
    """
    cache_path = Path(cache_path)
    grid_dir = build_shuffled_grid_cache_dir(cache_path.parent, cache_path.stem)
    if not grid_dir.exists():
        raise FileNotFoundError(
            f"Shuffled grid cache directory not found for {cache_path.name}: {grid_dir}"
        )

    grid_paths: List[Path] = []
    missing: List[Path] = []
    for trial_id in trial_ids:
        grid_path = build_shuffled_grid_cache_path(cache_path.parent, cache_path.stem, int(trial_id))
        if not grid_path.exists():
            missing.append(grid_path)
        grid_paths.append(grid_path)

    if missing:
        raise FileNotFoundError(
            "Missing shuffled grid cache files: "
            + ", ".join(str(path.name) for path in missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )

    return grid_paths


def create_vectorization(
    barcodes: List[List[Tuple[float, float, float]]],
    method: str,
    *,
    clip_frames: Optional[int] = None,
    output_folder: Optional[Path] = None,
    cache_stem: Optional[str] = None,
    mouse_name: Optional[str] = None,
    labels: Optional[np.ndarray] = None,
    trial_ids: Optional[Iterable[int]] = None,
    valid_frames: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Create vectorized features from barcodes and optionally save a .npz cache.

    The vectorization method is passed by name through ``method``.
    """
    clip_used = int(clip_frames) if clip_frames is not None else None
    barcodes_used = clip_barcodes(barcodes, clip_used) if clip_used is not None else barcodes

    diagrams = [normalize_diagram(b, drop_inf=True) for b in barcodes_used]
    vectorizer = build_vectorizer(method, clip_frames=clip_used)

    xmat = vectorizer.fit_transform(diagrams)
    if xmat.ndim == 1:
        xmat = xmat.reshape(-1, 1)
    xmat = np.nan_to_num(xmat)

    result: Dict[str, Any] = {
        "features": xmat,
        "method": method,
        "clip_frames": clip_used,
        "n_samples": int(xmat.shape[0]),
        "n_features": int(xmat.shape[1]),
        "n_finite_rows": int(np.isfinite(xmat).all(axis=1).sum()),
        "feature_min": float(np.nanmin(xmat)),
        "feature_max": float(np.nanmax(xmat)),
        "cache_path": None,
    }

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        if cache_stem is None:
            mouse_part = mouse_name if mouse_name is not None else "unknown_mouse"
            cache_stem = build_vectorization_cache_stem(
                mouse_part,
                method,
                p_active=0,
                per_trial_thresh=False,
                clip_frames=clip_used,
            )

        cache_path = output_folder / f"{cache_stem}.npz"
        payload: Dict[str, Any] = {
            "X": xmat,
            "features": xmat,
            "method": np.array(method),
            "clip_frames": np.array(-1 if clip_used is None else clip_used),
        }
        if mouse_name is not None:
            payload["mouse_name"] = np.array(mouse_name)
        if labels is not None:
            payload["labels"] = np.asarray(labels)
        if trial_ids is not None:
            payload["trial_ids"] = np.asarray(list(trial_ids))
        if valid_frames is not None:
            payload["valid_frames"] = np.asarray(valid_frames)

        np.savez_compressed(cache_path, **payload)
        result["cache_path"] = str(cache_path)

    return result


def compute_and_save_vectorization_cache(
    *,
    data_root: Path,
    mouse_name: str,
    method: str,
    p_active: int,
    per_trial_thresh: bool,
    clip_frames: Optional[int],
    barcodes: Sequence[Sequence[Tuple[float, float, float]]],
    labels: Optional[Sequence[str]],
    trial_ids: Optional[Sequence[int]],
    valid_frames: Optional[Sequence[int]],
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Path]:
    """Compute vectorization features and persist a cache file."""
    clip_used = int(clip_frames) if clip_frames is not None else None
    cache_stem = build_vectorization_cache_stem(
        mouse_name=mouse_name,
        method=method,
        p_active=p_active,
        per_trial_thresh=per_trial_thresh,
        clip_frames=clip_used,
    )
    mouse_cache_dir = cache_dir if cache_dir is not None else data_root / mouse_name / "cache"
    mouse_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = mouse_cache_dir / f"{cache_stem}.npz"

    vec_out = create_vectorization(
        list(barcodes),
        method,
        clip_frames=clip_used,
        output_folder=mouse_cache_dir,
        cache_stem=cache_stem,
        mouse_name=mouse_name,
        labels=None if labels is None else np.asarray(labels),
        trial_ids=trial_ids,
        valid_frames=None if valid_frames is None else np.asarray(valid_frames),
    )
    xmat = np.asarray(vec_out["features"])
    return xmat, cache_path


def _build_vectorization_precompute_hint(
    *,
    mouse_name: str,
    method: str,
    p_active: int,
    per_trial_thresh: bool,
    clip_frames: Optional[int],
) -> str:
    clip_token = "None" if clip_frames is None else str(int(clip_frames))
    return (
        "Compute it first with: "
        "python3 scripts/generate_vectorization_cache.py "
        f"--mice {mouse_name} "
        f"--vectorization-method {method} "
        f"--p-active {int(p_active)} "
        f"--per-trial-thresh {'true' if per_trial_thresh else 'false'} "
        f"--clip-frames {clip_token}"
    )


def load_or_compute_vectorization_features(
    *,
    data_root: Path,
    mouse_name: str,
    method: str,
    p_active: int,
    per_trial_thresh: bool,
    clip_frames: Optional[int],
    barcodes: Sequence[Sequence[Tuple[float, float, float]]],
    labels: Optional[Sequence[str]],
    trial_ids: Optional[Sequence[int]],
    valid_frames: Optional[Sequence[int]],
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    expected_trial_ids: Optional[Sequence[int]] = None,
    message_prefix: str = "",
) -> Tuple[np.ndarray, str, Path]:
    """Strictly load vectorization features from cache.

    This helper no longer computes vectorizations on cache miss. Use
    scripts/generate_vectorization_cache.py to precompute missing caches.
    """
    clip_used = int(clip_frames) if clip_frames is not None else None
    cache_stem = build_vectorization_cache_stem(
        mouse_name=mouse_name,
        method=method,
        p_active=p_active,
        per_trial_thresh=per_trial_thresh,
        clip_frames=clip_used,
    )
    mouse_cache_dir = cache_dir if cache_dir is not None else data_root / mouse_name / "cache"
    cache_path = mouse_cache_dir / f"{cache_stem}.npz"

    if force_recompute:
        raise RuntimeError(
            f"{message_prefix}force_recompute=True is not supported in strict cache mode. "
            + _build_vectorization_precompute_hint(
                mouse_name=mouse_name,
                method=method,
                p_active=p_active,
                per_trial_thresh=per_trial_thresh,
                clip_frames=clip_used,
            )
            + " and pass --force-recompute true there."
        )

    if not cache_path.exists():
        raise RuntimeError(
            f"{message_prefix}Missing vectorization cache: {cache_path}. "
            + _build_vectorization_precompute_hint(
                mouse_name=mouse_name,
                method=method,
                p_active=p_active,
                per_trial_thresh=per_trial_thresh,
                clip_frames=clip_used,
            )
        )

    cache = load_vectorization_cache(cache_path)
    if "features" in cache:
        xmat = np.asarray(cache["features"])
    elif "X" in cache:
        xmat = np.asarray(cache["X"])
    else:
        raise RuntimeError(f"{message_prefix}Cache missing feature matrix: {cache_path}")
    xmat = np.nan_to_num(xmat)

    if expected_trial_ids is not None:
        expected = [int(t) for t in expected_trial_ids]
        cache_trial_ids: Optional[List[int]] = None
        if "trial_ids" in cache:
            cache_trial_ids = [int(t) for t in np.asarray(cache["trial_ids"]).tolist()]

        if cache_trial_ids is not None:
            if len(cache_trial_ids) != int(xmat.shape[0]):
                raise RuntimeError(
                    f"{message_prefix}Cache mismatch for {mouse_name}: trial_ids length "
                    f"({len(cache_trial_ids)}) differs from feature rows ({int(xmat.shape[0])}) in {cache_path}. "
                    + _build_vectorization_precompute_hint(
                        mouse_name=mouse_name,
                        method=method,
                        p_active=p_active,
                        per_trial_thresh=per_trial_thresh,
                        clip_frames=clip_used,
                    )
                )
            if cache_trial_ids != expected:
                raise RuntimeError(
                    f"{message_prefix}Cache mismatch for {mouse_name}: cached trial_ids differ from current trials in {cache_path}. "
                    + _build_vectorization_precompute_hint(
                        mouse_name=mouse_name,
                        method=method,
                        p_active=p_active,
                        per_trial_thresh=per_trial_thresh,
                        clip_frames=clip_used,
                    )
                )
        elif int(xmat.shape[0]) != len(expected):
            raise RuntimeError(
                f"{message_prefix}Cache mismatch for {mouse_name}: feature rows ({int(xmat.shape[0])}) differ "
                f"from expected trial count ({len(expected)}) and cache has no trial_ids in {cache_path}. "
                + _build_vectorization_precompute_hint(
                    mouse_name=mouse_name,
                    method=method,
                    p_active=p_active,
                    per_trial_thresh=per_trial_thresh,
                    clip_frames=clip_used,
                )
            )

    return xmat, "cache", cache_path


def _discover_mice(data_root: Path) -> List[str]:
    return sorted(
        [d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith("dynamic")]
    )


def _short_mouse_name(name: str) -> str:
    """Return a compact mouse label from a full directory name.

    Example: ``dynamic29156-11-10-Video-...`` → ``rec-29156-11-10``.
    Falls back to the original name if the pattern is not matched.
    """
    m = re.match(r"dynamic(\d+)-(\d+)-(\d+)", name)
    if m:
        return f"rec-{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return name


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "t", "yes", "y"})


def _eligible_trials(df_trials: pd.DataFrame) -> pd.DataFrame:
    if "valid_response" not in df_trials.columns or "valid_trial" not in df_trials.columns:
        raise ValueError("Metadata CSV must contain valid_response and valid_trial columns")
    vr = _to_bool_series(df_trials["valid_response"])
    vt = _to_bool_series(df_trials["valid_trial"])
    return df_trials.loc[vr & vt].copy()


def _normalize_video_id_token(video_id: Any) -> Optional[str]:
    """Normalize metadata video ID to a filesystem token.

    Returns None for missing/invalid IDs.
    """
    if video_id is None:
        return None
    if isinstance(video_id, float) and np.isnan(video_id):
        return None
    if pd.isna(video_id):
        return None

    text = str(video_id).strip()
    if text == "":
        return None

    # Common CSV case: integer IDs loaded as float strings (e.g. "123.0").
    m = re.fullmatch(r"([0-9]+)\.0+", text)
    if m is not None:
        return m.group(1)

    return text


def load_video_metadata_json(
    meta_root: Path,
    video_id: Any,
    videos_subdir: str = "global_meta/videos",
    label: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[str]]:
    """Load one video metadata JSON by video ID.

    Files are named ``<label>-<ID>.json`` under *videos_subdir*.  When *label*
    is supplied the labelled form is tried first; plain ``<ID>.json`` is kept
    as a fallback for backwards compatibility.

    Returns (payload, path, token). If not found, payload/path are None.
    """
    token = _normalize_video_id_token(video_id)
    if token is None:
        return None, None, None

    videos_dir = Path(meta_root) / videos_subdir
    if not videos_dir.exists():
        return None, None, token

    candidates: List[Path] = []
    if label is not None:
        label_clean = str(label).strip()
        candidates.append(videos_dir / f"{label_clean}-{token}.json")
        raw = str(video_id).strip()
        if raw != token:
            candidates.append(videos_dir / f"{label_clean}-{raw}.json")
    candidates += [
        videos_dir / f"{token}.json",
        videos_dir / f"{str(video_id).strip()}.json",
    ]
    seen: set = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for path in unique_candidates:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload, path, token

    return None, None, token


def _coerce_segment_records(segments_payload: Any) -> List[Dict[str, Any]]:
    """Convert segment payload into a list of records with start/end/segment_id fields."""
    if segments_payload is None:
        return []

    out: List[Dict[str, Any]] = []

    # Format A: list of dicts
    if isinstance(segments_payload, list):
        for i, item in enumerate(segments_payload):
            if not isinstance(item, dict):
                continue
            seg_id = (
                item.get("segment_id")
                or item.get("segment_ID")
                or item.get("id")
                or item.get("ID")
                or item.get("name")
            )
            out.append(
                {
                    "segment_id": seg_id,
                    "frame_start": item.get("frame_start"),
                    "frame_end": item.get("frame_end"),
                    "segment_index": i,
                }
            )
        return out

    # Format B: dict of arrays
    if isinstance(segments_payload, dict):
        starts = segments_payload.get("frame_start")
        ends = segments_payload.get("frame_end")
        seg_ids = (
            segments_payload.get("segment_id")
            or segments_payload.get("segment_ID")
            or segments_payload.get("id")
            or segments_payload.get("ID")
            or segments_payload.get("segment_ids")
        )

        if not isinstance(starts, (list, tuple, np.ndarray)):
            starts = [starts]
        if not isinstance(ends, (list, tuple, np.ndarray)):
            ends = [ends]
        if not isinstance(seg_ids, (list, tuple, np.ndarray)):
            seg_ids = [seg_ids]

        n = max(len(starts), len(ends), len(seg_ids))
        for i in range(n):
            out.append(
                {
                    "segment_id": seg_ids[i] if i < len(seg_ids) else None,
                    "frame_start": starts[i] if i < len(starts) else None,
                    "frame_end": ends[i] if i < len(ends) else None,
                    "segment_index": i,
                }
            )
        return out

    return []


def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(str(value).strip()))
        except Exception:
            return None


def build_segment_sample_records(
    meta_root: Path,
    mouse_name: str,
    target_labels: Sequence[str],
    segment_length_by_label: Dict[str, int],
    videos_subdir: str = "global_meta/videos",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build per-segment sample records from trial metadata + video JSON metadata.

    Returned DataFrame columns:
      mouse, label, trial_id, video_id, video_token, video_json_path,
      segment_id, segment_index, start_frame, end_frame_exclusive,
      seg_length, valid_frames
    """
    counters: Dict[str, int] = {
        "rows_total": 0,
        "rows_eligible": 0,
        "rows_target_label": 0,
        "missing_video_id": 0,
        "missing_video_json": 0,
        "missing_segments": 0,
        "missing_segment_id": 0,
        "invalid_segment_start": 0,
        "start_out_of_bounds": 0,
        "insufficient_frames": 0,
        "records_built": 0,
    }

    df_trials = load_trial_metadata(meta_root, mouse_name)
    counters["rows_total"] = int(len(df_trials))

    required = {"trial", "label", "valid_frames", "video_ID", "valid_response", "valid_trial"}
    missing = sorted([c for c in required if c not in df_trials.columns])
    if missing:
        raise ValueError(f"Metadata for {mouse_name} missing required columns: {missing}")

    df_eligible = _eligible_trials(df_trials)
    counters["rows_eligible"] = int(len(df_eligible))

    label_set = set(target_labels)
    df_target = df_eligible[df_eligible["label"].isin(label_set)].copy()
    counters["rows_target_label"] = int(len(df_target))

    records: List[Dict[str, Any]] = []
    video_cache: Dict[Tuple[str, str], Tuple[Optional[Dict[str, Any]], Optional[Path], Optional[str]]] = {}

    for row in df_target.itertuples(index=False):
        label = str(getattr(row, "label"))
        trial_id = _to_int_or_none(getattr(row, "trial"))
        valid_frames = _to_int_or_none(getattr(row, "valid_frames"))
        video_id = getattr(row, "video_ID")

        if trial_id is None or valid_frames is None:
            counters["invalid_segment_start"] += 1
            continue
        if label not in segment_length_by_label:
            continue

        video_key = _normalize_video_id_token(video_id)
        if video_key is None:
            counters["missing_video_id"] += 1
            continue

        cache_key = (label, video_key)
        if cache_key not in video_cache:
            video_cache[cache_key] = load_video_metadata_json(
                meta_root=meta_root,
                video_id=video_id,
                label=label,
                videos_subdir=videos_subdir,
            )
        payload, json_path, video_token = video_cache[cache_key]

        if payload is None:
            counters["missing_video_json"] += 1
            continue

        seg_payload = payload.get("segments")
        seg_records = _coerce_segment_records(seg_payload)
        if len(seg_records) == 0:
            counters["missing_segments"] += 1
            continue

        seg_length = int(segment_length_by_label[label])

        for seg in seg_records:
            seg_id = seg.get("segment_id")
            start = _to_int_or_none(seg.get("frame_start"))

            if seg_id is None or str(seg_id).strip() == "":
                counters["missing_segment_id"] += 1
                continue
            if start is None:
                counters["invalid_segment_start"] += 1
                continue
            if start < 0 or start >= valid_frames:
                counters["start_out_of_bounds"] += 1
                continue

            end_exclusive = int(start + seg_length)
            if end_exclusive > valid_frames:
                counters["insufficient_frames"] += 1
                continue

            records.append(
                {
                    "mouse": mouse_name,
                    "label": label,
                    "trial_id": int(trial_id),
                    "video_id": str(video_id),
                    "video_token": str(video_token) if video_token is not None else str(video_key),
                    "video_json_path": str(json_path) if json_path is not None else "",
                    "segment_id": str(seg_id),
                    "segment_index": int(seg.get("segment_index", 0)),
                    "start_frame": int(start),
                    "end_frame_exclusive": int(end_exclusive),
                    "seg_length": int(seg_length),
                    "valid_frames": int(valid_frames),
                }
            )

    counters["records_built"] = int(len(records))
    return pd.DataFrame.from_records(records), counters
