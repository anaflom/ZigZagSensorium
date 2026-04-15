#!/usr/bin/env python3
"""Shared utilities for loading zigzag data and computing vectorizations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def load_labelled_barcodes(
    data_root: Path,
    meta_root: Path,
    mouse_name: str,
    zz_folder: str,
    max_trials: Optional[int] = None,
) -> Tuple[List[List[Tuple[float, float, float]]], np.ndarray, List[int], np.ndarray]:
    """Load barcodes and attach labels/valid frame counts from metadata."""
    df = load_trial_metadata(meta_root, mouse_name)
    trial_to_label = dict(zip(df["trial"].astype(int), df["label"]))
    trial_to_frames = dict(zip(df["trial"].astype(int), df["valid_frames"].astype(int)))

    zz_dir = data_root / mouse_name / zz_folder
    files = sorted(zz_dir.glob("zz-thresh-*.npy"))
    files = [f for f in files if "info" not in f.name]

    if max_trials is not None and files:
        indices = np.linspace(0, len(files) - 1, min(max_trials, len(files)), dtype=int)
        files = [files[i] for i in indices]

    barcodes: List[List[Tuple[float, float, float]]] = []
    labels_list: List[str] = []
    trial_ids: List[int] = []
    frames_list: List[int] = []

    for fpath in files:
        # Anchor to end of filename so we do not capture unrelated numbers.
        match = re.search(r"trial-(\d+)$", fpath.stem)
        if match is None:
            continue

        trial_num = int(match.group(1))
        if trial_num not in trial_to_label:
            continue

        raw = np.load(fpath, allow_pickle=True)
        bars = [tuple(row) for row in raw]
        barcodes.append(bars)
        labels_list.append(trial_to_label[trial_num])
        trial_ids.append(trial_num)
        frames_list.append(trial_to_frames[trial_num])

    return barcodes, np.array(labels_list), trial_ids, np.array(frames_list)


def clip_barcodes(
    barcodes: Iterable[Iterable[Tuple[float, float, float]]], n_frames: int
) -> List[List[Tuple[float, float, float]]]:
    """Clip bars to a fixed frame budget by truncating deaths and dropping late births."""
    clipped: List[List[Tuple[float, float, float]]] = []
    for bars in barcodes:
        new_bars: List[Tuple[float, float, float]] = []
        for dim, b, d in bars:
            if b >= n_frames:
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
