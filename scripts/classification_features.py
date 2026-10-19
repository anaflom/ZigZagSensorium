#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inputs for the classifiers: zigzag vectorization features and grid datasets.

Both take the ``samples`` table of a :class:`classification_data.EvalUnit`
(one row per sample with ``mouse, trial_id, start_frame, n_frames, ...``) and
return inputs aligned with its rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import (
    build_shuffled_grid_cache_path,
    build_vectorization_cache_stem,
    is_frame_aligned_method,
    load_feature_cache,
    slice_frame_aligned_features,
)


SHUFFLE_TYPES: Tuple[str, ...] = ("time", "spatial", "phase")


def shuffle_mode_token(different_shuffle_per_trial: bool) -> str:
    return "different" if different_shuffle_per_trial else "same"


def shuffle_stem_suffix(shuffle_type: str, different_shuffle_per_trial: bool, shuffle_id: int) -> str:
    """Suffix that generate_trials_ablation_shuffle.py appends to the vectorization cache stem."""
    return f"_{shuffle_type}_{shuffle_mode_token(different_shuffle_per_trial)}_shuffle{int(shuffle_id):04d}"


def mouse_cache_dir(data_root: Path, mouse: str, cache_dir: Optional[Path]) -> Path:
    return Path(cache_dir) if cache_dir is not None else Path(data_root) / mouse / "cache"


# ---------------------------------------------------------------------------
# Zigzag features
# ---------------------------------------------------------------------------

@dataclass
class ZigzagFeatures:
    """Vectorization features read from the per-mouse caches.

    For windows starting at frame 0 with a common length L (video_label,
    video_id) the cache for clip L is used: the exact ``..._clip<L>`` cache if
    present, otherwise (frame-aligned methods such as Turnover) the unclipped
    cache cut to L frames.  Other windows (segment_id) are cut from the
    unclipped cache and need a frame-aligned method.
    """

    data_root: Path
    method: str
    p_active: int
    per_trial_thresh: bool
    cache_dir: Optional[Path] = None
    stem_suffix: str = ""
    _cache: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = field(default_factory=dict, repr=False)
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def _load(self, mouse: str, clip: Optional[int]) -> Dict[str, Any]:
        key = (mouse, clip)
        if key not in self._cache:
            loaded = load_feature_cache(
                data_root=self.data_root,
                mouse_name=mouse,
                method=self.method,
                p_active=self.p_active,
                per_trial_thresh=self.per_trial_thresh,
                clip_frames=clip,
                cache_dir=self.cache_dir,
                stem_suffix=self.stem_suffix,
            )
            loaded["row_of_trial"] = {int(t): i for i, t in enumerate(loaded["trial_ids"])}
            self._cache[key] = loaded
            self.sources[f"{mouse}|clip={clip if clip is not None else 'none'}"] = {
                "cache_path": loaded["cache_path"],
                "source": loaded["source"],
            }
        return self._cache[key]

    def _rows(self, loaded: Dict[str, Any], mouse: str, trial_ids: Sequence[int]) -> np.ndarray:
        lookup = loaded["row_of_trial"]
        missing = [int(t) for t in trial_ids if int(t) not in lookup]
        if missing:
            raise RuntimeError(
                f"{len(missing)} trial(s) of {mouse} are missing from {loaded['cache_path']} "
                f"(e.g. {missing[:5]}); recompute the cache"
            )
        return np.asarray([lookup[int(t)] for t in trial_ids], dtype=np.int64)

    def features(self, samples: pd.DataFrame) -> np.ndarray:
        parts: List[np.ndarray] = []
        order: List[np.ndarray] = []
        for mouse, idx in samples.groupby("mouse", sort=False).indices.items():
            s = samples.iloc[idx]
            starts = s["start_frame"].to_numpy(dtype=np.int64)
            lengths = s["n_frames"].to_numpy(dtype=np.int64)
            if np.all(starts == 0) and len(np.unique(lengths)) == 1:
                loaded = self._load(str(mouse), int(lengths[0]))
                x = loaded["features"][self._rows(loaded, str(mouse), s["trial_id"])]
            else:
                if not is_frame_aligned_method(self.method):
                    raise RuntimeError(
                        f"Windows that do not start at frame 0 (segments) need a frame-aligned "
                        f"vectorization; '{self.method}' is not registered in utils.FRAME_ALIGNED_BLOCKS"
                    )
                loaded = self._load(str(mouse), None)
                full = loaded["features"][self._rows(loaded, str(mouse), s["trial_id"])]
                x = slice_frame_aligned_features(full, self.method, starts, lengths)
            parts.append(np.asarray(x, dtype=np.float64))
            order.append(np.asarray(idx))
        out = np.empty((len(samples), parts[0].shape[1]), dtype=np.float64)
        for idx, x in zip(order, parts):
            out[idx] = x
        return out


# ---------------------------------------------------------------------------
# Grid datasets
# ---------------------------------------------------------------------------

def grid_dataset(samples: pd.DataFrame, y: np.ndarray, *, normalize: bool, grid_paths: Optional[Sequence[str]] = None):
    """Lazy dataset of grid windows [start_frame, start_frame + n_frames).

    Frames beyond the trial's valid frames are zero-padded (cannot happen with
    the default windows, which never exceed the shortest included trial).
    """
    from classification_models import SegmentGridDataset

    return SegmentGridDataset(
        grid_paths=list(grid_paths) if grid_paths is not None else samples["grid_path"].tolist(),
        y=np.asarray(y, dtype=np.int64),
        valid_frames=samples["valid_frames"].to_numpy(dtype=np.int64),
        start_frames=samples["start_frame"].to_numpy(dtype=np.int64),
        seg_lengths=samples["n_frames"].to_numpy(dtype=np.int64),
        normalize_by_trial=normalize,
    )


# ---------------------------------------------------------------------------
# Shuffle caches (written by generate_trials_ablation_shuffle.py)
# ---------------------------------------------------------------------------

@dataclass
class ShuffleSource:
    """Identifies a set of shuffles produced by generate_trials_ablation_shuffle.py.

    The shuffled grids are stored next to the shuffled vectorization caches,
    under a folder named after the zigzag configuration used at generation
    time (method, p_active, threshold mode and clip).  The grids themselves do
    not depend on that configuration.
    """

    data_root: Path
    shuffle_type: str
    different_shuffle_per_trial: bool
    method: str
    p_active: int
    per_trial_thresh: bool
    generation_clip: Optional[int]
    cache_dir: Optional[Path] = None

    @property
    def mode(self) -> str:
        return shuffle_mode_token(self.different_shuffle_per_trial)

    def base_stem(self, mouse: str) -> str:
        return build_vectorization_cache_stem(
            mouse, self.method, self.p_active, self.per_trial_thresh, self.generation_clip
        )

    def available_ids(self, mouse: str) -> List[int]:
        mcd = mouse_cache_dir(self.data_root, mouse, self.cache_dir)
        prefix = f"{self.base_stem(mouse)}_{self.shuffle_type}_{self.mode}_shuffle"
        ids = []
        for p in mcd.glob(f"{prefix}*.npz"):
            try:
                ids.append(int(p.stem.rsplit("shuffle", 1)[1]))
            except (IndexError, ValueError):
                pass
        return sorted(ids)

    def common_ids(self, mice: Sequence[str]) -> List[int]:
        common: Optional[set] = None
        for m in mice:
            ids = set(self.available_ids(m))
            common = ids if common is None else common & ids
        return sorted(common or [])

    def grid_paths(self, samples: pd.DataFrame, shuffle_id: int) -> List[str]:
        suffix = shuffle_stem_suffix(self.shuffle_type, self.different_shuffle_per_trial, shuffle_id)
        paths: List[str] = []
        missing: List[str] = []
        for mouse, tid in zip(samples["mouse"], samples["trial_id"]):
            mcd = mouse_cache_dir(self.data_root, str(mouse), self.cache_dir)
            p = build_shuffled_grid_cache_path(mcd, f"{self.base_stem(str(mouse))}{suffix}", int(tid))
            if not p.exists():
                missing.append(str(p))
            paths.append(str(p))
        if missing:
            raise FileNotFoundError(f"{len(missing)} shuffled grid file(s) missing, e.g. {missing[:3]}")
        return paths

    def zigzag_features(self, shuffle_id: int) -> ZigzagFeatures:
        return ZigzagFeatures(
            data_root=self.data_root,
            method=self.method,
            p_active=self.p_active,
            per_trial_thresh=self.per_trial_thresh,
            cache_dir=self.cache_dir,
            stem_suffix=shuffle_stem_suffix(self.shuffle_type, self.different_shuffle_per_trial, shuffle_id),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shuffle_type": self.shuffle_type,
            "different_shuffle_per_trial": self.different_shuffle_per_trial,
            "shuffle_mode": self.mode,
            "generation_method": self.method,
            "generation_p_active": self.p_active,
            "generation_per_trial_thresh": self.per_trial_thresh,
            "generation_clip_frames": self.generation_clip,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else "<data_root>/<mouse>/cache",
        }


def sample_shuffle_ids(available: Sequence[int], n_shuffles: int, seed: int) -> List[int]:
    """Pick ``n_shuffles`` shuffle IDs (shared by all mice) reproducibly."""
    if len(available) < n_shuffles:
        raise RuntimeError(
            f"Need {n_shuffles} shuffles available for every mouse, found {len(available)} common IDs "
            f"({list(available)}). Generate more with generate_trials_ablation_shuffle.py"
        )
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in rng.choice(np.asarray(available), size=n_shuffles, replace=False))
