#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Precompute per-frame statistics of grid activity (structure-blind features).

For every eligible trial with a grid file, and every frame, the statistics are
computed across the grid cells, so they ignore where activity is and only keep
how much there is and how it is distributed:

GridMomentsRaw      [mean | CV | skewness] of the raw grid values.
GridMomentsStd      [mean | std | skewness] of the raw grid values (std instead of
                    CV, which is undefined for zero-mean grids such as by_trialmean).
GridMomentsStdDensity  same, on density-normalized cells.
GridMomentsDensity  [mean | CV | skewness] after dividing each cell by its neuron density (the
                    PCS-weighted neuron count used to build the grid), using only
                    cells with density >= --min-density.  The grid sums neuron
                    activity, so raw-grid spread partly reflects how unevenly
                    neurons are packed; this version removes that.
ActiveFraction      fraction of grid points above the zigzag threshold of the
                    trial (read from the zigzag output), one cache per P_ACTIVE
                    and per fixed threshold (--abs-thresholds).

Caches use the vectorization-cache format (features / trial_ids / labels /
valid_frames) with frame-aligned blocks of length max(valid_frames) of the
mouse, zero-padded after each trial's end:

    <mouse>_GridMomentsRaw_clipnone.npz
    <mouse>_GridMomentsDensity_clipnone.npz
    <mouse>_ActiveFraction_p<P>_pertrial<0|1>_clipnone.npz
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew

from classification_data import load_mouse_trials
from compute_grid_activation import get_ranges_from_positions, load_neuron_metadata
from utils import (
    _build_zz_folder,
    _discover_mice,
    _opt_csv_list,
    _str2bool,
    abs_threshold_token,
    build_vectorization_cache_stem,
)


# ---------------------------------------------------------------------------
# Neuron density on the grid
# ---------------------------------------------------------------------------

def _pcs_kernel(s: np.ndarray) -> np.ndarray:
    a = np.abs(s)
    return np.where(a < 1.0, (4.0 - 6.0 * a**2 + 3.0 * a**3) / 6.0, np.where(a < 2.0, (2.0 - a) ** 3 / 6.0, 0.0))


def neuron_density(
    positions: np.ndarray,
    xyz_ranges: Sequence[Tuple[float, float]],
    num_grid: Tuple[int, int, int],
) -> np.ndarray:
    """PCS-weighted neuron count per cell, shape (Nx, Ny, Nz).

    Same kernel and index convention as ``pcs_assign_3d`` in assign_grid.f90
    (after the transpose to (Nx, Ny, Nz) done by compute_grid_activation.py),
    i.e. the grid a recording would give if every neuron had activity 1.
    """
    density = np.zeros(num_grid, dtype=np.float64)
    offsets = np.arange(-2, 3)
    coords = [
        (positions[:, k] - xyz_ranges[k][0]) / ((xyz_ranges[k][1] - xyz_ranges[k][0]) / num_grid[k])
        for k in range(3)
    ]
    for i in range(positions.shape[0]):
        idx, wts = [], []
        for k in range(3):
            ii = np.floor(coords[k][i] + offsets).astype(int)
            ok = (ii >= 0) & (ii < num_grid[k])
            idx.append(ii[ok])
            wts.append(_pcs_kernel(coords[k][i] - ii[ok]))
        density[np.ix_(*idx)] += np.einsum("i,j,k->ijk", *wts)
    return density


MOMENT_SETS: Dict[str, List[str]] = {
    "GridMomentsRaw": ["mean", "cv", "skewness"],
    "GridMomentsDensity": ["mean", "cv", "skewness"],
    "GridMomentsStd": ["mean", "std", "skewness"],
    "GridMomentsStdDensity": ["mean", "std", "skewness"],
}


# ---------------------------------------------------------------------------
# Per-trial statistics
# ---------------------------------------------------------------------------

def _moments(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-frame statistics across rows of x (cells x frames), each (frames,)."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    cv = np.divide(std, mean, out=np.zeros_like(mean), where=mean > 0)
    sk = np.nan_to_num(skew(x, axis=0, bias=True))
    return {"mean": mean, "std": std, "cv": cv, "skewness": sk}


def _trial_stats(
    grid_path: str,
    valid_frames: int,
    density_flat: np.ndarray,
    density_mask: np.ndarray,
    thresholds: Dict[str, Optional[float]],
) -> Dict[str, np.ndarray]:
    grid = np.load(grid_path).astype(np.float64)
    n_frames = int(min(valid_frames, grid.shape[-1]))
    x = grid.reshape(-1, grid.shape[-1])[:, :n_frames]
    raw = _moments(x)
    dens = _moments(x[density_mask] / density_flat[density_mask, None])
    out = {}
    for method, names in MOMENT_SETS.items():
        src = dens if "Density" in method else raw
        out[method] = np.stack([src[n] for n in names])
    for label, thr in thresholds.items():
        out[f"ActiveFraction_{label}"] = None if thr is None else (x > thr).mean(axis=0)[None, :]
    return out


def _read_thresholds(zz_dir: Path, per_trial: bool) -> Tuple[Dict[str, float], Optional[float]]:
    """Activation thresholds used by compute_zigzag.py, as positive grid values.

    zz-top stores the negated percentile (cells active when -value < threshold),
    so a grid point is active when value > -threshold.
    """
    tag = zz_dir.name.replace("trials_", "")
    if per_trial:
        csv = zz_dir / f"{tag}_trial_thresholds.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Per-trial thresholds not found: {csv}")
        df = pd.read_csv(csv)
        return {str(r.trial): -float(r.threshold) for r in df.itertuples(index=False)}, None
    import json

    info = zz_dir / f"{tag}_info.json"
    with open(info, encoding="utf-8") as fp:
        return {}, -float(json.load(fp)["threshold"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save(path: Path, blocks: List[np.ndarray], trials: pd.DataFrame, method: str, extra: Dict) -> None:
    n_blocks = blocks[0].shape[0]
    t_max = int(trials["valid_frames"].max())
    feats = np.zeros((len(blocks), n_blocks * t_max), dtype=np.float64)
    for i, b in enumerate(blocks):
        for k in range(n_blocks):
            feats[i, k * t_max : k * t_max + b.shape[1]] = b[k]
    np.savez_compressed(
        path,
        X=feats,
        features=feats,
        method=np.array(method),
        clip_frames=np.array(-1),
        clip_frames_effective=np.array(t_max),
        mouse_name=np.array(str(trials["mouse"].iloc[0])),
        labels=trials["label"].to_numpy().astype(str),
        trial_ids=trials["trial_id"].to_numpy(dtype=np.int64),
        valid_frames=trials["valid_frames"].to_numpy(dtype=np.int64),
        **{k: np.asarray(v) for k, v in extra.items()},
    )


def process_mouse(args: argparse.Namespace, mouse: str) -> None:
    t0 = time.time()
    trials = load_mouse_trials(args.data_root, args.meta_root, mouse, args.grid_subdir)
    if trials.empty:
        print(f"  {mouse}: no eligible trials with grid files; skipped", flush=True)
        return
    cache_dir = args.cache_dir or args.data_root / mouse / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    first = np.load(trials["grid_path"].iloc[0], mmap_mode="r")
    num_grid = tuple(int(v) for v in first.shape[:3])
    positions, _min, _max = load_neuron_metadata(args.meta_root, mouse)
    ranges_file = args.data_root / mouse / "grid" / "xyz_ranges.npy"
    xyz_ranges = np.load(ranges_file) if ranges_file.exists() else np.asarray(get_ranges_from_positions(positions))
    density = neuron_density(positions, xyz_ranges, num_grid)
    density_flat = density.reshape(-1)
    mask = density_flat >= args.min_density
    print(
        f"  {mouse}: {len(trials)} trials, grid {num_grid}, neurons {len(positions)}, "
        f"density {density_flat.min():.2f}-{density_flat.max():.2f} (sum {density_flat.sum():.0f}), "
        f"cells used for density-normalized moments: {int(mask.sum())}/{mask.size}",
        flush=True,
    )

    # Zigzag thresholds (per trial, keyed by grid filename, or one fixed value)
    specs = [(f"p{p}", p, None) for p in args.p_active] + [
        (abs_threshold_token(a), None, a) for a in args.abs_thresholds
    ]
    thr_by_label: Dict[str, Tuple[Dict[str, float], Optional[float]]] = {}
    spec_by_label = {label: (p, a) for label, p, a in specs}
    for label, p, a in specs:
        zz_dir = args.data_root / mouse / _build_zz_folder(p, args.per_trial_thresh, a)
        try:
            thr_by_label[label] = _read_thresholds(zz_dir, args.per_trial_thresh and a is None)
        except FileNotFoundError as exc:
            print(f"    no thresholds for {label}: {exc}", flush=True)

    def thresholds_for(grid_path: str) -> Dict[str, Optional[float]]:
        name = Path(grid_path).name
        out: Dict[str, Optional[float]] = {}
        for label, (per_trial, glob_thr) in thr_by_label.items():
            out[label] = per_trial.get(name, glob_thr) if per_trial else glob_thr
        return out

    results: List[Optional[Dict[str, np.ndarray]]] = [None] * len(trials)
    jobs = [
        (r.grid_path, int(r.valid_frames), density_flat, mask, thresholds_for(r.grid_path))
        for r in trials.itertuples(index=False)
    ]
    with cf.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_trial_stats, *job): i for i, job in enumerate(jobs)}
        for n_done, fut in enumerate(cf.as_completed(futures), start=1):
            results[futures[fut]] = fut.result()
            if n_done % 100 == 0 or n_done == len(jobs):
                print(f"    {n_done}/{len(jobs)} trials ({time.time() - t0:.0f}s)", flush=True)

    common = {"min_density": args.min_density, "n_cells_density": int(mask.sum()), "num_grid": np.asarray(num_grid)}
    for method, names in MOMENT_SETS.items():
        path = cache_dir / f"{build_vectorization_cache_stem(mouse, method, None, False, None)}.npz"
        _save(path, [r[method] for r in results], trials, method, {**common, "feature_names": names})
        print(f"    saved {path.name}", flush=True)
    for label in thr_by_label:
        p, a = spec_by_label[label]
        blocks = [r[f"ActiveFraction_{label}"] for r in results]
        missing = [str(t) for t, b in zip(trials["trial_id"], blocks) if b is None]
        if missing:
            print(f"    {label}: {len(missing)} trial(s) without a zigzag threshold (e.g. {missing[:5]}); cache not written", flush=True)
            continue
        stem = build_vectorization_cache_stem(mouse, "ActiveFraction", p, args.per_trial_thresh, None, a)
        path = cache_dir / f"{stem}.npz"
        _save(
            path,
            blocks,
            trials,
            "ActiveFraction",
            {"p_active": -1 if p is None else p, "abs_threshold": np.nan if a is None else a,
             "per_trial_thresh": args.per_trial_thresh, "feature_names": ["active_fraction"]},
        )
        print(f"    saved {path.name}", flush=True)
    print(f"  {mouse}: done in {time.time() - t0:.0f}s", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--meta-root", required=True, type=Path)
    p.add_argument("--grid-subdir", default="trials_grid")
    p.add_argument("--cache-dir", default=None, type=Path, help="Default: <data-root>/<mouse>/cache")
    p.add_argument("--mice", default=None, type=_opt_csv_list)
    p.add_argument("--min-density", default=0.5, type=float, help="Minimum neuron density of cells used for GridMomentsDensity")
    p.add_argument("--p-active", default=[30, 50, 70], type=lambda s: [int(v) for v in s.split(",")] if s else [], help="Comma-separated P_ACTIVE values for ActiveFraction")
    p.add_argument("--per-trial-thresh", default=True, type=_str2bool)
    p.add_argument(
        "--abs-thresholds",
        default=[],
        type=lambda s: [float(v) for v in s.split(",")] if s else [],
        help="Comma-separated fixed zigzag thresholds for ActiveFraction (e.g. 0)",
    )
    p.add_argument("--num-workers", default=8, type=int)
    return p


def main() -> int:
    args = build_parser().parse_args()
    mice = args.mice or _discover_mice(args.data_root)
    print(f"compute_grid_stats_cache: {len(mice)} mice, p_active={args.p_active}, min_density={args.min_density}", flush=True)
    failed = []
    for mouse in mice:
        try:
            process_mouse(args, mouse)
        except Exception:
            traceback.print_exc()
            failed.append(mouse)
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
