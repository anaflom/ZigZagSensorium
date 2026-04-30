#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate and cache shuffled zigzag vectorizations for ablation studies.

For each mouse the script:
1. Discovers existing cached shuffle IDs for the requested configuration.
2. Appends new shuffles starting from max(existing_id) + 1.
3. Shuffles grid data (time / spatial / phase), computes zigzag persistence,
    vectorizes and saves a .npz cache per trial set.
4. Writes / updates a per-mouse manifest JSON that classification scripts
   can use to validate available shuffles before running.

--n-shuffles N means the script ensures N shuffles *exist in total* after it
finishes (not that it adds N new ones).  Mice that already have >= N cached
shuffles are skipped entirely.  New shuffle IDs are appended from
max(existing_id) + 1 so existing caches are never overwritten.

This script intentionally does NO classification so it can run on a CPU-only
partition with modest memory, and can be killed and resumed without loss.

Usage examples
--------------
    # Ensure 3 shuffles exist for all mice (time shuffle)
    python generate_trials_ablation_shuffle.py --n-shuffles 3

    # Ensure 5 spatial shuffles exist for a single mouse
    python generate_trials_ablation_shuffle.py --n-shuffles 5 --shuffle-type spatial \\
         --mice dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce

    # Force recompute all shuffles up to a target of 2
    python generate_trials_ablation_shuffle.py --n-shuffles 2 --force-recompute true
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shuffle_utils import (
    compute_threshold_from_grid_sample,
    compute_zigzag_from_grid,
    derive_trial_shuffle_seed,
    shuffle_grid_phase,
    shuffle_grid_spatial_dimensions,
    shuffle_grid_time_dimension,
    validate_mice_for_ablation,
)
from utils import (
    _build_zz_folder,
    _discover_mice,
    _opt_csv_list,
    _opt_int,
    _resolve_mouse_cache_dir,
    _str2bool,
    build_vectorization_cache_stem,
    create_vectorization,
    load_labelled_barcodes,
    load_labelled_grid_paths,
)


# ---------------------------------------------------------------------------
# Cache index helpers
# ---------------------------------------------------------------------------

def _shuffle_cache_stem(base_stem: str, shuffle_type: str) -> str:
    """Build deterministic cache stem prefix encoding shuffle type."""
    return f"{base_stem}_{shuffle_type}"


def _shuffle_mode_token(different_shuffle_per_trial: bool) -> str:
    return "different" if different_shuffle_per_trial else "same"


def _shuffle_cache_stem_with_mode(
    base_stem: str,
    shuffle_type: str,
    shuffle_id: int,
    different_shuffle_per_trial: bool,
) -> str:
    mode_token = _shuffle_mode_token(different_shuffle_per_trial)
    return f"{_shuffle_cache_stem(base_stem, shuffle_type)}_{mode_token}_shuffle{shuffle_id:04d}"


def _existing_shuffle_ids(
    cache_dir: Path,
    base_stem: str,
    shuffle_type: str,
    different_shuffle_per_trial: bool,
) -> List[int]:
    """Return sorted list of shuffle IDs that already have a valid .npz cache."""
    ids = []
    stem_prefix = _shuffle_cache_stem(base_stem, shuffle_type)
    mode_token = _shuffle_mode_token(different_shuffle_per_trial)
    for p in cache_dir.glob(f"{stem_prefix}_{mode_token}_shuffle*.npz"):
        stem = p.stem  # e.g. "..._time_shuffle0002"
        try:
            idx = int(stem.rsplit("shuffle", 1)[1])
            ids.append(idx)
        except (IndexError, ValueError):
            pass
    return sorted(ids)


def _next_shuffle_ids(
    cache_dir: Path,
    base_stem: str,
    shuffle_type: str,
    different_shuffle_per_trial: bool,
    n_target: int,
) -> List[int]:
    """Return shuffle IDs to generate to satisfy a target pool size.

    Behavior:
    - First fill missing IDs in [0, n_target-1] so the target window is dense.
      Example: existing [0, 2], n_target=3 -> generate [1].
    - If there are still fewer than n_target total cached IDs, append new IDs from
      max(existing)+1.
    """
    existing = _existing_shuffle_ids(
        cache_dir,
        base_stem,
        shuffle_type,
        different_shuffle_per_trial,
    )
    existing_set = set(existing)

    if len(existing_set) >= n_target:
        return []

    planned: List[int] = []

    # Fill holes in the target range first.
    for sid in range(n_target):
        if sid not in existing_set:
            planned.append(sid)

    # If caches are sparse and outside the target range, we may still need extras.
    total_after_fill = len(existing_set) + len(planned)
    if total_after_fill < n_target:
        next_id = (max(existing_set) + 1) if existing_set else 0
        while total_after_fill < n_target:
            if next_id not in existing_set:
                planned.append(next_id)
                total_after_fill += 1
            next_id += 1

    return planned


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "shuffle_manifest.json"


def _latest_manifest_path(cache_dir: Path, shuffle_type: Optional[str]) -> Optional[Path]:
    legacy = cache_dir / MANIFEST_FILENAME
    if legacy.exists():
        return legacy
    return None


def _load_manifest(cache_dir: Path, shuffle_type: Optional[str]) -> Dict[str, Any]:
    mp = _latest_manifest_path(cache_dir, shuffle_type)
    if mp is not None:
        with open(mp, encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def _save_manifest(cache_dir: Path, manifest: Dict[str, Any], shuffle_type: str) -> Path:
    mp = cache_dir / MANIFEST_FILENAME
    tmp_path = cache_dir / f"{MANIFEST_FILENAME}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    tmp_path.replace(mp)
    return mp


def _manifest_key(
    shuffle_type: str,
    method: str,
    p_active: int,
    per_trial_thresh: bool,
    clip_frames: int,
    different_shuffle_per_trial: bool,
) -> str:
    pt = "pertrial1" if per_trial_thresh else "pertrial0"
    mode_token = _shuffle_mode_token(different_shuffle_per_trial)
    return f"{shuffle_type}__{method}__{p_active}__{pt}__clip{clip_frames}__shufflemode-{mode_token}"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate shuffled zigzag vectorization caches for ablation studies"
    )

    # Paths
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root folder with neural data (one subdirectory per mouse)")
    p.add_argument("--meta-root", type=Path, required=True,
                   help="Root folder with trial metadata CSVs")
    p.add_argument("--cache-dir", type=lambda x: Path(x) if x else None, default=None,
                   help="Override cache directory (default: <data-root>/<mouse>/cache)")

    # Mouse selection
    p.add_argument("--mice", type=_opt_csv_list, default=None,
                   help="Comma-separated mouse names, or None for all discovered mice")

    # Shuffle parameters
    p.add_argument("--n-shuffles", type=int, required=True,
                   help="Target total number of shuffles to have cached after this run "
                        "(mice already at or above this count are skipped)")
    p.add_argument(
        "--shuffle-type",
        type=str,
        choices=["time", "spatial", "phase"],
        default="time",
        help="Shuffle mode: 'time' permutes per-voxel time axis; "
             "'spatial' permutes 3D positions reused across frames; "
             "'phase' applies FFT phase shifting",
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed; shuffle i uses seed + i*1000")
    p.add_argument(
        "--different-shuffle-per-trial",
        type=_str2bool,
        default=True,
        help="If True, derive a different deterministic shuffle for each trial. "
             "If False, reuse the same shuffle for all trials within a shuffle ID.",
    )

    # Vectorization / zigzag parameters
    p.add_argument("--p-active", type=int, default=30,
                   help="Percentile of positive activations for threshold")
    p.add_argument("--per-trial-thresh", type=_str2bool, default=True,
                   help="Compute threshold independently per trial (True/False)")
    p.add_argument("--vectorization-method", type=str, default="Turnover",
                   help="Zigzag vectorization method name")
    p.add_argument("--clip-frames", type=_opt_int, default=None,
                   help="Fixed temporal clip (None = auto-detect from data)")
    p.add_argument("--max-trials", type=_opt_int, default=None,
                   help="Max trials per mouse (None = all)")
    p.add_argument("--max-dim", type=int, default=2,
                   help="Max homology dimension for zigzag (0/1/2)")
    p.add_argument("--grid-subdir", type=str, default="trials_grid",
                   help="Grid subdirectory name under each mouse folder")
    p.add_argument("--zz-folder", type=str, default=None,
                   help="Zigzag folder name (auto-derived from p_active if None)")

    # Control
    p.add_argument("--force-recompute", type=_str2bool, default=False,
                   help="Force recomputation even if cache already exists")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress every N trials after first 3 (default 10)")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Number of parallel worker processes over trials (default 1)")

    return p


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _resolve_global_clip(
    eligible_mice: List[str],
    data_root: Path,
    meta_root: Path,
    zz_folder: str,
    clip_frames: Optional[int],
    max_trials: Optional[int],
) -> int:
    if clip_frames is not None:
        return int(clip_frames)

    print("Pre-scanning valid_frames to determine global clip_frames ...", flush=True)
    min_frames_list: List[int] = []
    for mouse_name in eligible_mice:
        try:
            _, _, _, valid_frames = load_labelled_barcodes(
                data_root, meta_root, mouse_name, zz_folder, max_trials=max_trials
            )
            if len(valid_frames) > 0:
                min_frames_list.append(int(valid_frames.min()))
        except Exception as exc:
            print(f"  Warning: could not scan {mouse_name}: {exc}", flush=True)
    if not min_frames_list:
        raise RuntimeError("Could not determine global clip_frames from any mouse")
    gc_val = min(min_frames_list)
    print(f"  -> global_clip_frames = {gc_val}", flush=True)
    return gc_val


def _apply_shuffle(
    grid: np.ndarray,
    shuffle_type: str,
    seed: int,
) -> np.ndarray:
    if shuffle_type == "time":
        return shuffle_grid_time_dimension(grid, seed=seed)
    if shuffle_type == "spatial":
        return shuffle_grid_spatial_dimensions(grid, seed=seed)
    if shuffle_type == "phase":
        return shuffle_grid_phase(grid, seed=seed)
    raise ValueError(f"Unknown shuffle_type: {shuffle_type!r}")


def _fmt_duration(seconds: float) -> str:
    """Format duration as H:MM:SS for readable ETAs."""
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def _should_log_trial(trial_number: int, n_trials: int, progress_every: int) -> bool:
    """Log first few trials, then every N trials, and always the last trial."""
    if trial_number <= 3:
        return True
    if trial_number == n_trials:
        return True
    return (trial_number % progress_every) == 0


def _process_single_trial(
    gpath: Path,
    shuffle_type: str,
    shuffle_seed: int,
    trial_id: int,
    different_shuffle_per_trial: bool,
    p_active: int,
    per_trial_thresh: bool,
    global_threshold: Optional[float],
    max_dim: int,
) -> Tuple[bool, Optional[List[Tuple[int, float, float]]], str, float]:
    """Process one trial and return (ok, bars, error, elapsed_sec)."""
    t0 = time.time()
    try:
        grid = np.load(gpath)
        trial_seed = (
            derive_trial_shuffle_seed(shuffle_seed, trial_id)
            if different_shuffle_per_trial
            else shuffle_seed
        )
        shuffled_grid = _apply_shuffle(grid, shuffle_type, seed=trial_seed)
        del grid

        if per_trial_thresh:
            thresh = compute_threshold_from_grid_sample([gpath], p_active, n_sample=1)
        else:
            if global_threshold is None:
                raise RuntimeError("global_threshold is None while per_trial_thresh=False")
            thresh = global_threshold

        bars = compute_zigzag_from_grid(
            shuffled_grid,
            threshold=thresh,
            p_active=p_active,
            max_dim=max_dim,
        )
        del shuffled_grid
        return True, bars, "", time.time() - t0
    except Exception:
        return False, None, traceback.format_exc(), time.time() - t0


def _generate_for_mouse(
    *,
    mouse_name: str,
    data_root: Path,
    meta_root: Path,
    cache_dir: Optional[Path],
    zz_folder: str,
    grid_subdir: str,
    shuffle_type: str,
    shuffle_ids: List[int],
    base_seed: int,
    different_shuffle_per_trial: bool,
    method: str,
    p_active: int,
    per_trial_thresh: bool,
    global_clip: int,
    max_trials: Optional[int],
    max_dim: int,
    force_recompute: bool,
    progress_every: int,
    num_workers: int,
) -> Dict[str, Any]:
    """Generate shuffled vectorization caches for one mouse.

    Returns a summary dict with generated / skipped / failed IDs.
    """
    from dataclasses import dataclass

    @dataclass
    class _FakeState:
        cache_dir: Optional[Path]
        data_root: Path

    fs = _FakeState(cache_dir=cache_dir, data_root=data_root)
    mouse_cache_dir = _resolve_mouse_cache_dir(fs, mouse_name)
    mouse_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*80}", flush=True)
    print(f"Mouse: {mouse_name}", flush=True)
    print(f"  Cache dir : {mouse_cache_dir}", flush=True)
    print(f"  Shuffle IDs to generate: {shuffle_ids}", flush=True)

    # Load barcodes and grid paths
    t0 = time.time()
    print(f"  [{_ts()}] Loading metadata and grid paths ...", flush=True)
    try:
        barcodes, labels, trial_ids, valid_frames = load_labelled_barcodes(
            data_root, meta_root, mouse_name, zz_folder, max_trials=max_trials
        )
        grid_paths, grid_labels, grid_trial_ids, grid_valid_frames = load_labelled_grid_paths(
            data_root, meta_root, mouse_name, grid_subdir=grid_subdir
        )
    except Exception as exc:
        print(f"  FAILED to load data: {exc}", flush=True)
        return {"generated": [], "skipped": [], "failed": shuffle_ids, "error": str(exc)}

    if len(grid_paths) == 0 or len(barcodes) == 0:
        msg = f"no grids ({len(grid_paths)}) or barcodes ({len(barcodes)})"
        print(f"  Skipping: {msg}", flush=True)
        return {"generated": [], "skipped": shuffle_ids, "failed": [], "error": msg}

    # Align trial IDs
    vec_idx = {int(tid): i for i, tid in enumerate(trial_ids)}
    grid_idx = {int(tid): i for i, tid in enumerate(grid_trial_ids)}
    common_trial_ids = [tid for tid in trial_ids if int(tid) in grid_idx]
    if not common_trial_ids:
        msg = "no overlapping trial IDs"
        print(f"  Skipping: {msg}", flush=True)
        return {"generated": [], "skipped": shuffle_ids, "failed": [], "error": msg}

    vec_take = np.array([vec_idx[int(tid)] for tid in common_trial_ids], dtype=np.int64)
    grid_take = np.array([grid_idx[int(tid)] for tid in common_trial_ids], dtype=np.int64)

    labels_common = labels[vec_take]
    trial_ids_common = np.array([int(trial_ids[i]) for i in vec_take], dtype=np.int64)
    valid_frames_common = valid_frames[vec_take]
    grid_paths_common = [grid_paths[i] for i in grid_take]

    n_trials = len(common_trial_ids)
    print(f"  [{_ts()}] Data loaded in {time.time()-t0:.1f}s: {n_trials} trials", flush=True)

    # Build base stem once
    base_stem = build_vectorization_cache_stem(
        mouse_name=mouse_name,
        method=method,
        p_active=p_active,
        per_trial_thresh=per_trial_thresh,
        clip_frames=global_clip,
    )

    generated: List[int] = []
    skipped: List[int] = []
    failed: List[int] = []

    # Compute global threshold once if not per-trial
    global_threshold: Optional[float] = None
    if not per_trial_thresh:
        global_threshold = compute_threshold_from_grid_sample(
            grid_paths_common, p_active, n_sample=min(20, n_trials)
        )
        print(f"  Global threshold: {global_threshold:.6f}", flush=True)

    for shuffle_id in shuffle_ids:
        shuffle_stem = _shuffle_cache_stem_with_mode(
            base_stem,
            shuffle_type,
            shuffle_id,
            different_shuffle_per_trial,
        )
        cache_path = mouse_cache_dir / f"{shuffle_stem}.npz"

        if cache_path.exists() and not force_recompute:
            print(f"  [shuffle {shuffle_id}] SKIP (cache exists: {cache_path.name})", flush=True)
            skipped.append(shuffle_id)
            continue

        shuffle_seed = base_seed + shuffle_id * 1000
        print(
            f"\n  [shuffle {shuffle_id}] seed={shuffle_seed} "
            f"mode={_shuffle_mode_token(different_shuffle_per_trial)}",
            flush=True,
        )

        t_shuf = time.time()
        barcodes_shuffled: List[List[Tuple[int, float, float]]] = []
        ok = True

        if num_workers <= 1:
            for trial_idx, gpath in enumerate(grid_paths_common):
                trial_number = trial_idx + 1
                ok_trial, bars, err_text, elapsed_trial = _process_single_trial(
                    gpath=gpath,
                    shuffle_type=shuffle_type,
                    shuffle_seed=shuffle_seed,
                    trial_id=int(trial_ids_common[trial_idx]),
                    different_shuffle_per_trial=different_shuffle_per_trial,
                    p_active=p_active,
                    per_trial_thresh=per_trial_thresh,
                    global_threshold=global_threshold,
                    max_dim=max_dim,
                )
                if not ok_trial or bars is None:
                    print(
                        f"    ERROR at trial {trial_idx} ({gpath.name})",
                        flush=True,
                    )
                    if err_text:
                        print(err_text, flush=True)
                    ok = False
                    break

                barcodes_shuffled.append(bars)
                if _should_log_trial(trial_number, n_trials, progress_every):
                    elapsed_shuffle = time.time() - t_shuf
                    avg_sec_per_trial = elapsed_shuffle / float(trial_number)
                    eta_sec = avg_sec_per_trial * float(n_trials - trial_number)
                    print(
                        f"    [{_ts()}] trial {trial_number}/{n_trials} "
                        f"zigzag done (bars={len(bars)}, "
                        f"trial={elapsed_trial:.2f}s, "
                        f"avg={avg_sec_per_trial:.2f}s/trial, "
                        f"eta={_fmt_duration(eta_sec)})",
                        flush=True,
                    )
        else:
            print(
                f"  [shuffle {shuffle_id}] parallel trial processing with {num_workers} workers",
                flush=True,
            )
            bars_by_trial: List[Optional[List[Tuple[int, float, float]]]] = [None] * n_trials
            completed = 0
            with cf.ProcessPoolExecutor(max_workers=num_workers) as pool:
                future_to_idx = {
                    pool.submit(
                        _process_single_trial,
                        gpath,
                        shuffle_type,
                        shuffle_seed,
                        int(trial_ids_common[i]),
                        different_shuffle_per_trial,
                        p_active,
                        per_trial_thresh,
                        global_threshold,
                        max_dim,
                    ): i
                    for i, gpath in enumerate(grid_paths_common)
                }

                for fut in cf.as_completed(future_to_idx):
                    trial_idx = future_to_idx[fut]
                    trial_number = trial_idx + 1
                    try:
                        ok_trial, bars, err_text, elapsed_trial = fut.result()
                    except Exception:
                        ok_trial, bars, err_text, elapsed_trial = False, None, traceback.format_exc(), 0.0

                    if not ok_trial or bars is None:
                        print(
                            f"    ERROR at trial {trial_idx} ({grid_paths_common[trial_idx].name})",
                            flush=True,
                        )
                        if err_text:
                            print(err_text, flush=True)
                        ok = False
                        for pending in future_to_idx:
                            pending.cancel()
                        break

                    bars_by_trial[trial_idx] = bars
                    completed += 1
                    if _should_log_trial(completed, n_trials, progress_every):
                        elapsed_shuffle = time.time() - t_shuf
                        avg_sec_per_trial = elapsed_shuffle / float(completed)
                        eta_sec = avg_sec_per_trial * float(n_trials - completed)
                        print(
                            f"    [{_ts()}] trial {trial_number}/{n_trials} done "
                            f"(bars={len(bars)}, trial={elapsed_trial:.2f}s, "
                            f"completed={completed}/{n_trials}, "
                            f"avg={avg_sec_per_trial:.2f}s/trial, "
                            f"eta={_fmt_duration(eta_sec)})",
                            flush=True,
                        )

            if ok:
                if any(b is None for b in bars_by_trial):
                    ok = False
                    print("    ERROR: missing trial outputs after parallel processing", flush=True)
                else:
                    barcodes_shuffled = [b for b in bars_by_trial if b is not None]

        if not ok:
            failed.append(shuffle_id)
            print(f"  [shuffle {shuffle_id}] FAILED after {time.time()-t_shuf:.1f}s", flush=True)
            continue

        # Vectorize and save
        print(
            f"  [{_ts()}] [{shuffle_id}] Vectorizing {len(barcodes_shuffled)} barcodes "
            f"(method={method}) ...",
            flush=True,
        )
        create_vectorization(
            barcodes_shuffled,
            method,
            clip_frames=global_clip,
            output_folder=mouse_cache_dir,
            cache_stem=shuffle_stem,
            mouse_name=mouse_name,
            labels=labels_common,
            trial_ids=trial_ids_common,
            valid_frames=valid_frames_common,
        )

        elapsed = time.time() - t_shuf
        print(
            f"  [{_ts()}] [shuffle {shuffle_id}] DONE — saved {cache_path.name} ({elapsed:.1f}s)",
            flush=True,
        )
        generated.append(shuffle_id)

    return {
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "n_trials": n_trials,
        "clip_frames_used": global_clip,
        "max_trials_used": max_trials,
        "global_threshold": global_threshold,
    }


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import gc

    parser = _build_parser()
    args = parser.parse_args()

    zz_folder = args.zz_folder or _build_zz_folder(args.p_active, args.per_trial_thresh)

    print("=" * 90, flush=True)
    print("Generate Shuffled Zigzag Vectorizations", flush=True)
    print(f"Timestamp : {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print("=" * 90, flush=True)
    print(f"  data_root           : {args.data_root}", flush=True)
    print(f"  meta_root           : {args.meta_root}", flush=True)
    print(f"  cache_dir override  : {args.cache_dir}", flush=True)
    print(f"  shuffle_type        : {args.shuffle_type}", flush=True)
    print(f"  n_shuffles (target) : {args.n_shuffles}", flush=True)
    print(f"  method              : {args.vectorization_method}", flush=True)
    print(f"  p_active            : {args.p_active}", flush=True)
    print(f"  per_trial_thresh    : {args.per_trial_thresh}", flush=True)
    print(f"  clip_frames         : {args.clip_frames}", flush=True)
    print(f"  max_trials          : {args.max_trials}", flush=True)
    print(f"  max_dim             : {args.max_dim}", flush=True)
    print(f"  seed                : {args.seed}", flush=True)
    print(
        f"  different_shuffle_per_trial : {args.different_shuffle_per_trial}",
        flush=True,
    )
    print(f"  force_recompute     : {args.force_recompute}", flush=True)
    print(f"  progress_every      : {args.progress_every}", flush=True)
    print(f"  num_workers         : {args.num_workers}", flush=True)

    if args.progress_every < 1:
        print("ERROR: --progress-every must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.num_workers < 1:
        print("ERROR: --num-workers must be >= 1", file=sys.stderr)
        sys.exit(2)

    # Discover mice
    discovered = _discover_mice(args.data_root)
    print(f"\nDiscovered mice: {len(discovered)}", flush=True)

    eligible_mice = validate_mice_for_ablation(
        args.mice, args.data_root, args.meta_root, discovered
    )
    print(f"Eligible mice (>=2 labels): {len(eligible_mice)}", flush=True)
    if not eligible_mice:
        print("ERROR: No eligible mice found.", file=sys.stderr)
        sys.exit(1)

    # Determine global clip
    global_clip = _resolve_global_clip(
        eligible_mice,
        args.data_root,
        args.meta_root,
        zz_folder,
        args.clip_frames,
        args.max_trials,
    )

    # Per-mouse: discover existing shuffles and decide what to generate
    print("\n" + "=" * 90, flush=True)
    print("Existing shuffle cache status:", flush=True)

    from dataclasses import dataclass

    @dataclass
    class _FakeState:
        cache_dir: Optional[Path]
        data_root: Path

    fs = _FakeState(cache_dir=args.cache_dir, data_root=args.data_root)

    per_mouse_new_ids: Dict[str, List[int]] = {}
    for mouse_name in eligible_mice:
        mcdir = _resolve_mouse_cache_dir(fs, mouse_name)
        mcdir.mkdir(parents=True, exist_ok=True)
        base_stem = build_vectorization_cache_stem(
            mouse_name=mouse_name,
            method=args.vectorization_method,
            p_active=args.p_active,
            per_trial_thresh=args.per_trial_thresh,
            clip_frames=global_clip,
        )
        existing = _existing_shuffle_ids(
            mcdir,
            base_stem,
            args.shuffle_type,
            args.different_shuffle_per_trial,
        )
        new_ids = _next_shuffle_ids(
            mcdir,
            base_stem,
            args.shuffle_type,
            args.different_shuffle_per_trial,
            args.n_shuffles,
        )
        per_mouse_new_ids[mouse_name] = new_ids
        print(
            f"  {mouse_name}: {len(existing)}/{args.n_shuffles} existing {existing} "
            f"-> will generate {new_ids}"
            + (" (already at target, skipping)" if not new_ids else ""),
            flush=True,
        )

    # Generate
    all_results: Dict[str, Any] = {}
    t_global = time.time()
    for mouse_name in eligible_mice:
        result = _generate_for_mouse(
            mouse_name=mouse_name,
            data_root=args.data_root,
            meta_root=args.meta_root,
            cache_dir=args.cache_dir,
            zz_folder=zz_folder,
            grid_subdir=args.grid_subdir,
            shuffle_type=args.shuffle_type,
            shuffle_ids=per_mouse_new_ids[mouse_name],
            base_seed=args.seed,
            different_shuffle_per_trial=args.different_shuffle_per_trial,
            method=args.vectorization_method,
            p_active=args.p_active,
            per_trial_thresh=args.per_trial_thresh,
            global_clip=global_clip,
            max_trials=args.max_trials,
            max_dim=args.max_dim,
            force_recompute=args.force_recompute,
            progress_every=args.progress_every,
            num_workers=args.num_workers,
        )
        all_results[mouse_name] = result

        # Update manifest
        mcdir = _resolve_mouse_cache_dir(fs, mouse_name)
        manifest = _load_manifest(mcdir, args.shuffle_type)
        mkey = _manifest_key(
            args.shuffle_type,
            args.vectorization_method,
            args.p_active,
            args.per_trial_thresh,
            global_clip,
            args.different_shuffle_per_trial,
        )
        existing_after = _existing_shuffle_ids(
            mcdir,
            build_vectorization_cache_stem(
                mouse_name=mouse_name,
                method=args.vectorization_method,
                p_active=args.p_active,
                per_trial_thresh=args.per_trial_thresh,
                clip_frames=global_clip,
            ),
            args.shuffle_type,
            args.different_shuffle_per_trial,
        )
        entry = manifest.get(mkey, {})
        entry.update({
            "shuffle_type": args.shuffle_type,
            "method": args.vectorization_method,
            "p_active": args.p_active,
            "per_trial_thresh": args.per_trial_thresh,
            "clip_frames_used": global_clip,
            "max_trials_used": args.max_trials,
            "base_seed": args.seed,
            "different_shuffle_per_trial": args.different_shuffle_per_trial,
            "available_ids": existing_after,
            "n_available": len(existing_after),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        })
        if result.get("n_trials") is not None:
            entry["n_trials_per_shuffle"] = result["n_trials"]
        manifest[mkey] = entry
        _save_manifest(mcdir, manifest, args.shuffle_type)

        gc.collect()

    # Summary
    total_elapsed = time.time() - t_global
    print("\n" + "=" * 90, flush=True)
    print(f"DONE — total elapsed: {total_elapsed/60:.1f} min", flush=True)
    print("Summary per mouse:", flush=True)
    all_ok = True
    for mouse_name, res in all_results.items():
        gen = res.get("generated", [])
        skip = res.get("skipped", [])
        fail = res.get("failed", [])
        print(
            f"  {mouse_name}: generated={gen}, skipped={skip}, failed={fail}",
            flush=True,
        )
        if fail:
            all_ok = False

    if not all_ok:
        print("\nWARNING: Some shuffles failed (see above).", file=sys.stderr)
        sys.exit(2)

    print("\nAll shuffles generated successfully.", flush=True)


if __name__ == "__main__":
    main()
