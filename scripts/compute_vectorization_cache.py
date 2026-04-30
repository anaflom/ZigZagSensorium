#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate vectorization cache files for later analysis/classification scripts.

This script precomputes one cache file per mouse for a given configuration:
- vectorization method
- p_active
- per_trial_thresh
- clip_frames (including None/full-trial mode)

It uses per-trial parallel processing with concurrent.futures.ProcessPoolExecutor,
matching the backend family used in generate_trials_ablation_shuffle.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute vectorization caches (strict-cache workflow)")

    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--meta-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=lambda x: Path(x) if x else None, default=None)

    p.add_argument("--mice", type=_opt_csv_list, default=None,
                   help="Comma-separated mouse names; default is all discovered mice")

    p.add_argument("--vectorization-method", type=str, required=True)
    p.add_argument("--p-active", type=int, required=True)
    p.add_argument("--per-trial-thresh", type=_str2bool, required=True)
    p.add_argument("--clip-frames", type=_opt_int, default=None,
                   help="Optional clip length. Use None for full-trial mode")
    p.add_argument("--max-trials", type=_opt_int, default=None)

    p.add_argument("--num-workers", type=int, default=1,
                   help="Number of parallel workers over trials")
    p.add_argument("--progress-every", type=int, default=20,
                   help="Progress cadence in completed trials")
    p.add_argument("--force-recompute", type=_str2bool, default=False)

    return p


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def _should_log_trial(trial_number: int, n_trials: int, progress_every: int) -> bool:
    if trial_number <= 3:
        return True
    if trial_number == n_trials:
        return True
    return (trial_number % progress_every) == 0


def _infer_frames_from_metadata(valid_frames: Sequence[int]) -> int:
    """Derive effective frame length from metadata valid_frames column."""
    vf = np.asarray(valid_frames, dtype=np.int64)
    if vf.size == 0:
        raise RuntimeError("No valid_frames available in metadata")
    if int(vf.min()) <= 0:
        raise RuntimeError(f"Invalid valid_frames values found: min={int(vf.min())}")
    # Use the largest valid frame count so we avoid clipping long trials in full-trial mode.
    return int(vf.max())


def _vectorize_single_trial(
    barcode: Sequence[Tuple[float, float, float]],
    method: str,
    effective_clip_frames: Optional[int],
) -> Tuple[bool, Optional[np.ndarray], str, float]:
    """Return (ok, feature_row, error_text, elapsed_sec)."""
    t0 = time.time()
    try:
        out = create_vectorization(
            [list(barcode)],
            method,
            clip_frames=effective_clip_frames,
            output_folder=None,
        )
        row = np.asarray(out["features"])  # shape (1, n_features)
        if row.ndim != 2 or row.shape[0] != 1:
            raise RuntimeError(f"Unexpected feature row shape: {row.shape}")
        return True, np.asarray(row[0], dtype=np.float64), "", time.time() - t0
    except Exception:
        return False, None, traceback.format_exc(), time.time() - t0


def _vectorize_trials_parallel(
    barcodes: Sequence[Sequence[Tuple[float, float, float]]],
    method: str,
    effective_clip_frames: Optional[int],
    num_workers: int,
    progress_every: int,
) -> np.ndarray:
    n_trials = len(barcodes)
    if n_trials == 0:
        raise RuntimeError("No barcodes available for vectorization")

    rows: List[Optional[np.ndarray]] = [None] * n_trials
    t_start = time.time()

    if num_workers <= 1:
        for idx, barcode in enumerate(barcodes):
            ok, row, err_text, elapsed_trial = _vectorize_single_trial(
                barcode,
                method,
                effective_clip_frames,
            )
            if not ok or row is None:
                raise RuntimeError(
                    f"Vectorization failed at trial index {idx}.\n{err_text}"
                )
            rows[idx] = row

            trial_number = idx + 1
            if _should_log_trial(trial_number, n_trials, progress_every):
                elapsed_total = time.time() - t_start
                avg = elapsed_total / float(trial_number)
                eta = avg * float(n_trials - trial_number)
                print(
                    f"    [{_ts()}] trial {trial_number}/{n_trials} done "
                    f"(trial={elapsed_trial:.2f}s, avg={avg:.2f}s, eta={_fmt_duration(eta)})",
                    flush=True,
                )
    else:
        completed = 0
        print(
            f"  [{_ts()}] parallel trial processing with {num_workers} workers",
            flush=True,
        )

        with cf.ProcessPoolExecutor(max_workers=num_workers) as pool:
            future_to_idx = {
                pool.submit(
                    _vectorize_single_trial,
                    list(barcode),
                    method,
                    effective_clip_frames,
                ): idx
                for idx, barcode in enumerate(barcodes)
            }

            for fut in cf.as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    ok, row, err_text, elapsed_trial = fut.result()
                except Exception:
                    ok, row, err_text, elapsed_trial = False, None, traceback.format_exc(), 0.0

                if not ok or row is None:
                    for pending in future_to_idx:
                        pending.cancel()
                    raise RuntimeError(
                        f"Vectorization failed at trial index {idx}.\n{err_text}"
                    )

                rows[idx] = row
                completed += 1
                if _should_log_trial(completed, n_trials, progress_every):
                    elapsed_total = time.time() - t_start
                    avg = elapsed_total / float(completed)
                    eta = avg * float(n_trials - completed)
                    print(
                        f"    [{_ts()}] trial {idx + 1}/{n_trials} done "
                        f"(completed={completed}/{n_trials}, trial={elapsed_trial:.2f}s, "
                        f"avg={avg:.2f}s, eta={_fmt_duration(eta)})",
                        flush=True,
                    )

    if any(r is None for r in rows):
        raise RuntimeError("Missing trial outputs after parallel vectorization")

    xmat = np.vstack([r for r in rows if r is not None])
    return np.nan_to_num(np.asarray(xmat, dtype=np.float64))


def _save_cache_npz(
    *,
    cache_path: Path,
    xmat: np.ndarray,
    method: str,
    clip_frames_for_key: Optional[int],
    clip_frames_effective: Optional[int],
    mouse_name: str,
    labels: Sequence[str],
    trial_ids: Sequence[int],
    valid_frames: Sequence[int],
) -> None:
    payload: Dict[str, Any] = {
        "X": xmat,
        "features": xmat,
        "method": np.array(method),
        "clip_frames": np.array(-1 if clip_frames_for_key is None else int(clip_frames_for_key)),
        "clip_frames_effective": np.array(-1 if clip_frames_effective is None else int(clip_frames_effective)),
        "mouse_name": np.array(mouse_name),
        "labels": np.asarray(labels),
        "trial_ids": np.asarray([int(t) for t in trial_ids]),
        "valid_frames": np.asarray(valid_frames),
    }
    np.savez_compressed(cache_path, **payload)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.num_workers < 1:
        print("ERROR: --num-workers must be >= 1", file=sys.stderr)
        return 2
    if args.progress_every < 1:
        print("ERROR: --progress-every must be >= 1", file=sys.stderr)
        return 2

    zz_folder = _build_zz_folder(args.p_active, args.per_trial_thresh)

    print("=" * 90, flush=True)
    print("Generate Vectorization Cache", flush=True)
    print(f"Timestamp          : {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"data_root          : {args.data_root}", flush=True)
    print(f"meta_root          : {args.meta_root}", flush=True)
    print(f"cache_dir override : {args.cache_dir}", flush=True)
    print(f"method             : {args.vectorization_method}", flush=True)
    print(f"p_active           : {args.p_active}", flush=True)
    print(f"per_trial_thresh   : {args.per_trial_thresh}", flush=True)
    print(f"clip_frames        : {args.clip_frames}", flush=True)
    print(f"max_trials         : {args.max_trials}", flush=True)
    print(f"num_workers        : {args.num_workers}", flush=True)
    print(f"progress_every     : {args.progress_every}", flush=True)
    print(f"force_recompute    : {args.force_recompute}", flush=True)
    print(f"zz_folder          : {zz_folder}", flush=True)
    print("=" * 90, flush=True)

    discovered = _discover_mice(args.data_root)
    selected_mice = args.mice if args.mice is not None else discovered
    selected_mice = [m for m in selected_mice if m in discovered]

    if not selected_mice:
        print("ERROR: No valid mice selected.", file=sys.stderr)
        return 1

    @dataclass
    class _FakeState:
        cache_dir: Optional[Path]
        data_root: Path

    fs = _FakeState(cache_dir=args.cache_dir, data_root=args.data_root)

    failures: Dict[str, str] = {}
    generated: List[str] = []
    skipped: List[str] = []

    for mouse_name in selected_mice:
        print(f"\n{'-' * 88}", flush=True)
        print(f"Mouse: {mouse_name}", flush=True)

        try:
            barcodes, labels, trial_ids, valid_frames = load_labelled_barcodes(
                args.data_root,
                args.meta_root,
                mouse_name,
                zz_folder,
                max_trials=args.max_trials,
            )
            if len(barcodes) == 0:
                print("  No labelled barcodes found; skipping mouse.", flush=True)
                skipped.append(mouse_name)
                continue

            clip_for_key = args.clip_frames
            clip_effective = (
                int(clip_for_key)
                if clip_for_key is not None
                else _infer_frames_from_metadata(valid_frames)
            )
            print(
                f"  Trials={len(barcodes)}, clip_for_key={clip_for_key}, "
                f"clip_effective={clip_effective} (metadata valid_frames)",
                flush=True,
            )

            cache_dir_mouse = _resolve_mouse_cache_dir(fs, mouse_name)
            cache_dir_mouse.mkdir(parents=True, exist_ok=True)
            cache_stem = build_vectorization_cache_stem(
                mouse_name=mouse_name,
                method=args.vectorization_method,
                p_active=args.p_active,
                per_trial_thresh=args.per_trial_thresh,
                clip_frames=clip_for_key,
            )
            cache_path = cache_dir_mouse / f"{cache_stem}.npz"

            if cache_path.exists() and not args.force_recompute:
                print(f"  SKIP cache exists: {cache_path}", flush=True)
                skipped.append(mouse_name)
                continue

            print(
                f"  [{_ts()}] Computing vectorization rows in parallel ({args.num_workers} workers) ...",
                flush=True,
            )
            t0 = time.time()
            xmat = _vectorize_trials_parallel(
                barcodes=barcodes,
                method=args.vectorization_method,
                effective_clip_frames=clip_effective,
                num_workers=args.num_workers,
                progress_every=args.progress_every,
            )

            if xmat.shape[0] != len(trial_ids):
                raise RuntimeError(
                    f"Feature/trial mismatch: rows={xmat.shape[0]} != n_trials={len(trial_ids)}"
                )

            _save_cache_npz(
                cache_path=cache_path,
                xmat=xmat,
                method=args.vectorization_method,
                clip_frames_for_key=clip_for_key,
                clip_frames_effective=clip_effective,
                mouse_name=mouse_name,
                labels=labels,
                trial_ids=trial_ids,
                valid_frames=valid_frames,
            )

            print(
                f"  [{_ts()}] DONE saved {cache_path.name} "
                f"(shape={xmat.shape}, elapsed={time.time() - t0:.1f}s)",
                flush=True,
            )
            generated.append(mouse_name)

        except Exception as exc:
            failures[mouse_name] = str(exc)
            print(f"  FAILED: {exc}", flush=True)
            traceback.print_exc()

    print(f"\n{'=' * 90}", flush=True)
    print("Summary", flush=True)
    print(f"  selected  : {len(selected_mice)}", flush=True)
    print(f"  generated : {len(generated)}", flush=True)
    print(f"  skipped   : {len(skipped)}", flush=True)
    print(f"  failed    : {len(failures)}", flush=True)
    if failures:
        print("  failure details:", flush=True)
        for mouse_name in sorted(failures.keys()):
            print(f"    - {mouse_name}: {failures[mouse_name]}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
