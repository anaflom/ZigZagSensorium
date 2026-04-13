#!/usr/bin/env python3
"""Run zigzag persistence on all trials for a given mouse.

This script computes cubical zigzag persistence on 3D neural response grids
(15×15×10×n_frames). It is designed to be called from a Slurm array job,
one task per mouse.

Usage:
    python run_zigzag.py --mouse-dir <path> --p-active 30 [--n-workers 32]

Output:
    <mouse-dir>/trials_zz-thresh-<p_active>/
        zz-thresh-<p_active>_<original_trial_filename>.npy   (one per trial)
        zz-thresh-<p_active>_info.json                       (metadata)
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Flush logs immediately (important for Slurm job monitoring).
# Use PYTHONUNBUFFERED=1 or python -u to ensure real-time output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_threshold(data: np.ndarray, p_active: float) -> float:
    """Compute the activation threshold from a percentile of positive values.

    Parameters
    ----------
    data : np.ndarray
        Grid data of shape (N1, ..., Nd, n_frames).
    p_active : float
        Percentile (0-100) of positive activations.

    Returns
    -------
    float
        Threshold value (negative, matching zz-top negated convention).
    """
    positive_vals = data[data > 0].ravel()
    if len(positive_vals) == 0:
        return 0.0
    return -np.percentile(positive_vals, p_active)


def compute_threshold_from_sample(
    trial_files: list, p_active: float, n_sample: int = 20
) -> float:
    """Compute a stable threshold by averaging over a sample of trials."""
    n = min(n_sample, len(trial_files))
    indices = np.linspace(0, len(trial_files) - 1, n, dtype=int)
    thresholds = []
    for idx in indices:
        data = np.load(trial_files[idx])
        t = compute_threshold(data, p_active)
        thresholds.append(t)
    return float(np.mean(thresholds))


def process_single_trial(args):
    """Process a single trial file. Designed for multiprocessing.

    Parameters
    ----------
    args : tuple
        (trial_path, output_path, threshold, p_active, max_dim)

    Returns
    -------
    dict
        Result info: trial name, number of bars, elapsed time, success flag.
    """
    trial_path, output_path, threshold, p_active, max_dim = args
    trial_name = os.path.basename(trial_path)

    try:
        # Import inside worker to ensure clean state per process
        from zztop import run_cubical_zigzag

        t0 = time.time()
        data = np.load(trial_path)
        assert data.ndim == 4, f"Expected 4D array, got shape {data.shape}"

        if threshold is None:
            threshold = compute_threshold(data, p_active)

        bars = run_cubical_zigzag(data, threshold=threshold)
        if max_dim is not None:
            bars = [(dim, b, d) for dim, b, d in bars if dim <= max_dim]
        elapsed = time.time() - t0

        # Save as numpy array of tuples
        bars_array = np.array(bars, dtype=object)
        np.save(output_path, bars_array)

        return {
            "trial": trial_name,
            "n_bars": len(bars),
            "shape": data.shape,
            "elapsed": elapsed,
            "success": True,
            "threshold": threshold,
        }

    except Exception as e:
        return {
            "trial": trial_name,
            "error": str(e),
            "elapsed": 0.0,
            "success": False,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run zigzag persistence on all trials for a mouse."
    )
    parser.add_argument(
        "--mouse-dir",
        type=str,
        required=True,
        help="Path to the mouse directory (containing a 'trials/' subfolder).",
    )
    parser.add_argument(
        "--p-active",
        type=float,
        default=30.0,
        help="Percentile of positive activations for threshold (default: 30).",
    )
    parser.add_argument(
        "--p-active-per-trial",
        type=bool,
        default=True,
        help="Whether to compute p_active per trial (default: True).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override: use this exact threshold instead of computing from p_active.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <mouse-dir>/trials_zz-thresh-<p_active>/",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1).",
    )
    parser.add_argument(
        "--n-threshold-samples",
        type=int,
        default=20,
        help="Number of trials to sample for threshold estimation (default: 20).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials that already have output files.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Maximum homological dimension to keep (default: all). "
             "Use 2 for 3D grids to discard spurious dim-3 bars.",
    )
    args = parser.parse_args()

    mouse_dir = Path(args.mouse_dir)
    trials_dir = mouse_dir / "trials"

    if not trials_dir.is_dir():
        logger.error(f"Trials directory not found: {trials_dir}")
        sys.exit(1)

    # Find all trial files
    trial_files = sorted(trials_dir.glob("*.npy"))
    logger.info(f"Found {len(trial_files)} trial files in {trials_dir}")

    if len(trial_files) == 0:
        logger.error("No trial files found.")
        sys.exit(1)

    # Compute or use provided threshold
    p_active = args.p_active
    if args.threshold is not None:
        threshold = args.threshold
        abs_threshold = True
        logger.info(f"Using provided threshold: {threshold}")
    else:
        abs_threshold = False
        if not args.p_active_per_trial:
            logger.info(
                f"Computing threshold from p_active={p_active} "
                f"(sampling {args.n_threshold_samples} trials)..."
            )
            threshold = compute_threshold_from_sample(
                trial_files, p_active, n_sample=args.n_threshold_samples
            )
            logger.info(f"Computed threshold: {threshold:.8f}")
        else:
            logger.info(
                f"Will compute threshold per trial from p_active={p_active} "
                f"(overriding --threshold if provided)."
            )
            threshold = None  # Will compute per trial

    # Set up threshold string for filenames
    if abs_threshold:
        thresh_str = f"abs-{threshold:.8f}".rstrip("0").rstrip(".")
    else:
        thresh_str = f"{int(p_active)}" if p_active == int(p_active) else f"{p_active}"
        if args.p_active_per_trial:
            thresh_str += "-per-trial"

    # Set up output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = mouse_dir / f"trials_zz-thresh-{thresh_str}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Build task list
    tasks = []
    skipped = 0
    for trial_file in trial_files:
        out_name = f"zz-thresh-{thresh_str}_{trial_file.name}"
        out_path = output_dir / out_name

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        tasks.append((str(trial_file), str(out_path), threshold, p_active, args.max_dim))

    if skipped > 0:
        logger.info(f"Skipping {skipped} trials with existing output files.")
    logger.info(f"Processing {len(tasks)} trials with {args.n_workers} workers...")

    # Process trials
    t_start = time.time()
    results = []
    n_success = 0
    n_fail = 0

    if args.n_workers <= 1:
        # Sequential processing
        for i, task in enumerate(tasks):
            result = process_single_trial(task)
            results.append(result)
            if result["success"]:
                n_success += 1
                logger.info(
                    f"  [{i+1}/{len(tasks)}] {result['trial']}: "
                    f"{result['n_bars']} bars, shape={result['shape']}, "
                    f"{result['elapsed']:.1f}s"
                )
            else:
                n_fail += 1
                logger.error(
                    f"  [{i+1}/{len(tasks)}] {result['trial']}: FAILED — {result['error']}"
                )
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(process_single_trial, task): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results.append(result)
                if result["success"]:
                    n_success += 1
                    logger.info(
                        f"  [{n_success + n_fail}/{len(tasks)}] {result['trial']}: "
                        f"{result['n_bars']} bars, {result['elapsed']:.1f}s"
                    )
                else:
                    n_fail += 1
                    logger.error(
                        f"  [{n_success + n_fail}/{len(tasks)}] {result['trial']}: "
                        f"FAILED — {result['error']}"
                    )

    total_time = time.time() - t_start

    # Write info JSON
    info = {
        "num_files": n_success,
        "num_failed": n_fail,
        "total_trials": len(trial_files),
        "n_workers": args.n_workers,
        "total_time_seconds": total_time,
        "mouse_dir": str(mouse_dir),
        "skip_existing": args.skip_existing,
        "skipped_existing": skipped,
    }
    if abs_threshold:
        info["threshold"] = threshold
    else:
        info["p_active"] = p_active
        if not args.p_active_per_trial:
            info["threshold"] = threshold
        else:
            # create a csv file with trial-specific thresholds
            thresholds_csv = output_dir / f"zz-thresh-{thresh_str}_trial_thresholds.csv"
            with open(thresholds_csv, "w") as f:
                f.write("trial,threshold\n")
                for res in results:
                    if res["success"]:
                        f.write(f"{res['trial']},{res['threshold']}\n")
            logger.info(f"Trial-specific thresholds saved to {thresholds_csv}")
    
    info_path = output_dir / f"zz-thresh-{thresh_str}_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(
        f"\nDone: {n_success} succeeded, {n_fail} failed, "
        f"{skipped} skipped in {total_time:.1f}s total."
    )
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Info written to {info_path}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()