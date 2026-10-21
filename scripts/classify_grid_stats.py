#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classify from structure-blind statistics of grid activity.

Controls for how much the time course of the activity *level* explains,
without any spatial structure.  Uses the same problems (samples, frame
windows, folds) and models as classify_zigzag.py.  Features are precomputed
by compute_grid_stats_cache.py:

  moments_raw      per-frame [mean | CV | skewness] across grid cells
  moments_density  same, cells divided by their neuron density
  moments_std, moments_std_density
                   [mean | std | skewness]: use these on zero-mean grids
                   (by_trialmean), where the CV is undefined
  active_fraction  per-frame fraction of grid points above the zigzag threshold
                   (needs --p-active / --per-trial-thresh, or --abs-threshold)

``--time-average true`` replaces each per-frame series by its mean over the
window (one number per statistic): the trial-averaged control, which keeps
the overall level but no time course.

Output: results/grid_stats/<features>[-trialavg]/<model>/[p<P>-<...>/]<target>_<scope>/<run>/repeat-NN/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from classification_data import build_problems
from classification_engine import (
    grid_type,
    results_base,
    MODEL_TITLES,
    TrainConfig,
    add_output_args,
    add_problem_args,
    add_training_args,
    default_run_name,
    problem_config_from_args,
    run_vector_classification,
    thresh_mode,
    thresh_tag,
)
from classification_features import GRID_STATS_FEATURES, CachedFeatures
from classify_zigzag import add_repeat_args
from utils import FRAME_ALIGNED_BLOCKS, _opt_int, _str2bool


FEATURE_METHODS = GRID_STATS_FEATURES


def time_average(method: str):
    n_blocks = FRAME_ALIGNED_BLOCKS[method]

    def _avg(x: np.ndarray, _unit) -> np.ndarray:
        return x.reshape(x.shape[0], n_blocks, -1).mean(axis=2)

    return _avg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_problem_args(p)
    g = p.add_argument_group("grid statistics")
    g.add_argument("--features", required=True, choices=sorted(FEATURE_METHODS))
    g.add_argument("--time-average", default=False, type=_str2bool, help="Average each statistic over the window")
    g.add_argument("--p-active", default=None, type=_opt_int, help="Zigzag threshold (active_fraction only)")
    g.add_argument("--per-trial-thresh", default=True, type=_str2bool, help="(active_fraction only)")
    g.add_argument("--abs-threshold", default=None, type=float, help="Fixed zigzag threshold (active_fraction only)")
    g.add_argument("--cache-dir", default=None, type=Path, help="Default: <data-root>/<mouse>/cache")
    add_repeat_args(p)
    add_training_args(p)
    add_output_args(p)
    return p


def main() -> int:
    args = build_parser().parse_args()
    method = FEATURE_METHODS[args.features]
    uses_threshold = args.features == "active_fraction"
    if uses_threshold and (args.p_active is None) == (args.abs_threshold is None):
        print("--features active_fraction needs exactly one of --p-active or --abs-threshold", file=sys.stderr)
        return 2
    p_active = args.p_active if uses_threshold else None
    abs_threshold = args.abs_threshold if uses_threshold else None

    problem_cfg = problem_config_from_args(args)
    train_cfg = TrainConfig.from_args(args)
    feature_name = args.features.replace("_", "-") + ("-trialavg" if args.time_average else "")

    run_dir = results_base(args) / "grid_stats" / feature_name / args.model
    if uses_threshold:
        run_dir = run_dir / thresh_tag(p_active, args.per_trial_thresh, abs_threshold)
    run_dir = args.output_dir or (run_dir / f"{args.target}_{args.scope}" / (args.run_name or default_run_name()))
    run_dir.mkdir(parents=True, exist_ok=True)

    problems = build_problems(problem_cfg)
    if not problems.units:
        print(f"No classification problems could be built. Skips: {problems.skips}", file=sys.stderr)
        return 1

    return run_vector_classification(
        args,
        problems=problems,
        problem_cfg=problem_cfg,
        train_cfg=train_cfg,
        run_dir=run_dir,
        make_features=lambda: CachedFeatures(
            data_root=args.data_root,
            method=method,
            p_active=p_active,
            per_trial_thresh=args.per_trial_thresh,
            cache_dir=args.cache_dir,
            abs_threshold=abs_threshold,
        ),
        context={
            "grid_type": grid_type(args),
            "target": args.target,
            "scope": args.scope,
            "input": "grid_stats",
            "vectorization": feature_name,
            "model": args.model,
            "p_active": p_active,
            "thresh_mode": thresh_mode(args.per_trial_thresh, abs_threshold) if uses_threshold else "",
            "abs_threshold": abs_threshold,
        },
        title=f"{args.target} {args.scope} | grid stats {feature_name} | {MODEL_TITLES[args.model]}",
        transform=time_average(method) if args.time_average else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
