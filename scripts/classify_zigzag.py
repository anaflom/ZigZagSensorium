#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classify video labels, video IDs or segment IDs from zigzag vectorizations.

The problems (samples, classes, frame windows, folds) come from
classification_data.py and are identical to those of classify_grid.py.
Features are read from the vectorization caches (compute_vectorization_cache.py).

Output: results/zigzag/<vectorization>/<model>/p<P>-<per-trial|global>/<target>_<scope>/<run>/repeat-NN/

Examples
--------
  python classify_zigzag.py --target video_label --scope within_mouse \\
      --data-root <grid-root> --meta-root <metadata> --p-active 30 --per-trial-thresh true

  python classify_zigzag.py --target segment_id --scope cross_mouse --model logreg ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from classification_data import build_problems
from classification_engine import (
    grid_type,
    results_base,
    MODEL_TITLES,
    VECTOR_MODELS,
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
from classification_features import ZigzagFeatures
from utils import _str2bool


def add_repeat_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="logreg", choices=VECTOR_MODELS)
    p.add_argument("--n-repeats", default=1, type=int, help="Repeats with different init seeds (NN models)")
    p.add_argument("--repeat-start", default=0, type=int, help="Index of the first repeat (for array jobs)")
    p.add_argument("--init-seed", default=42, type=int, help="Base seed for model initialisation/training")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_problem_args(p)
    g = p.add_argument_group("zigzag features")
    g.add_argument("--vectorization", default="Turnover", help="Vectorization method of the caches")
    g.add_argument("--p-active", default=None, type=int)
    g.add_argument("--per-trial-thresh", default=True, type=_str2bool)
    g.add_argument("--abs-threshold", default=None, type=float, help="Fixed zigzag threshold instead of --p-active")
    g.add_argument("--cache-dir", default=None, type=Path, help="Default: <data-root>/<mouse>/cache")
    add_repeat_args(p)
    add_training_args(p)
    add_output_args(p)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if (args.p_active is None) == (args.abs_threshold is None):
        print("Give exactly one of --p-active or --abs-threshold", file=sys.stderr)
        return 2
    problem_cfg = problem_config_from_args(args)
    train_cfg = TrainConfig.from_args(args)
    tag = thresh_tag(args.p_active, args.per_trial_thresh, args.abs_threshold)

    run_dir = args.output_dir or (
        results_base(args)
        / "zigzag"
        / args.vectorization
        / args.model
        / tag
        / f"{args.target}_{args.scope}"
        / (args.run_name or default_run_name())
    )
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
        make_features=lambda: ZigzagFeatures(
            data_root=args.data_root,
            method=args.vectorization,
            p_active=args.p_active,
            per_trial_thresh=args.per_trial_thresh,
            cache_dir=args.cache_dir,
            abs_threshold=args.abs_threshold,
        ),
        context={
            "grid_type": grid_type(args),
            "target": args.target,
            "scope": args.scope,
            "input": "zigzag",
            "vectorization": args.vectorization,
            "model": args.model,
            "p_active": args.p_active,
            "thresh_mode": thresh_mode(args.per_trial_thresh, args.abs_threshold),
            "abs_threshold": args.abs_threshold,
        },
        title=f"{args.target} {args.scope} | zigzag {args.vectorization} {tag} | {MODEL_TITLES[args.model]}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
