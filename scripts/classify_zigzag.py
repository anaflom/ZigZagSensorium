#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classify video labels, video IDs or segment IDs from zigzag vectorizations.

The problems (samples, classes, frame windows, folds) come from
classification_data.py and are identical to those of classify_grid.py.
Features are read from the vectorization caches (compute_vectorization_cache.py).

Output: results/<grid_type>/zigzag/<features>/<model>/<threshold>/<target>_<scope>/<run>/repeat-NN/
where <features> is the vectorization, with ``-H<dims>`` when only some homology
dimensions are used (--blocks) and ``+<feature>`` for each added grid statistic
(--add-features), e.g. ``Turnover-H1+active-fraction``.

Incremental information: comparing ``Turnover+active-fraction`` with the
active-fraction control alone (classify_grid_stats.py, same folds) measures what
the topology adds beyond the activity level.

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
from classification_features import GRID_STATS_FEATURES, CachedFeatures, ConcatFeatures
from utils import _opt_csv_list, _str2bool


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
    g.add_argument(
        "--blocks",
        default=None,
        type=lambda s: [int(v) for v in s.split(",")] if s else None,
        help="Keep only these frame-aligned blocks, i.e. homology dimensions for Turnover (e.g. 1 or 1,2)",
    )
    g.add_argument(
        "--add-features",
        default=None,
        type=_opt_csv_list,
        help=f"Grid statistics appended to the zigzag features ({', '.join(GRID_STATS_FEATURES)}); "
        "active_fraction uses the same zigzag threshold",
    )
    add_repeat_args(p)
    add_training_args(p)
    add_output_args(p)
    return p


def make_features(args: argparse.Namespace):
    """Zigzag vectorization (optionally some blocks only), plus appended grid statistics."""
    zigzag = CachedFeatures(
        data_root=args.data_root,
        method=args.vectorization,
        p_active=args.p_active,
        per_trial_thresh=args.per_trial_thresh,
        cache_dir=args.cache_dir,
        abs_threshold=args.abs_threshold,
        blocks=args.blocks,
    )
    if not args.add_features:
        return zigzag
    extra = []
    for name in args.add_features:
        uses_threshold = name == "active_fraction"
        extra.append(
            CachedFeatures(
                data_root=args.data_root,
                method=GRID_STATS_FEATURES[name],
                p_active=args.p_active if uses_threshold else None,
                per_trial_thresh=args.per_trial_thresh,
                cache_dir=args.cache_dir,
                abs_threshold=args.abs_threshold if uses_threshold else None,
            )
        )
    return ConcatFeatures([zigzag, *extra])


def main() -> int:
    args = build_parser().parse_args()
    if (args.p_active is None) == (args.abs_threshold is None):
        print("Give exactly one of --p-active or --abs-threshold", file=sys.stderr)
        return 2
    problem_cfg = problem_config_from_args(args)
    train_cfg = TrainConfig.from_args(args)
    tag = thresh_tag(args.p_active, args.per_trial_thresh, args.abs_threshold)
    unknown = [f for f in (args.add_features or []) if f not in GRID_STATS_FEATURES]
    if unknown:
        print(f"Unknown --add-features {unknown}; choose from {list(GRID_STATS_FEATURES)}", file=sys.stderr)
        return 2
    feature_label = args.vectorization
    if args.blocks is not None:
        feature_label += "-H" + "".join(str(b) for b in args.blocks)
    for f in args.add_features or []:
        feature_label += "+" + f.replace("_", "-")

    run_dir = args.output_dir or (
        results_base(args)
        / "zigzag"
        / feature_label
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
        make_features=lambda: make_features(args),
        context={
            "grid_type": grid_type(args),
            "target": args.target,
            "scope": args.scope,
            "input": "zigzag",
            "vectorization": feature_label,
            "model": args.model,
            "p_active": args.p_active,
            "thresh_mode": thresh_mode(args.per_trial_thresh, args.abs_threshold),
            "abs_threshold": args.abs_threshold,
            "blocks": "" if args.blocks is None else "|".join(str(b) for b in args.blocks),
            "added_features": "|".join(args.add_features or []),
        },
        title=f"{args.target} {args.scope} | zigzag {feature_label} {tag} | {MODEL_TITLES[args.model]}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
