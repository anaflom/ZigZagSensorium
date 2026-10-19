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
import traceback
from pathlib import Path

from classification_data import build_problems
from classification_engine import (
    MODEL_TITLES,
    VECTOR_MODELS,
    Tee,
    TrainConfig,
    add_output_args,
    add_problem_args,
    add_training_args,
    base_config,
    configure_torch,
    default_run_name,
    fit_predict_vector,
    problem_config_from_args,
    run_units,
    thresh_tag,
    write_outputs,
)
from classification_features import ZigzagFeatures
from utils import _str2bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_problem_args(p)
    g = p.add_argument_group("zigzag features")
    g.add_argument("--vectorization", default="Turnover", help="Vectorization method of the caches")
    g.add_argument("--p-active", required=True, type=int)
    g.add_argument("--per-trial-thresh", required=True, type=_str2bool)
    g.add_argument("--cache-dir", default=None, type=Path, help="Default: <data-root>/<mouse>/cache")
    p.add_argument("--model", default="logreg", choices=VECTOR_MODELS)
    p.add_argument("--n-repeats", default=1, type=int, help="Repeats with different init seeds (NN models)")
    p.add_argument("--repeat-start", default=0, type=int, help="Index of the first repeat (for array jobs)")
    p.add_argument("--init-seed", default=42, type=int, help="Base seed for model initialisation/training")
    add_training_args(p)
    add_output_args(p)
    return p


def main() -> int:
    args = build_parser().parse_args()
    problem_cfg = problem_config_from_args(args)
    train_cfg = TrainConfig.from_args(args)

    run_dir = args.output_dir or (
        args.results_root
        / "zigzag"
        / args.vectorization
        / args.model
        / thresh_tag(args.p_active, args.per_trial_thresh)
        / f"{args.target}_{args.scope}"
        / (args.run_name or default_run_name())
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    problems = build_problems(problem_cfg)
    if not problems.units:
        print(f"No classification problems could be built. Skips: {problems.skips}", file=sys.stderr)
        return 1
    device = configure_torch(train_cfg) if args.model != "logreg" else None

    for repeat in range(args.repeat_start, args.repeat_start + args.n_repeats):
        out_dir = run_dir / f"repeat-{repeat:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        init_seed = args.init_seed + repeat
        with Tee(out_dir / "run.log"):
            try:
                print(f"classify_zigzag: {args.target} / {args.scope} / {args.vectorization} / {args.model}")
                print(f"output: {out_dir}")
                print(f"problems: {len(problems.units)} (skipped {len(problems.skips)}), notes={problems.notes}")
                features = ZigzagFeatures(
                    data_root=args.data_root,
                    method=args.vectorization,
                    p_active=args.p_active,
                    per_trial_thresh=args.per_trial_thresh,
                    cache_dir=args.cache_dir,
                )

                def fit_fn_for_unit(unit):
                    x = features.features(unit.samples)
                    return lambda tr, te, seed: fit_predict_vector(
                        args.model,
                        x,
                        unit.y,
                        tr,
                        te,
                        n_classes=unit.n_classes,
                        window_frames=unit.window_frames,
                        seed=seed,
                        train_cfg=train_cfg,
                        device=device,
                    )

                summaries, folds, preds = run_units(problems.units, fit_fn_for_unit, init_seed=init_seed)

                context = {
                    "target": args.target,
                    "scope": args.scope,
                    "input": "zigzag",
                    "vectorization": args.vectorization,
                    "model": args.model,
                    "p_active": args.p_active,
                    "thresh_mode": "per-trial" if args.per_trial_thresh else "global",
                    "repeat": repeat,
                    "init_seed": init_seed,
                    "split_seed": args.split_seed,
                }
                config = base_config(args, problem_cfg, train_cfg if args.model != "logreg" else None)
                config.update(
                    {
                        "context": context,
                        "notes": problems.notes,
                        "mice_used": problems.mice_used,
                        "feature_caches": features.sources,
                    }
                )
                write_outputs(
                    out_dir,
                    context=context,
                    config=config,
                    summaries=summaries,
                    fold_rows=folds,
                    predictions=preds,
                    skips=problems.skips,
                    title=(
                        f"{args.target} {args.scope} | zigzag {args.vectorization} "
                        f"{thresh_tag(args.p_active, args.per_trial_thresh)} | {MODEL_TITLES[args.model]}"
                    ),
                )
                print(f"\nDone: {out_dir}")
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
