#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classify video labels, video IDs or segment IDs from grid activity with a 3D-CNN.

The problems (samples, classes, frame windows, folds) come from
classification_data.py and are identical to those of classify_zigzag.py.
Nothing here depends on the zigzag threshold (P_ACTIVE).

Repeats re-train the network with a different initialisation seed while
keeping the CV folds fixed (``--split-seed``), which measures the
run-to-run variability of the CNN.  Run repeats in one job with
``--n-repeats N`` or one per array task with ``--repeat-start $SLURM_ARRAY_TASK_ID
--n-repeats 1 --run-name <shared name>``.

Output: results/grid/<model>/<target>_<scope>/<run>/repeat-NN/
"""

from __future__ import annotations

import argparse
import sys
import traceback

from classification_data import build_problems
from classification_engine import (
    grid_type,
    results_base,
    GRID_MODELS,
    MODEL_TITLES,
    Tee,
    TrainConfig,
    add_output_args,
    add_problem_args,
    add_training_args,
    base_config,
    configure_torch,
    default_run_name,
    fit_predict_grid,
    problem_config_from_args,
    run_units,
    write_outputs,
)
from classification_features import grid_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_problem_args(p)
    p.add_argument("--model", default="cnn3d_raw", choices=GRID_MODELS)
    p.add_argument("--n-repeats", default=1, type=int)
    p.add_argument("--repeat-start", default=0, type=int, help="Index of the first repeat (for array jobs)")
    p.add_argument("--init-seed", default=42, type=int, help="Base seed; repeat r uses init_seed + r")
    p.add_argument(
        "--part",
        default="",
        help="Tag for jobs that each run a subset of mice into the same run folder "
        "(results go to repeat-NN_<part>), e.g. one job per mouse",
    )
    add_training_args(p)
    add_output_args(p)
    return p


def main() -> int:
    args = build_parser().parse_args()
    problem_cfg = problem_config_from_args(args)
    train_cfg = TrainConfig.from_args(args)

    run_dir = args.output_dir or (
        results_base(args) / "grid" / args.model / f"{args.target}_{args.scope}" / (args.run_name or default_run_name())
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    problems = build_problems(problem_cfg)
    if not problems.units:
        print(f"No classification problems could be built. Skips: {problems.skips}", file=sys.stderr)
        return 1
    device = configure_torch(train_cfg)
    normalize = args.model == "cnn3d_norm"

    for repeat in range(args.repeat_start, args.repeat_start + args.n_repeats):
        out_dir = run_dir / (f"repeat-{repeat:02d}" + (f"_{args.part}" if args.part else ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        init_seed = args.init_seed + repeat
        with Tee(out_dir / "run.log"):
            try:
                print(f"classify_grid: {args.target} / {args.scope} / {args.model} / repeat {repeat} (init_seed={init_seed})")
                print(f"output: {out_dir}, device: {device}, deterministic: {train_cfg.deterministic}")
                print(f"problems: {len(problems.units)} (skipped {len(problems.skips)}), notes={problems.notes}")

                def fit_fn_for_unit(unit):
                    ds = grid_dataset(unit.samples, unit.y, normalize=normalize)
                    return lambda tr, te, seed: fit_predict_grid(
                        args.model,
                        ds,
                        unit.y,
                        tr,
                        te,
                        n_classes=unit.n_classes,
                        seed=seed,
                        train_cfg=train_cfg,
                        device=device,
                    )

                summaries, folds, preds = run_units(problems.units, fit_fn_for_unit, init_seed=init_seed)

                context = {
                    "grid_type": grid_type(args),
                    "target": args.target,
                    "scope": args.scope,
                    "input": "grid",
                    "vectorization": "",
                    "model": args.model,
                    "p_active": None,
                    "thresh_mode": "",
                    "repeat": repeat,
                    "part": args.part,
                    "init_seed": init_seed,
                    "split_seed": args.split_seed,
                }
                config = base_config(args, problem_cfg, train_cfg)
                config.update(
                    {
                        "context": context,
                        "notes": problems.notes,
                        "mice_used": problems.mice_used,
                        "device": str(device),
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
                    title=f"{args.target} {args.scope} | grid | {MODEL_TITLES[args.model]} | repeat {repeat}",
                )
                print(f"\nDone: {out_dir}")
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
