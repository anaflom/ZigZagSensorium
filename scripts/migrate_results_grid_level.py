#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move results written before the grid-type level into results/<grid_type>/.

Old layout:  results/{zigzag,grid,grid_stats,ablation_shuffle}/...
New layout:  results/<grid_type>/{zigzag,grid,grid_stats,ablation_shuffle}/...

Folders are merged: result folders (those holding metrics.csv) are moved one by
one, so runs whose repeats were written partly in each layout end up together.
Nothing is overwritten; conflicts are reported and left in place.  Run only when
no classification job is running.  Use --dry-run first.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
TOPS = ("zigzag", "grid", "grid_stats", "ablation_shuffle")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default=PROJECT_DIR / "results", type=Path)
    p.add_argument("--grid-type", default="grid-15x15x10_norm-by_minmax")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ignore-queue", action="store_true", help="Do not check squeue for running classification jobs")
    args = p.parse_args()

    if not args.ignore_queue:
        q = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True)
        active = [j for j in q.stdout.split() if j.startswith(("zz-classify", "zz-ablation"))]
        if active:
            print(f"{len(active)} classification job(s) still queued/running ({sorted(set(active))}); wait or pass --ignore-queue")
            return 1

    root = args.results_root
    moved, conflicts = 0, []
    for top in TOPS:
        src_top = root / top
        if not src_top.exists():
            continue
        for metrics in sorted(src_top.rglob("metrics.csv")):
            src = metrics.parent
            dst = root / args.grid_type / src.relative_to(root)
            if dst.exists():
                conflicts.append(str(src))
                continue
            print(f"{'[dry-run] ' if args.dry_run else ''}{src.relative_to(root)} -> {dst.relative_to(root)}")
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            moved += 1
        # Remaining files (e.g. incomplete runs without metrics.csv) are reported, not moved.
        if not args.dry_run:
            for d in sorted((d for d in src_top.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
                if not any(d.iterdir()):
                    d.rmdir()
            if src_top.exists() and not any(src_top.iterdir()):
                src_top.rmdir()

    leftovers = [str(f) for top in TOPS if (root / top).exists() for f in (root / top).rglob("*") if f.is_file()]
    print(f"\n{'Would move' if args.dry_run else 'Moved'} {moved} result folder(s).")
    if conflicts:
        print(f"{len(conflicts)} conflict(s) (destination exists, left in place):")
        for c in conflicts:
            print(f"  {c}")
    if leftovers and not args.dry_run:
        print(f"{len(leftovers)} file(s) left in the old layout (folders without metrics.csv, e.g. failed runs):")
        for f in leftovers[:20]:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
