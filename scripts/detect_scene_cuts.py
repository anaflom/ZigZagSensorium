#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detect scene cuts in the NaturalVideo clips and compare them with the metadata cuts.

For every unique NaturalVideo video ID, one presentation (trial) is read from the raw
Sensorium data and three sets of cuts are produced (cut = first frame of the new shot,
the convention of the metadata `frame_start`):

- metadata   : segment starts > 0 in global_meta/videos/NaturalVideo-<ID>.json
               (derived from frame-to-frame MSE when the metadata were built)
- adaptive   : PySceneDetect AdaptiveDetector (luma only)
- content    : PySceneDetect ContentDetector (luma only)
- transnetv2 : TransNetV2 (PyTorch port), frames resized to 48 x 27 and replicated to 3 channels;
               the per-frame transition probability is also stored

Outputs (in --output-dir):
- <stem>.npz   : video_id, valid_frames, cuts per method (object arrays of int arrays),
                 TransNetV2 per-frame probabilities (NaN-padded)
- <stem>.json  : parameters and conventions
- <stem>_per_video.csv : number of cuts per method and pairwise matches per video
- <stem>_agreement.csv : pairwise precision / recall / F1 across all videos (tolerance in frames)
- review/      : contact sheets around candidate cuts for a random sample of videos with disagreements
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch

from utils import _coerce_segment_records, _discover_mice, _to_int_or_none, load_video_metadata_json

METHODS = ["metadata", "adaptive", "content", "transnetv2"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scene-cut detection for NaturalVideo clips")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--meta-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--adaptive-threshold", type=float, default=3.0)
    p.add_argument("--content-threshold", type=float, default=27.0)
    p.add_argument("--min-scene-len", type=int, default=15, help="Frames (PySceneDetect)")
    p.add_argument("--transnet-threshold", type=float, default=0.5)
    p.add_argument("--tolerance", type=int, default=2, help="Frames for matching cuts between methods")
    p.add_argument("--max-videos", type=int, default=None, help="For quick tests")
    p.add_argument("--review-sample", type=int, default=60, help="Videos with disagreements to render")
    p.add_argument("--num-workers", type=int, default=1, help="Parallel processes over clips")
    p.add_argument("--num-threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--seed", type=int, default=0)
    return p


def _video_index(raw_root: Path, meta_root: Path) -> pd.DataFrame:
    """One row per unique NaturalVideo ID: a (mouse, trial) presentation with a valid video."""
    rows = []
    for mouse in _discover_mice(raw_root):
        df = pd.read_csv(meta_root / mouse / "trials" / f"meta-trials_{mouse}.csv")
        df = df[(df["label"] == "NaturalVideo") & df["valid_video"].astype(bool)]
        for r in df.itertuples(index=False):
            rows.append({"video_id": str(r.video_ID), "mouse": mouse, "trial": int(r.trial),
                         "valid_frames": int(r.valid_frames)})
    return pd.DataFrame(rows).drop_duplicates("video_id").sort_values("video_id").reset_index(drop=True)


def metadata_cuts(meta_root: Path, video_id: str) -> np.ndarray:
    payload, _, _ = load_video_metadata_json(meta_root, video_id, label="NaturalVideo")
    if payload is None:
        return np.array([], dtype=int)
    starts = [_to_int_or_none(s.get("frame_start")) for s in _coerce_segment_records(payload.get("segments"))]
    return np.array(sorted(s for s in starts if s is not None and s > 0), dtype=int)


def pyscenedetect_cuts(frames_u8: np.ndarray, detector, fps: float) -> np.ndarray:
    """Run a PySceneDetect detector on (n, H, W) uint8 frames; return cut frame numbers."""
    from scenedetect.common import FrameTimecode

    cuts: List[int] = []
    for i in range(frames_u8.shape[0]):
        bgr = cv2.cvtColor(frames_u8[i], cv2.COLOR_GRAY2BGR)
        cuts += [tc.frame_num for tc in detector.process_frame(FrameTimecode(i, fps=fps), bgr)]
    cuts += [tc.frame_num for tc in detector.post_process(FrameTimecode(frames_u8.shape[0] - 1, fps=fps))]
    return np.array(sorted(c for c in set(cuts) if c > 0), dtype=int)


def transnet_cuts(model, frames_u8: np.ndarray, threshold: float):
    """TransNetV2 on (n, H, W) uint8 frames; return (cut frames, per-frame transition probability)."""
    small = np.stack([cv2.resize(f, (48, 27), interpolation=cv2.INTER_AREA) for f in frames_u8])
    rgb = torch.from_numpy(np.repeat(small[..., None], 3, axis=-1).copy())
    single, _ = model.predict_frames(rgb, quiet=True)
    prob = single.numpy().astype(np.float64)
    scenes = model.predictions_to_scenes(prob, threshold=threshold)
    return np.array(sorted(int(s) for s in scenes[1:, 0] if s > 0), dtype=int), prob


_MODEL = None


def _init_worker(num_threads: int):
    """Load one TransNetV2 per worker process."""
    global _MODEL
    torch.set_num_threads(num_threads)
    from transnetv2_pytorch import TransNetV2

    _MODEL = TransNetV2(device="cpu")


def process_clip(raw_root: Path, meta_root: Path, mouse: str, trial: int, valid_frames: int, video_id: str,
                 cfg: dict) -> dict:
    """All methods for one clip."""
    from scenedetect.detectors import AdaptiveDetector, ContentDetector

    video = np.load(raw_root / mouse / "data" / "videos" / f"{trial}.npy")[:, :, :valid_frames]
    frames = np.clip(np.moveaxis(video, 2, 0), 0, 255).astype(np.uint8)      # (n, H, W)
    out = {"metadata": metadata_cuts(meta_root, video_id)}
    out["adaptive"] = pyscenedetect_cuts(
        frames, AdaptiveDetector(adaptive_threshold=cfg["adaptive_threshold"], min_scene_len=cfg["min_scene_len"],
                                 luma_only=True), cfg["fps"])
    out["content"] = pyscenedetect_cuts(
        frames, ContentDetector(threshold=cfg["content_threshold"], min_scene_len=cfg["min_scene_len"],
                                luma_only=True), cfg["fps"])
    out["transnetv2"], out["prob"] = transnet_cuts(_MODEL, frames, cfg["transnet_threshold"])
    return out


def match_count(ref: np.ndarray, test: np.ndarray, tol: int) -> int:
    """Greedy one-to-one matching of cuts within +-tol frames."""
    used = np.zeros(len(test), dtype=bool)
    n = 0
    for c in ref:
        d = np.abs(test - c).astype(float)
        d[used] = np.inf
        if d.size and d.min() <= tol:
            used[np.argmin(d)] = True
            n += 1
    return n


def render_review_sheet(frames: np.ndarray, cuts: Dict[str, np.ndarray], video_id: str, tol: int, out_path: Path):
    """One row per candidate cut (union over methods): frames t-2 .. t+1 and which methods flagged it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates: List[int] = []
    for c in sorted(np.concatenate([v for v in cuts.values()]).tolist()):
        if not candidates or c - candidates[-1] > tol:
            candidates.append(int(c))
    if not candidates:
        return
    offsets = [-2, -1, 0, 1]
    fig, axes = plt.subplots(len(candidates), len(offsets), figsize=(1.9 * len(offsets) + 2.6, 1.15 * len(candidates) + 0.6),
                             squeeze=False)
    fig.suptitle(f"NaturalVideo {video_id}: candidate cuts (frame t = first frame of the new shot)", fontsize=9)
    for r, c in enumerate(candidates):
        flagged = [m for m, v in cuts.items() if np.any(np.abs(v - c) <= tol)]
        for k, off in enumerate(offsets):
            ax = axes[r, k]
            t = c + off
            if 0 <= t < frames.shape[0]:
                ax.imshow(frames[t], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t{off:+d}" if off else "t", fontsize=8)
        axes[r, 0].set_ylabel(f"t={c}", fontsize=8, rotation=0, ha="right", va="center")
        axes[r, -1].text(1.05, 0.5, ", ".join(flagged), transform=axes[r, -1].transAxes, fontsize=7, va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    args = _build_parser().parse_args()

    stem = f"scene_cuts_naturalvideo_tol{args.tolerance}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.output_dir / "review"

    index = _video_index(args.raw_root, args.meta_root)
    if args.max_videos is not None:
        index = index.head(args.max_videos)
    print(f"[{datetime.now():%H:%M:%S}] {len(index)} NaturalVideo clips", flush=True)

    cfg = {"fps": args.fps, "adaptive_threshold": args.adaptive_threshold, "content_threshold": args.content_threshold,
           "min_scene_len": args.min_scene_len, "transnet_threshold": args.transnet_threshold}
    per_clip: List[dict] = [None] * len(index)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker,
                                initargs=(args.num_threads,)) as pool:
        futures = {pool.submit(process_clip, args.raw_root, args.meta_root, r.mouse, r.trial, r.valid_frames,
                               r.video_id, cfg): i for i, r in enumerate(index.itertuples(index=False))}
        for k, fut in enumerate(cf.as_completed(futures), start=1):
            per_clip[futures[fut]] = fut.result()
            if k % 100 == 0 or k == len(index):
                print(f"  [{datetime.now():%H:%M:%S}] {k}/{len(index)} clips ({time.time() - t0:.0f}s)", flush=True)
    results: Dict[str, List[np.ndarray]] = {m: [c[m] for c in per_clip] for m in METHODS}
    probs: List[np.ndarray] = [c["prob"] for c in per_clip]

    # -- Agreement ------------------------------------------------------------------------
    per_video = index[["video_id", "mouse", "trial", "valid_frames"]].copy()
    for m in METHODS:
        per_video[f"n_{m}"] = [len(c) for c in results[m]]
    agree_rows = []
    for a, b in combinations(METHODS, 2):
        matched = [match_count(ca, cb, args.tolerance) for ca, cb in zip(results[a], results[b])]
        per_video[f"match_{a}_{b}"] = matched
        n_a, n_b, n_m = per_video[f"n_{a}"].sum(), per_video[f"n_{b}"].sum(), int(np.sum(matched))
        # precision / recall of b taking a as the reference
        prec = n_m / n_b if n_b else np.nan
        rec = n_m / n_a if n_a else np.nan
        agree_rows.append({"reference": a, "test": b, "n_reference": int(n_a), "n_test": int(n_b), "n_matched": n_m,
                           "precision": prec, "recall": rec,
                           "f1": 2 * prec * rec / (prec + rec) if prec and rec else np.nan,
                           "videos_identical": float(np.mean([(len(x) == len(y) == m) for x, y, m in
                                                              zip(results[a], results[b], matched)]))})
    agreement = pd.DataFrame(agree_rows)
    per_video.to_csv(args.output_dir / f"{stem}_per_video.csv", index=False)
    agreement.to_csv(args.output_dir / f"{stem}_agreement.csv", index=False)
    print(agreement.round(3).to_string(index=False), flush=True)

    # -- Save cuts --------------------------------------------------------------------------
    n_max = int(index["valid_frames"].max())
    prob_arr = np.full((len(index), n_max), np.nan, dtype=np.float32)
    for i, p in enumerate(probs):
        prob_arr[i, : len(p)] = p
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        video_id=index["video_id"].to_numpy(dtype=str),
        mouse=index["mouse"].to_numpy(dtype=str),
        trial=index["trial"].to_numpy(dtype=np.int64),
        valid_frames=index["valid_frames"].to_numpy(dtype=np.int64),
        transnetv2_prob=prob_arr,
        **{f"cuts_{m}": np.array(results[m], dtype=object) for m in METHODS},
    )
    meta = {
        "description": "Scene cuts of the NaturalVideo clips (one presentation per video ID).",
        "created": datetime.now().isoformat(timespec="seconds"),
        "created_by": "ZigZagSensorium/scripts/detect_scene_cuts.py",
        "convention": "cut = first frame of the new shot (as metadata frame_start); frame 0 excluded",
        "n_videos": len(index),
        "methods": {
            "metadata": "segment starts in global_meta/videos (derived from frame-to-frame MSE)",
            "adaptive": {"detector": "scenedetect.AdaptiveDetector", "adaptive_threshold": args.adaptive_threshold,
                         "min_scene_len": args.min_scene_len, "luma_only": True},
            "content": {"detector": "scenedetect.ContentDetector", "threshold": args.content_threshold,
                        "min_scene_len": args.min_scene_len, "luma_only": True},
            "transnetv2": {"package": "transnetv2-pytorch 1.0.5", "input": "48x27, grey replicated to 3 channels",
                           "threshold": args.transnet_threshold,
                           "cut": "first frame after a run of frames with transition probability > threshold"},
        },
        "agreement_tolerance_frames": args.tolerance,
        "npz_note": "cuts_<method> are object arrays: load with np.load(..., allow_pickle=True)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(args.output_dir / f"{stem}.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    # -- Review sheets for a random sample of videos with disagreements -------------------
    disagree = [i for i in range(len(index))
                if len({len(results[m][i]) for m in METHODS}) > 1
                or any(match_count(results["metadata"][i], results[m][i], args.tolerance) != len(results["metadata"][i])
                       for m in METHODS[1:])]
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(disagree, size=min(args.review_sample, len(disagree)), replace=False) if disagree else []
    review_dir.mkdir(parents=True, exist_ok=True)
    for i in sorted(sample):
        r = index.iloc[i]
        video = np.load(args.raw_root / r.mouse / "data" / "videos" / f"{r.trial}.npy")[:, :, : r.valid_frames]
        frames = np.clip(np.moveaxis(video, 2, 0), 0, 255).astype(np.uint8)
        render_review_sheet(frames, {m: results[m][i] for m in METHODS}, r.video_id, args.tolerance,
                            review_dir / f"review_{r.video_id}.png")
    print(f"{len(disagree)} / {len(index)} clips with disagreements; {len(sample)} review sheets in {review_dir}",
          flush=True)
    print(f"Saved {stem}.npz / .json / _per_video.csv / _agreement.csv in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
