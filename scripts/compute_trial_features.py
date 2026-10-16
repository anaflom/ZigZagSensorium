#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compute per-frame stimulus and behaviour features for every valid trial.

For each valid trial (valid_response & valid_trial in the metadata) of each mouse,
this script reads the stimulus video and the behaviour traces from the raw Sensorium
data and stores, per frame t:

- lum             mean luminance of frame t
- contrast        RMS contrast (pixel standard deviation) of frame t
- lum_on          max(lum[t] - lum[t-1], 0)         (brightening)
- lum_off         max(lum[t-1] - lum[t], 0)         (darkening)
- contrast_change |contrast[t] - contrast[t-1]|
- flow_speed      mean optical-flow magnitude between frames t-1 and t (pixels/frame,
                  skimage.registration.optical_flow_ilk)
- structure_change      1 - mean SSIM structure term s = (cov + C3) / (sd_x sd_y + C3) between
                        frames t-1 and t (Gaussian window, local mean and contrast removed)
- local_lum_change      mean |local mean(t) - local mean(t-1)| (same Gaussian window)
- local_contrast_change mean |local SD(t) - local SD(t-1)|
- sf_low / sf_mid / sf_high              log10 power of frame t in spatial-frequency bands
                                         (cycles per frame width, edges --sf-edges)
- sf_low_change / sf_mid_change / sf_high_change   |sf_band[t] - sf_band[t-1]|
- pupil           pupil size            (behavior/<trial>.npy row 0)
- running         locomotion speed      (behavior/<trial>.npy row 1)

Change features are assigned to the frame where the change appears (t), so they are
NaN at t = 0. All features are NaN at t >= valid_frames.

Output: one compressed .npz (curves NaN-padded to a common length) plus a .json
sidecar describing the content and parameters.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import skimage
from scipy.ndimage import gaussian_filter
from skimage.registration import optical_flow_ilk

from utils import _discover_mice, _eligible_trials, _opt_csv_list, load_trial_metadata

FEATURES = [
    "lum", "contrast", "lum_on", "lum_off", "contrast_change", "flow_speed",
    "structure_change", "local_lum_change", "local_contrast_change",
    "sf_low", "sf_mid", "sf_high", "sf_low_change", "sf_mid_change", "sf_high_change",
    "pupil", "running",
]
SF_BANDS = ["sf_low", "sf_mid", "sf_high"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute per-frame video and behaviour features per trial")
    p.add_argument("--raw-root", type=Path, required=True,
                   help="Raw Sensorium data root with <mouse>/data/{videos,behavior}")
    p.add_argument("--meta-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mice", type=_opt_csv_list, default=None,
                   help="Comma-separated mouse names; default is all mice under --raw-root")
    p.add_argument("--flow-radius", type=int, default=4)
    p.add_argument("--flow-num-warp", type=int, default=10)
    p.add_argument("--local-sigma", type=float, default=1.5,
                   help="Gaussian window SD (pixels) for structure / local luminance / local contrast")
    p.add_argument("--data-range", type=float, default=255.0)
    p.add_argument("--sf-edges", type=str, default="4,12",
                   help="Band edges in cycles per frame width: low < e1 <= mid < e2 <= high")
    p.add_argument("--max-trials", type=int, default=None, help="Per mouse, for quick tests")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--output-stem", type=str, default=None)
    return p


def local_change_features(prev: np.ndarray, curr: np.ndarray, sigma: float, data_range: float):
    """Structure change, local luminance change and local contrast change between paired frames.

    prev, curr have shape (H, W, n); frames are compared along the last axis.
    """
    filt = lambda a: gaussian_filter(a, sigma=(sigma, sigma, 0), truncate=3.5)
    mu_x, mu_y = filt(prev), filt(curr)
    sd_x = np.sqrt(np.maximum(filt(prev * prev) - mu_x ** 2, 0.0))
    sd_y = np.sqrt(np.maximum(filt(curr * curr) - mu_y ** 2, 0.0))
    cov = filt(prev * curr) - mu_x * mu_y
    c3 = (0.03 * data_range) ** 2 / 2.0
    structure = (cov + c3) / (sd_x * sd_y + c3)
    return (
        1.0 - structure.mean(axis=(0, 1)),
        np.abs(mu_y - mu_x).mean(axis=(0, 1)),
        np.abs(sd_y - sd_x).mean(axis=(0, 1)),
    )


def sf_band_log_power(frames: np.ndarray, edges: Tuple[float, float]) -> np.ndarray:
    """log10 power per frame in low / mid / high spatial-frequency bands, shape (3, n).

    Frequencies in cycles per frame width (isotropic, square pixels); DC excluded;
    mean-removed, Hann-windowed frames.
    """
    h, w, _ = frames.shape
    window = np.outer(np.hanning(h), np.hanning(w))[:, :, None]
    spec = np.abs(np.fft.fft2((frames - frames.mean(axis=(0, 1))) * window, axes=(0, 1))) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    f_cpw = w * np.hypot(fx, fy)
    masks = [(f_cpw > 0) & (f_cpw < edges[0]), (f_cpw >= edges[0]) & (f_cpw < edges[1]), f_cpw >= edges[1]]
    return np.log10(np.stack([spec[m].sum(axis=0) for m in masks]) + 1e-12)


def compute_trial_features(
    raw_root: Path, mouse: str, trial: int, valid_frames: int, flow_radius: int, flow_num_warp: int,
    local_sigma: float = 1.5, data_range: float = 255.0, sf_edges: Tuple[float, float] = (4.0, 12.0),
) -> Tuple[str, int, Dict[str, np.ndarray]]:
    """Return (mouse, trial, {feature: curve}) for one trial."""
    video = np.load(raw_root / mouse / "data" / "videos" / f"{trial}.npy").astype(np.float64)
    behavior = np.load(raw_root / mouse / "data" / "behavior" / f"{trial}.npy").astype(np.float64)
    n_frames = video.shape[2]
    vf = min(int(valid_frames), n_frames)

    out = {f: np.full(n_frames, np.nan) for f in FEATURES}
    frames = video[:, :, :vf]
    lum = frames.mean(axis=(0, 1))
    contrast = frames.std(axis=(0, 1))
    out["lum"][:vf] = lum
    out["contrast"][:vf] = contrast
    d_lum = np.diff(lum)
    out["lum_on"][1:vf] = np.maximum(d_lum, 0.0)
    out["lum_off"][1:vf] = np.maximum(-d_lum, 0.0)
    out["contrast_change"][1:vf] = np.abs(np.diff(contrast))
    for t in range(1, vf):
        flow = optical_flow_ilk(frames[:, :, t - 1], frames[:, :, t], radius=flow_radius, num_warp=flow_num_warp)
        out["flow_speed"][t] = np.hypot(flow[0], flow[1]).mean()

    struct, local_lum, local_con = local_change_features(frames[:, :, :-1], frames[:, :, 1:], local_sigma, data_range)
    out["structure_change"][1:vf] = struct
    out["local_lum_change"][1:vf] = local_lum
    out["local_contrast_change"][1:vf] = local_con
    sf = sf_band_log_power(frames, sf_edges)
    for b, name in enumerate(SF_BANDS):
        out[name][:vf] = sf[b]
        out[f"{name}_change"][1:vf] = np.abs(np.diff(sf[b]))

    nb = min(vf, behavior.shape[1])
    out["pupil"][:nb] = behavior[0, :nb]
    out["running"][:nb] = behavior[1, :nb]
    return mouse, int(trial), out


def main() -> int:
    args = _build_parser().parse_args()
    sf_edges = tuple(float(e) for e in args.sf_edges.split(","))
    stem = args.output_stem or (f"trial_features-v2_flow-ilk-r{args.flow_radius}-w{args.flow_num_warp}"
                                f"_local-s{args.local_sigma:g}_sf-{args.sf_edges.replace(',', '-')}")
    out_npz = args.output_dir / f"{stem}.npz"
    out_json = args.output_dir / f"{stem}.json"

    mice = args.mice if args.mice is not None else _discover_mice(args.raw_root)
    jobs: List[Tuple[str, int, int, str, str]] = []
    for mouse in mice:
        df = _eligible_trials(load_trial_metadata(args.meta_root, mouse))
        if args.max_trials is not None:
            df = df.head(args.max_trials)
        for row in df.itertuples(index=False):
            jobs.append((mouse, int(row.trial), int(row.valid_frames), str(row.label), str(row.video_ID)))
    print(f"[{datetime.now():%H:%M:%S}] {len(jobs)} trials from {len(mice)} mice, "
          f"{args.num_workers} workers -> {out_npz}", flush=True)

    results: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(compute_trial_features, args.raw_root, m, t, vf, args.flow_radius, args.flow_num_warp,
                               args.local_sigma, args.data_range, sf_edges)
                   for m, t, vf, _, _ in jobs]
        for k, fut in enumerate(cf.as_completed(futures), start=1):
            mouse, trial, feats = fut.result()
            results[(mouse, trial)] = feats
            if k % 200 == 0 or k == len(jobs):
                print(f"  [{datetime.now():%H:%M:%S}] {k}/{len(jobs)} trials ({time.time() - t0:.0f}s)", flush=True)

    n_max = max(len(next(iter(f.values()))) for f in results.values())
    arrays = {f: np.full((len(jobs), n_max), np.nan, dtype=np.float32) for f in FEATURES}
    for i, (m, t, _, _, _) in enumerate(jobs):
        for f in FEATURES:
            curve = results[(m, t)][f]
            arrays[f][i, : len(curve)] = curve

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        mouse=np.array([j[0] for j in jobs], dtype=str),
        trial_id=np.array([j[1] for j in jobs], dtype=np.int64),
        valid_frames=np.array([j[2] for j in jobs], dtype=np.int64),
        label=np.array([j[3] for j in jobs], dtype=str),
        video_id=np.array([j[4] for j in jobs], dtype=str),
        **arrays,
    )
    meta = {
        "description": "Per-frame stimulus and behaviour features of each valid trial (Sensorium 2023).",
        "created": datetime.now().isoformat(timespec="seconds"),
        "created_by": "ZigZagSensorium/scripts/compute_trial_features.py",
        "raw_root": str(args.raw_root),
        "meta_root": str(args.meta_root),
        "n_trials": len(jobs),
        "n_mice": len(mice),
        "trials_included": "valid_response & valid_trial in meta-trials_<mouse>.csv",
        "conventions": "curves NaN-padded to n_max; change features at frame t compare frames t-1 and t "
                       "(NaN at t=0); all features NaN at t >= valid_frames",
        "arrays": {
            "lum": "mean pixel value of frame t (0-255)",
            "contrast": "pixel standard deviation of frame t (RMS contrast)",
            "lum_on": "max(lum[t] - lum[t-1], 0)",
            "lum_off": "max(lum[t-1] - lum[t], 0)",
            "contrast_change": "|contrast[t] - contrast[t-1]|",
            "flow_speed": "mean |optical flow| between frames t-1 and t, pixels/frame",
            "structure_change": "1 - mean SSIM structure term (cov + C3)/(sd_x sd_y + C3), C3 = (0.03 R)^2 / 2",
            "local_lum_change": "mean |local mean_t - local mean_(t-1)| (Gaussian window)",
            "local_contrast_change": "mean |local SD_t - local SD_(t-1)| (Gaussian window)",
            "sf_low": f"log10 power, 0 < f < {sf_edges[0]:g} cycles per frame width (DC excluded)",
            "sf_mid": f"log10 power, {sf_edges[0]:g} <= f < {sf_edges[1]:g} cycles per frame width",
            "sf_high": f"log10 power, f >= {sf_edges[1]:g} cycles per frame width",
            "sf_low_change": "|sf_low[t] - sf_low[t-1]|",
            "sf_mid_change": "|sf_mid[t] - sf_mid[t-1]|",
            "sf_high_change": "|sf_high[t] - sf_high[t-1]|",
            "pupil": "pupil size, behavior/<trial>.npy row 0",
            "running": "locomotion speed, behavior/<trial>.npy row 1",
        },
        "optical_flow": {"method": "skimage.registration.optical_flow_ilk", "radius": args.flow_radius,
                         "num_warp": args.flow_num_warp, "scikit_image": skimage.__version__},
        "local_features": {"gaussian_sigma": args.local_sigma, "truncate": 3.5, "data_range": args.data_range},
        "spatial_frequency": {"edges_cycles_per_frame_width": list(sf_edges), "window": "Hann 2D",
                              "frame_width_pixels": "64 (frames are 36 x 64)"},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(out_json, "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"Saved {out_npz} and {out_json.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
