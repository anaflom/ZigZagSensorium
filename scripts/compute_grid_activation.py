#!/usr/bin/env python3
"""Compute 3D grid activations for Sensorium responses.

This script is standalone with respect to Sensorium grid helpers:
- grid range construction is implemented locally,
- grid assignment loops are implemented locally,
- Fortran kernel ``pcs_assign_3d`` is compiled/loaded from assign_grid.f90.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import (
    load_trial_metadata,
    _discover_mice,
    _to_bool_series,
    _eligible_trials,
    _extract_trial_id_from_name,
    _resolve_trial_response_file,
)


@dataclass
class RunState:
    data_root: Path
    meta_root: Path
    root_output: Path
    num_grid: Tuple[int, int, int]
    normalization: Optional[str]
    mice: Optional[List[str]]
    n_workers: int
    skip_existing: bool
    compare_reference_root: Optional[Path]
    fortran_source: Path


def _default_fortran_source() -> Path:
    """Return default assign_grid.f90 next to this script."""
    return Path(__file__).resolve().parent / "assign_grid.f90"


def _opt_csv_list(value: str) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items if items else None


def _normalization_token(normalization: Optional[str]) -> str:
    if normalization is None:
        return "no-normalization"
    if normalization == "by_minmax":
        return "by_minmax"
    raise ValueError("Normalization can only be None or 'by_minmax'.")


def load_neuron_metadata(meta_root: Path, mouse_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = meta_root / mouse_name / "neurons" / f"meta-neurons_{mouse_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Neuron metadata not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = [
        "coord_x",
        "coord_y",
        "coord_z",
        "min_activation",
        "max_activation",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required neuron metadata columns in {csv_path}: {missing}")

    positions = df[["coord_x", "coord_y", "coord_z"]].to_numpy(dtype=np.float64)
    min_act = df["min_activation"].to_numpy(dtype=np.float64)
    max_act = df["max_activation"].to_numpy(dtype=np.float64)
    return positions, min_act, max_act


def get_ranges_from_positions(positions: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    x_range = (float(positions[:, 0].min()), float(positions[:, 0].max()))
    y_range = (float(positions[:, 1].min()), float(positions[:, 1].max()))
    z_pos = np.unique(positions[:, 2])
    layers_distance = np.unique(np.diff(z_pos))
    if len(layers_distance) != 1:
        raise ValueError("Z positions are not evenly spaced.")
    z_step = float(layers_distance[0])
    z_range = (float(positions[:, 2].min() - z_step / 2), float(positions[:, 2].max() + z_step / 2))
    return (x_range, y_range, z_range)


def create_grid_3d(
    xyz_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    num_grid: Tuple[int, int, int],
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x_range = xyz_ranges[0]
    y_range = xyz_ranges[1]
    z_range = xyz_ranges[2]

    x_step = (x_range[1] - x_range[0]) / num_grid[0]
    y_step = (y_range[1] - y_range[0]) / num_grid[1]
    z_step = (z_range[1] - z_range[0]) / num_grid[2]

    x_lin = np.linspace(x_range[0], x_range[1], num_grid[0] + 1)
    y_lin = np.linspace(y_range[0], y_range[1], num_grid[1] + 1)
    z_lin = np.linspace(z_range[0], z_range[1], num_grid[2] + 1)

    x_centers = np.asarray([x + x_step / 2 for x in x_lin[:-1]], dtype=np.float64)
    y_centers = np.asarray([y + y_step / 2 for y in y_lin[:-1]], dtype=np.float64)
    z_centers = np.asarray([z + z_step / 2 for z in z_lin[:-1]], dtype=np.float64)

    xg, yg, zg = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    return (xg, yg, zg), (x_lin, y_lin, z_lin)


def normalize_responses(
    data: np.ndarray,
    normalization: Optional[str],
    min_activation: np.ndarray,
    max_activation: np.ndarray,
) -> Tuple[np.ndarray, int]:
    if normalization is None:
        return data, 0

    if normalization != "by_minmax":
        raise ValueError("Normalization can only be None or 'by_minmax'.")

    denom = max_activation - min_activation
    zero_denom = int(np.sum(denom == 0))
    # Keep expression parity with Responses.get_data(by_minmax).
    data_norm = (data - min_activation[:, None]) / denom[:, None]
    return data_norm, zero_denom


def compute_grid_activity_3d(
    positions: np.ndarray,
    activities: np.ndarray,
    num_grid: Tuple[int, int, int],
    xyz_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    assign_module,
) -> np.ndarray:
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
    if activities.ndim != 2:
        raise ValueError(f"activities must have shape (N, T), got {activities.shape}")
    if activities.shape[0] != positions.shape[0]:
        raise ValueError(
            f"Neuron count mismatch: positions={positions.shape[0]}, activities={activities.shape[0]}"
        )

    nx, ny, nz = (int(num_grid[0]), int(num_grid[1]), int(num_grid[2]))
    n_frames = int(activities.shape[1])
    grid_activity = np.zeros((nx, ny, nz, n_frames), dtype=np.float64)

    xp = np.asarray(positions[:, 0], dtype=np.float64)
    yp = np.asarray(positions[:, 1], dtype=np.float64)
    zp = np.asarray(positions[:, 2], dtype=np.float64)

    for t in range(n_frames):
        values = np.asarray(activities[:, t], dtype=np.float64)
        grid_t = assign_module.pcs_assign_3d(
            xp,
            yp,
            zp,
            values,
            len(positions),
            nx,
            ny,
            nz,
            xyz_ranges[0],
            xyz_ranges[1],
            xyz_ranges[2],
        )
        grid_activity[:, :, :, t] = np.transpose(grid_t, (2, 1, 0))

    return grid_activity


def compile_and_load_assign_module(
    fortran_source: Path,
    build_dir: Path,
    module_name: str = "assign_grid_zigzag",
):
    if not fortran_source.exists():
        raise FileNotFoundError(f"Fortran source not found: {fortran_source}")

    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        "-m",
        module_name,
        str(fortran_source),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (
            "Failed to compile assign_grid.f90 with f2py.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        raise RuntimeError(msg)

    candidates = sorted(build_dir.glob(f"{module_name}*.so")) + sorted(build_dir.glob(f"{module_name}*.pyd"))
    if not candidates:
        raise RuntimeError(
            f"Compilation succeeded but no extension module matching {module_name}*.so was found in {build_dir}."
        )

    module_path = candidates[-1]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "pcs_assign_3d"):
        raise RuntimeError(f"Compiled module {module_path} does not expose pcs_assign_3d")

    return module, cmd, module_path


def _run_single_trial(
    mouse_name: str,
    trial_id: int,
    valid_frames: int,
    responses_dir: Path,
    file_prefix: str,
    output_trials_dir: Path,
    positions: np.ndarray,
    min_activation: np.ndarray,
    max_activation: np.ndarray,
    normalization: Optional[str],
    num_grid: Tuple[int, int, int],
    xyz_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    assign_module,
    skip_existing: bool,
) -> Dict[str, object]:
    src_file = _resolve_trial_response_file(responses_dir, trial_id)
    if src_file is None:
        return {"status": "missing", "trial": int(trial_id), "message": "response file not found"}

    out_file = output_trials_dir / f"{file_prefix}rec-{mouse_name}_trial-{int(trial_id)}.npy"
    if skip_existing and out_file.exists():
        return {"status": "skipped_existing", "trial": int(trial_id), "file": str(out_file)}

    data = np.load(src_file)
    if data.ndim != 2:
        return {
            "status": "failed",
            "trial": int(trial_id),
            "message": f"response must be 2D (n_neurons, n_frames), got {data.shape}",
        }

    if data.shape[0] != positions.shape[0]:
        return {
            "status": "failed",
            "trial": int(trial_id),
            "message": f"n_neurons mismatch response={data.shape[0]} positions={positions.shape[0]}",
        }

    vf = int(valid_frames)
    if vf <= 0 or vf > data.shape[1]:
        return {
            "status": "failed",
            "trial": int(trial_id),
            "message": f"invalid valid_frames={vf} for response length={data.shape[1]}",
        }

    trial_data = np.asarray(data[:, :vf], dtype=np.float64)
    if np.isnan(trial_data).any():
        return {"status": "failed", "trial": int(trial_id), "message": "NaN values found in responses"}

    trial_data, zero_denom = normalize_responses(
        trial_data,
        normalization=normalization,
        min_activation=min_activation,
        max_activation=max_activation,
    )

    grid_activity = compute_grid_activity_3d(
        positions=positions,
        activities=trial_data,
        num_grid=num_grid,
        xyz_ranges=xyz_ranges,
        assign_module=assign_module,
    )
    np.save(out_file, grid_activity)

    return {
        "status": "processed",
        "trial": int(trial_id),
        "file": str(out_file),
        "source": str(src_file),
        "shape": tuple(int(x) for x in grid_activity.shape),
        "zero_denom_neurons": int(zero_denom),
    }


def compare_with_reference(
    output_root: Path,
    reference_root: Path,
    mouse_name: str,
) -> Dict[str, object]:
    out_mouse = output_root / mouse_name
    ref_mouse = reference_root / mouse_name

    out_trials = out_mouse / "trials_grid"
    ref_trials = ref_mouse / "trials"
    out_grid = out_mouse / "grid"
    ref_grid = ref_mouse / "grid"

    if not out_trials.exists() or not ref_trials.exists():
        return {
            "ok": False,
            "message": f"Missing trial directories output={out_trials.exists()} reference={ref_trials.exists()}",
        }

    out_map: Dict[int, Path] = {}
    for f in sorted(out_trials.glob("*.npy")):
        tid = _extract_trial_id_from_name(f)
        if tid is not None:
            out_map[tid] = f

    ref_map: Dict[int, Path] = {}
    for f in sorted(ref_trials.glob("*.npy")):
        tid = _extract_trial_id_from_name(f)
        if tid is not None:
            ref_map[tid] = f

    common = sorted(set(out_map.keys()) & set(ref_map.keys()))
    out_only = sorted(set(out_map.keys()) - set(ref_map.keys()))
    ref_only = sorted(set(ref_map.keys()) - set(out_map.keys()))

    mismatches: List[Dict[str, object]] = []
    for tid in common:
        out_arr = np.load(out_map[tid])
        ref_arr = np.load(ref_map[tid])
        if out_arr.shape != ref_arr.shape:
            mismatches.append(
                {
                    "trial": tid,
                    "reason": "shape",
                    "out_shape": tuple(int(x) for x in out_arr.shape),
                    "ref_shape": tuple(int(x) for x in ref_arr.shape),
                }
            )
            continue
        if not np.allclose(out_arr, ref_arr, rtol=0.0, atol=0.0, equal_nan=True):
            max_abs_diff = float(np.nanmax(np.abs(out_arr - ref_arr)))
            mismatches.append({"trial": tid, "reason": "values", "max_abs_diff": max_abs_diff})

    grid_match = {"num_grid": None, "xyz_ranges": None}
    num_grid_out = out_grid / "num_grid.npy"
    num_grid_ref = ref_grid / "num_grid.npy"
    xyz_out = out_grid / "xyz_ranges.npy"
    xyz_ref = ref_grid / "xyz_ranges.npy"
    if num_grid_out.exists() and num_grid_ref.exists():
        grid_match["num_grid"] = bool(np.array_equal(np.load(num_grid_out), np.load(num_grid_ref)))
    if xyz_out.exists() and xyz_ref.exists():
        grid_match["xyz_ranges"] = bool(np.array_equal(np.load(xyz_out), np.load(xyz_ref)))

    ok = len(mismatches) == 0 and len(out_only) == 0 and len(ref_only) == 0
    return {
        "ok": ok,
        "n_common": len(common),
        "n_out_only": len(out_only),
        "n_ref_only": len(ref_only),
        "out_only_trials": out_only,
        "ref_only_trials": ref_only,
        "mismatches": mismatches,
        "grid_match": grid_match,
    }


def run_pipeline(state: RunState) -> Dict[str, object]:
    state.root_output.mkdir(parents=True, exist_ok=True)
    build_dir = state.root_output / "_build_assign_grid"

    assign_module, compile_cmd, module_path = compile_and_load_assign_module(
        fortran_source=state.fortran_source,
        build_dir=build_dir,
    )

    discovered_mice = _discover_mice(state.data_root)
    selected_mice = state.mice if state.mice is not None else discovered_mice
    selected_mice = [m for m in selected_mice if m in discovered_mice]
    if not selected_mice:
        raise RuntimeError("No valid mice selected for grid computation.")

    all_results: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selected_mice": selected_mice,
        "per_mouse": {},
    }

    for mouse_name in selected_mice:
        print(f"\n## Mouse: {mouse_name}")
        mouse_out = state.root_output / mouse_name
        out_trials = mouse_out / "trials_grid"
        out_grid = mouse_out / "grid"
        out_trials.mkdir(parents=True, exist_ok=True)
        out_grid.mkdir(parents=True, exist_ok=True)

        df_trials = load_trial_metadata(state.meta_root, mouse_name)
        df_eligible = _eligible_trials(df_trials)

        positions, min_act, max_act = load_neuron_metadata(state.meta_root, mouse_name)
        xyz_ranges = get_ranges_from_positions(positions)
        _xyz_coords, xyz_lines = create_grid_3d(xyz_ranges, state.num_grid)
        _ = xyz_lines

        np.save(out_grid / "num_grid.npy", np.asarray(state.num_grid, dtype=np.int64))
        np.save(out_grid / "xyz_ranges.npy", np.asarray(xyz_ranges, dtype=np.float64))

        norm_token = _normalization_token(state.normalization)
        prefix = f"grid-{state.num_grid[0]}x{state.num_grid[1]}x{state.num_grid[2]}_norm-{norm_token}_"

        responses_dir = state.data_root / mouse_name / "data" / "responses"
        if not responses_dir.exists():
            raise FileNotFoundError(f"Responses folder not found: {responses_dir}")

        processed: List[Dict[str, object]] = []
        missing: List[Dict[str, object]] = []
        failed: List[Dict[str, object]] = []
        skipped_existing: List[Dict[str, object]] = []

        for _, row in df_eligible.iterrows():
            trial_id = int(row["trial"])
            valid_frames = int(row["valid_frames"])
            result = _run_single_trial(
                mouse_name=mouse_name,
                trial_id=trial_id,
                valid_frames=valid_frames,
                responses_dir=responses_dir,
                file_prefix=prefix,
                output_trials_dir=out_trials,
                positions=positions,
                min_activation=min_act,
                max_activation=max_act,
                normalization=state.normalization,
                num_grid=state.num_grid,
                xyz_ranges=xyz_ranges,
                assign_module=assign_module,
                skip_existing=state.skip_existing,
            )
            status = result["status"]
            if status == "processed":
                processed.append(result)
            elif status == "missing":
                missing.append(result)
            elif status == "failed":
                failed.append(result)
                print(f"  Trial {trial_id}: FAILED - {result['message']}")
            elif status == "skipped_existing":
                skipped_existing.append(result)

        specs = {
            "mouse": mouse_name,
            "num_grid": [int(x) for x in state.num_grid],
            "normalization": state.normalization,
            "normalization_source": "responses.get_data(by_minmax) parity",
            "trial_filter": "valid_response == True and valid_trial == True",
            "trial_file_match_rule": "trial id extracted from filename via trial-<id> or numeric stem",
            "output_filename_pattern": f"{prefix}rec-<mouse_name>_trial-<trial_id>.npy",
            "output_prefix": prefix,
            "fortran_source": str(state.fortran_source),
            "fortran_compile_command": [str(c) for c in compile_cmd],
            "fortran_module_path": str(module_path),
            "axis_convention": "Fortran output (Nz,Ny,Nx) transposed to (Nx,Ny,Nz)",
            "fortran_transpose": [2, 1, 0],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "counts": {
                "eligible": int(len(df_eligible)),
                "processed": int(len(processed)),
                "missing": int(len(missing)),
                "failed": int(len(failed)),
                "skipped_existing": int(len(skipped_existing)),
            },
            "zero_denom_neurons": int(np.sum((max_act - min_act) == 0)),
            "n_workers": int(state.n_workers),
            "note": "n_workers parameter currently reserved; processing is sequential for deterministic behavior.",
        }
        with open(mouse_out / "grid_specs.json", "w", encoding="utf-8") as f:
            json.dump(specs, f, indent=2)

        mouse_result = {
            "counts": specs["counts"],
            "failed_examples": failed[:10],
            "missing_examples": missing[:10],
            "grid_specs": str(mouse_out / "grid_specs.json"),
        }

        if state.compare_reference_root is not None:
            comp = compare_with_reference(
                output_root=state.root_output,
                reference_root=state.compare_reference_root,
                mouse_name=mouse_name,
            )
            mouse_result["comparison"] = comp
            print(
                "  Comparison:",
                f"ok={comp['ok']}",
                f"common={comp.get('n_common', 0)}",
                f"mismatches={len(comp.get('mismatches', []))}",
                f"out_only={comp.get('n_out_only', 0)}",
                f"ref_only={comp.get('n_ref_only', 0)}",
            )

        all_results["per_mouse"][mouse_name] = mouse_result
        print(
            "  Summary:",
            f"eligible={len(df_eligible)}",
            f"processed={len(processed)}",
            f"missing={len(missing)}",
            f"failed={len(failed)}",
            f"skipped_existing={len(skipped_existing)}",
        )

    summary_path = state.root_output / "compute_grid_activation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    all_results["summary_path"] = str(summary_path)
    return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute 3D grid activations for valid trials and save per-mouse outputs."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--meta-root", required=True, type=Path)
    parser.add_argument("--root-output", required=True, type=Path)
    parser.add_argument("--num-grid", nargs=3, type=int, default=[15, 15, 10], metavar=("NX", "NY", "NZ"))
    parser.add_argument(
        "--normalization",
        type=str,
        default="by_minmax",
        help="Use 'none' or 'by_minmax'.",
    )
    parser.add_argument("--mice", type=_opt_csv_list, default=None)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compare-reference-root", type=Path, default=None)
    parser.add_argument(
        "--fortran-source",
        type=Path,
        default=_default_fortran_source(),
        help=(
            "Path to assign_grid.f90. "
            "Default points to ZigZagSensorium/scripts/assign_grid.f90."
        ),
    )
    return parser


def _normalize_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"none", "null", ""}:
        return None
    if v == "by_minmax":
        return "by_minmax"
    raise ValueError("--normalization must be one of: none, by_minmax")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    num_grid = tuple(int(x) for x in args.num_grid)
    if len(num_grid) != 3 or any(x <= 0 for x in num_grid):
        raise ValueError("--num-grid requires 3 positive integers")

    state = RunState(
        data_root=args.data_root,
        meta_root=args.meta_root,
        root_output=args.root_output,
        num_grid=(num_grid[0], num_grid[1], num_grid[2]),
        normalization=_normalize_arg(args.normalization),
        mice=args.mice,
        n_workers=int(args.n_workers),
        skip_existing=bool(args.skip_existing),
        compare_reference_root=args.compare_reference_root,
        fortran_source=args.fortran_source,
    )

    result = run_pipeline(state)
    print("\nCompleted grid activation computation.")
    print(f"Summary: {result['summary_path']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise