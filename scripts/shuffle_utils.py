#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ana Flo <anaflom@gmail.com>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shuffle utilities for ablation studies."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def derive_trial_shuffle_seed(base_seed: int, trial_id: int) -> int:
    """Derive a deterministic per-trial seed from a shuffle-level base seed."""
    return int(np.random.SeedSequence([int(base_seed), int(trial_id)]).generate_state(1)[0])


def shuffle_grid_time_dimension(grid: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle a 4D grid along the time dimension (last axis).
    
    Parameters
    ----------
    grid : np.ndarray
        Grid of shape (Nx, Ny, Nz, T) where T is the time dimension.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Shuffled grid with same shape, where one time permutation is applied
        to the full 3D grid stack for this trial.
    """
    if grid.ndim != 4:
        raise ValueError(f"Grid must be 4D, got shape {grid.shape}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(grid.shape[3])
    return np.copy(grid[:, :, :, perm])


def shuffle_grid_spatial_dimensions(grid: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle spatial positions of a 4D grid with one mapping reused for all frames.

    Parameters
    ----------
    grid : np.ndarray
        Grid of shape (Nx, Ny, Nz, T).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shuffled grid with same shape, where spatial voxel positions are permuted
        once and the same spatial permutation is applied across all time frames.
    """
    if grid.ndim != 4:
        raise ValueError(f"Grid must be 4D, got shape {grid.shape}")

    rng = np.random.default_rng(seed)
    nx, ny, nz, nt = grid.shape
    n_voxels = nx * ny * nz

    # One permutation over spatial voxel indices; reused for all frames.
    flat = np.reshape(np.copy(grid), (n_voxels, nt))
    perm = rng.permutation(n_voxels)
    shuffled_flat = flat[perm, :]
    return np.reshape(shuffled_flat, (nx, ny, nz, nt))


def shuffle_grid_phase(grid: np.ndarray, seed: int) -> np.ndarray:
    """Apply random phase shift in Fourier domain to a 4D grid (vectorized).
    
    For each voxel's time series, compute FFT, add random phase to each frequency bin
    (the phase draw is reused within this trial across all voxels), then IFFT back
    to temporal domain.
    Uses vectorized FFT operations for performance (~3-5× faster than voxel-by-voxel).
    
    Parameters
    ----------
    grid : np.ndarray
        Grid of shape (Nx, Ny, Nz, T).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Grid with phase-shuffled time series, same shape as input.
    """
    if grid.ndim != 4:
        raise ValueError(f"Grid must be 4D, got shape {grid.shape}")
    
    rng = np.random.default_rng(seed)
    nx, ny, nz, nt = grid.shape
    
    # Generate random phases for all frequency bins (same for all voxels)
    random_phases = rng.uniform(0, 2 * np.pi, size=nt)
    
    # Reshape grid to (n_voxels, T) for vectorized FFT
    flat = np.reshape(np.copy(grid), (nx * ny * nz, nt))
    
    # Apply FFT to all voxels at once (axis=1 is time dimension)
    fft_vals = np.fft.fft(flat, axis=1)
    
    # Apply phase shift (broadcasts across all voxels)
    phase_shift = np.exp(1j * random_phases)
    fft_shuffled = fft_vals * phase_shift
    
    # IFFT back to temporal domain (take real part)
    shuffled_flat = np.real(np.fft.ifft(fft_shuffled, axis=1))
    
    return np.reshape(shuffled_flat, (nx, ny, nz, nt))


def compute_zigzag_from_grid(
    grid: np.ndarray,
    threshold: Optional[float] = None,
    p_active: float = 30,
    max_dim: Optional[int] = None,
) -> List[Tuple[int, float, float]]:
    """Compute zigzag persistence bars from a 4D grid.
    
    Parameters
    ----------
    grid : np.ndarray
        Grid of shape (Nx, Ny, Nz, T).
    threshold : float, optional
        Filtration threshold. If None, computed from grid percentile.
    p_active : float
        Percentile of positive activations to use for threshold.
    max_dim : int, optional
        Maximum homology dimension to include (default: include all).
    
    Returns
    -------
    List[Tuple[int, float, float]]
        List of (dim, birth, death) tuples representing zigzag persistence bars.
    """
    from zztop import run_cubical_zigzag
    
    if grid.ndim != 4:
        raise ValueError(f"Grid must be 4D, got shape {grid.shape}")
    
    # Compute threshold if not provided
    if threshold is None:
        positive_vals = grid[grid > 0].ravel()
        if len(positive_vals) == 0:
            threshold = 0.0
        else:
            threshold = -np.percentile(positive_vals, p_active)
    
    # Run zigzag persistence
    bars = run_cubical_zigzag(grid, threshold=threshold)
    
    if max_dim is not None:
        bars = [(dim, b, d) for dim, b, d in bars if dim <= max_dim]
    
    return bars


def validate_mice_for_ablation(
    selected_mice: Optional[List[str]],
    data_root: Path,
    meta_root: Path,
    discovered_mice: List[str],
) -> List[str]:
    """Filter mice to those with at least 2 labels.
    
    Parameters
    ----------
    selected_mice : Optional[List[str]]
        User-specified mice, or None to use discovered mice.
    data_root : Path
        Root directory for data.
    meta_root : Path
        Root directory for metadata.
    discovered_mice : List[str]
        All discovered mice (fallback if selected_mice is None).
    
    Returns
    -------
    List[str]
        Filtered list of mice with at least 2 labels.
    """
    from utils import load_trial_metadata, _eligible_trials
    
    target_mice = selected_mice if selected_mice is not None else discovered_mice
    target_mice = [m for m in target_mice if m in discovered_mice]
    
    eligible_mice: List[str] = []
    for mouse_name in target_mice:
        try:
            df = load_trial_metadata(meta_root, mouse_name)
            df_eligible = _eligible_trials(df)
            n_labels = len(np.unique(df_eligible["label"].values))
            if n_labels >= 2:
                eligible_mice.append(mouse_name)
        except Exception:
            # Skip mice where metadata cannot be loaded
            pass
    
    return sorted(eligible_mice)


def compute_threshold_from_grid_sample(
    grid_paths: List[Path],
    p_active: float,
    n_sample: int = 20,
) -> float:
    """Compute a stable threshold by averaging over a sample of grids.
    
    Parameters
    ----------
    grid_paths : List[Path]
        Paths to grid files.
    p_active : float
        Percentile of positive activations.
    n_sample : int
        Number of grids to sample for threshold estimation.
    
    Returns
    -------
    float
        Estimated threshold.
    """
    n = min(n_sample, len(grid_paths))
    indices = np.linspace(0, len(grid_paths) - 1, n, dtype=int)
    thresholds = []
    for idx in indices:
        grid = np.load(grid_paths[idx])
        positive_vals = grid[grid > 0].ravel()
        if len(positive_vals) > 0:
            t = -np.percentile(positive_vals, p_active)
            thresholds.append(t)
    
    return float(np.mean(thresholds)) if thresholds else 0.0
