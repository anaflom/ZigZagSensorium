import numpy as np
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys
from pathlib import Path



def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save 3D grid activation per trial for selected recordings."
    )
    parser.add_argument(
        "--folder-data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Path to the folder containing the data.",
    )
    parser.add_argument(
        "--folder-metadata",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "metadata",
        help="Path to the folder containing the metadata.",
    )
    parser.add_argument(
        "--folder-derivatives",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "derivatives",
        help="Path to the folder containing the derivatives.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default='by_minmax',
        help="Normalization passed to response.get_data(normalization=...). Use 'none' for no normalization.",
    )
    parser.add_argument(
        "--num-grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(15, 15, 10),
        help="Grid shape as three integers: NX NY NZ.",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        nargs="+",
        default='none',
        help="Recording IDs to process (space-separated). Use 'none' to apply it to all recordings.",
    )
    return parser.parse_args()



def _compute_and_save_trial(ds, rec, trial, normalization, grid, positions, folder_output_rec_trials, file_prefix=""):
    response = ds.load_response_by_trial(recording=rec, trial=trial, load_metadata_from_global_video=False)
    activities = response.get_data(normalization=normalization)
    grid_activity = grid.compute_grid_activity(positions, activities)
    output_file = f"{file_prefix}rec-{rec}_trial-{trial}.npy"
    np.save(folder_output_rec_trials / output_file, grid_activity)
    return trial


def main(folder_data, folder_metadata, folder_derivatives, normalization=None, num_grid=(15, 15, 10), recordings=None):       

    from ssdatam.dataset import DataSet
    from ssdatam.grids import Grid3D, get_ranges_from_positions

    # path to the output folder for the grids activity per trial
    folder_derivatives.mkdir(exist_ok=True)

    # Number of workers for trial-level parallelism (set GRID_TRIAL_WORKERS env var to override)
    num_workers = int(os.getenv("GRID_TRIAL_WORKERS", "0"))
    if num_workers <= 0:
        cpu_count = os.cpu_count() or 1
        num_workers = min(8, max(1, cpu_count - 1))

    # create output folders
    folder_name = 'grid' 
    folder_name = folder_name + f"-{num_grid[0]}x{num_grid[1]}x{num_grid[2]}"
    if normalization is not None:
        folder_name = folder_name + f"_norm-{normalization}"
    else:
        folder_name = folder_name + "_no-normalization"
    folder_output = folder_derivatives / folder_name
    folder_output.mkdir(exist_ok=True)

    # initialize the object to handle the dataset
    ds = DataSet(folder_data, 
                folder_metadata=folder_metadata, 
                recording=recordings, 
                check=False,
                verbose=True)

    _ = ds.get_trials_metadata()

    # Compute grid activity per trial and save it
    for rec in ds.recording:
        print(f"Computing grid activity for recording: {rec}")

        folder_output_rec = folder_output / rec
        folder_output_rec.mkdir(exist_ok=True)
        folder_output_rec_trials = folder_output_rec / 'trials'
        folder_output_rec_trials.mkdir(exist_ok=True)
        folder_output_rec_grid = folder_output_rec / 'grid'
        folder_output_rec_grid.mkdir(exist_ok=True)

        # Initialize Grid object for that recording
        positions = ds.info[rec]["neurons"].coord_xyz
        xyz_ranges = get_ranges_from_positions(positions)
        grid = Grid3D(xyz_ranges, num_grid)
        
        # Get a DataFrame with all the trials of that recording
        trials_df = ds.filter_trials(recording=rec, valid_response=True, valid_trial=True)

        # save the grid parameters for that recording
        grid.save(folder_output_rec_grid)

        # save a json file with the parameters used to generate grid activity for that recording
        params = {
            "num_grid": list(num_grid),
            "normalization": normalization,
        }
        with open(folder_output_rec / 'params.json', 'w') as f:
            json.dump(params, f, indent=4)

        # Compute grid activity per trial and save it
        all_trials = trials_df['trial'].unique()
        if len(all_trials) == 0:
            print(f"No trials found for recording {rec}, skipping.")
            continue

        workers = min(num_workers, len(all_trials))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _compute_and_save_trial,
                    ds,
                    rec,
                    trial,
                    normalization,
                    grid,
                    positions,
                    folder_output_rec_trials,
                    file_prefix=folder_name + "_",
                )
                for trial in all_trials
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing trials for recording {rec}",
            ):
                future.result()
        


if __name__ == "__main__":
    args = parse_args()
    normalization = None if str(args.normalization).lower() == "none" else args.normalization
    recordings = None if str(args.recordings).lower() == "none" else args.recordings
    try:
        main(
            args.folder_data,
            args.folder_metadata,
            args.folder_derivatives,
            normalization=normalization,
            num_grid=tuple(args.num_grid),
            recordings=recordings,
        )
        print("\nComputation of activity over grids completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")
