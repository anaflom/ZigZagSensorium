import numpy as np
import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and save 3D grid activation stats across trials for selected recordings."
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
        "--subfolder-derivatives",
        type=Path,
        default=Path("grid-15x15x10_norm-by_minmax"),
        help="Subfolder name within the derivatives folder.",
    )
    parser.add_argument(
        "--recordings",
        type=str,
        nargs="+",
        default='none',
        help="Recording IDs to process (space-separated). Use 'none' to apply it to all recordings.",
    )
    return parser.parse_args()


def main(folder_data, folder_metadata, folder_derivatives, subfolder_derivatives, recordings=None):

    from ssdatam.grids import DataSetGrid

    folder_derivatives = Path(folder_derivatives) / subfolder_derivatives

    # initialize the object to handle the dataset
    ds = DataSetGrid(folder_data=folder_data, 
                     folder_metadata=folder_metadata, 
                     folder_derivatives=folder_derivatives,
                     recording=recordings, 
                     check=False,
                     verbose=True)

    # load a dataframe with all trials metadata
    trials_df = ds.get_trials_metadata()
    
    for rec in ds.recording:

        stats = ds.compute_grid_stats(rec, save=True)


if __name__ == "__main__":
    
    args = parse_args()
    recordings = None if str(args.recordings).lower() == "none" else args.recordings
    try:
        main(args.folder_data, 
             args.folder_metadata, 
             args.folder_derivatives, 
             args.subfolder_derivatives, 
             recordings=recordings)
        print("\nGrid activation stats computation completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")



    
