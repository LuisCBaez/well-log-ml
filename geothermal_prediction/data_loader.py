import os
import welly
import pandas as pd


def generate_log_data(list_wells, list_logs, base_path="../data/raw/logs"):
    """
    Extract log data for multiple wells and logs.

    Parameters:
        list_wells (list): List of well names.
        list_logs (list): List of log names.
        base_path (str): Base path for the raw data.

    Returns:
        dict: Dictionary of DataFrames for each well.
    """
    log_data = {}

    for well_name in list_wells:
        path = os.path.join(base_path, well_name)
        las_data = welly.Project.from_las(os.path.join(path, "*.las"))

        well_logs = {}
        for file_index, file_data in enumerate(las_data):
            for log in list_logs:
                if log in file_data.data:
                    depth = file_data.data[log].basis
                    curve = file_data.data[log].as_numpy()
                    unit = file_data.data[log].units
                    depth_unit = file_data.data[log].index_units

                    df = pd.DataFrame({
                        "Depth": depth,
                        "Value": curve,
                        "Log": log,
                        "File_Index": file_index,
                        "Unit": unit,
                        "Depth_Unit": depth_unit,
                    })
                    if log not in well_logs:
                        well_logs[log] = []
                    well_logs[log].append(df)

        # Combine DataFrames for each log
        log_data[well_name] = {
            log: pd.concat(well_logs[log], ignore_index=True) for log in well_logs
        }

    return log_data
