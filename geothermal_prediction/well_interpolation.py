import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

def cols_to_interpolation(df_inter, cols_interpolation, x_col):
    """
    Perform interpolation for specified columns in a DataFrame using PCHIP Interpolation.

    Parameters:
        df_inter (pd.DataFrame): DataFrame containing the data to be interpolated.
        cols_interpolation (list): List of column names to interpolate.
        x_col (str): Column name to be used as the x-axis for interpolation.

    Returns:
        pd.DataFrame: DataFrame with new columns containing interpolated values for each specified column.
    """
    for col in cols_interpolation:
        df_inter_cleaned = df_inter.dropna(subset=[col, x_col])
        df_inter_cleaned = df_inter_cleaned.sort_values(by=x_col)
        
        # Check if there are at least 2 data points for interpolation
        if len(df_inter_cleaned) < 2:
            # If insufficient data, fill the new interpolated column with NaN
            df_inter[f'{col}_interpolated'] = np.nan
            continue

        x = df_inter_cleaned[x_col]
        y = df_inter_cleaned[col]

        interp_func = PchipInterpolator(x, y)
        interpolated_values = interp_func(df_inter[x_col])

        df_inter[f'{col}_interpolated'] = interpolated_values

    return df_inter

def identify_missing_intervals_start_end(df, logs_to_check, depth):
    """
    Identify missing intervals at the start and end of depth for specified logs.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        logs_to_check (list): List of log column names to check for missing intervals.
        depth (str): Column name representing depth.

    Returns:
        dict: Dictionary where keys are log names and values are lists of tuples
              representing the start and end depths of missing intervals.
    """
    missing_intervals_start_end = {}

    for log in logs_to_check:
        depth_values = df[depth]
        log_values = df[log]

        missing_intervals = []

        # Check for missing intervals at the beginning
        start_depth = None
        end_depth = None

        # Find start depth
        i = 0
        while i < len(log_values) and pd.isnull(log_values.iloc[i]):
            i += 1
        if i > 0:
            start_depth = depth_values.iloc[0]
            end_depth = depth_values.iloc[i-1] if i - 1 < len(depth_values) else None
            missing_intervals.append((start_depth, end_depth))

        # Find end depth
        j = len(log_values) - 1
        while j >= 0 and pd.isnull(log_values.iloc[j]):
            j -= 1
        if j < len(log_values) - 1:
            start_depth = depth_values.iloc[j+1] if j + 1 < len(depth_values) else None
            end_depth = depth_values.iloc[-1]
            missing_intervals.append((start_depth, end_depth))

        # Swap values if start depth is greater than end depth
        missing_intervals = [(min(start_depth, end_depth), max(start_depth, end_depth)) for start_depth, end_depth in missing_intervals]

        missing_intervals_start_end[log] = missing_intervals

    return missing_intervals_start_end

def replace_nan_with_interpolated_start_end(df, intervals_dict, logs_to_interpolate, depth):
    """
    Replace NaN values within specified intervals at the start and end of depth with interpolated values.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        intervals_dict (dict): Dictionary of intervals with start and end depths for each log.
        logs_to_interpolate (list): List of log column names to replace NaNs.
        depth (str): Column name representing depth.

    Returns:
        pd.DataFrame: DataFrame with NaN values replaced in the specified intervals.
    """
    for log in logs_to_interpolate:
        interpolated_column = f'{log}_interpolated'
        for interval_start, interval_end in intervals_dict.get(log, []):
            df.loc[(df[depth] >= interval_start) & (df[depth] <= interval_end), interpolated_column] = \
                df.loc[(df[depth] >= interval_start) & (df[depth] <= interval_end), log]

    return df

def identify_missing_intervals(depth_values, log_values):
    """
    Identify all missing intervals in a log based on depth.

    Parameters:
        depth_values (array-like): Array of depth values.
        log_values (array-like): Array of log values.

    Returns:
        list: List of tuples representing the start and end depths of missing intervals.
    """
    missing_intervals = []
    start_depth = None

    for i, (depth, value) in enumerate(zip(depth_values, log_values)):
        if np.isnan(value):
            if start_depth is None:
                start_depth = depth
        else:
            if start_depth is not None:
                missing_intervals.append((start_depth, depth))
                start_depth = None

    if start_depth is not None:
        missing_intervals.append((start_depth, depth_values[-1]))

    return missing_intervals

def find_intervals_larger_than_threshold(intervals, threshold):
    """
    Find intervals larger than a given threshold.

    Parameters:
        intervals (list): List of tuples representing intervals.
        threshold (float): Threshold for interval size.

    Returns:
        list: List of intervals larger than the specified threshold.
    """
    return [interval for interval in intervals if interval[1] - interval[0] > threshold]

def identify_and_store_missing_intervals(df, logs_to_interpolate, threshold, depth):
    """
    Identify and store missing intervals in logs that exceed a given size threshold.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        logs_to_interpolate (list): List of log column names to check for missing intervals.
        threshold (float): Threshold for interval size.
        depth (str): Column name representing depth.

    Returns:
        dict: Dictionary where keys are log names and values are lists of intervals exceeding the threshold.
    """
    missing_intervals_dict = {}

    for log in logs_to_interpolate:
        depth_values = df[depth].values
        log_values = df[log].values

        missing_intervals = identify_missing_intervals(depth_values, log_values)
        large_intervals = find_intervals_larger_than_threshold(missing_intervals, threshold)

        missing_intervals_dict[log] = large_intervals

    return missing_intervals_dict

def replace_nan_with_interpolated(df, intervals_dict, logs_to_interpolate, depth):
    """
    Replace NaN values within specified intervals with interpolated values.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        intervals_dict (dict): Dictionary of intervals with start and end depths for each log.
        logs_to_interpolate (list): List of log column names to replace NaNs.
        depth (str): Column name representing depth.

    Returns:
        pd.DataFrame: DataFrame with NaN values replaced in the specified intervals.
    """
    for log in logs_to_interpolate:
        interpolated_column = f'{log}_interpolated'
        for interval in intervals_dict.get(log, []):
            start_depth, end_depth = interval
            df.loc[(df[depth] >= start_depth) & (df[depth] <= end_depth), interpolated_column] = \
                df.loc[(df[depth] >= start_depth) & (df[depth] <= end_depth), log]

    return df

def perform_interpolation(wells, cols_to_interpolate, threshold, depth_name, well_name):
    """
    Perform interpolation for logs within individual wells, handling missing intervals.

    Parameters:
        wells (pd.DataFrame): DataFrame containing well data.
        cols_to_interpolate (list): List of column names to interpolate.
        threshold (float): Threshold for identifying large missing intervals.
        depth_name (str): Column name representing depth.
        well_name (str): Column name representing well identifiers.

    Returns:
        pd.DataFrame: DataFrame with interpolated values for each specified column.
    """
    wells_unique = wells[well_name].unique()
    interpolated_wells = []

    for each_well in wells_unique:
        df_interpolation = wells[wells[well_name] == each_well].copy()
        df_complete = cols_to_interpolation(df_inter=df_interpolation,
                                            cols_interpolation=cols_to_interpolate,
                                            x_col=depth_name)
        missing_intervals_start_end = identify_missing_intervals_start_end(df_complete, cols_to_interpolate, depth_name)
        missing_intervals_threshold = identify_and_store_missing_intervals(df_complete, cols_to_interpolate, threshold, depth_name)
        df_inter_nan = replace_nan_with_interpolated_start_end(df_complete, missing_intervals_start_end, cols_to_interpolate, depth_name) 
        df_inter_nan_complete = replace_nan_with_interpolated(df_inter_nan, missing_intervals_threshold, cols_to_interpolate, depth_name)

        interpolated_wells.append(df_inter_nan_complete)



    interpolated_df = pd.concat(interpolated_wells, ignore_index=True)
    return interpolated_df