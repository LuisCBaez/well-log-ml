import numpy as np
import pandas as pd



def check_logs_by_properties(log_data, list_wells, property_mapping):
    """
    Check the available logs for each well and group them by properties.

    Parameters:
        log_data (dict): Dictionary containing processed log data for each well.
        list_wells (list): List of well names.
        property_mapping (dict): Mapping of properties to their corresponding log names.

    Returns:
        dict: Dictionary summarizing available properties and logs for each well.
    """
    summary = {}

    print("Logs available for each well grouped by properties:\n")
    for well_name in list_wells:
        well_summary = {}
        for prop, logs in property_mapping.items():
            available_logs = []
            for log in logs:
                if log in log_data[well_name] and not log_data[well_name][log].empty:
                    available_logs.append(log)
            if available_logs:
                well_summary[prop] = available_logs
        summary[well_name] = well_summary

        print(f"{well_name}:")
        for prop, logs in well_summary.items():
            print(f"  {prop}: {', '.join(logs)}")
        print()

    return summary


def standardize_depth(df):
    """
    Convert depth to meters.

    Parameters:
        df (pd.DataFrame): DataFrame with depth and depth units.

    Returns:
        pd.DataFrame: Updated DataFrame with depth in meters.
    """
    df["Depth"] = df.apply(
        lambda row: row["Depth"] * 0.3048 if row["Depth_Unit"] == "F" else row["Depth"],
        axis=1,
    )
    df["Depth_Unit"] = "m"
    return df

def replace_outliers(df, outlier_values=None):
    """
    Replace specified outlier values with NaN while preserving data length.

    Parameters:
        df (pd.DataFrame): DataFrame to clean.
        outlier_values (list, optional): List of values to replace with NaN. Defaults to [-9999.0].

    Returns:
        pd.DataFrame: DataFrame with outliers replaced by NaN.
    """
    if outlier_values is None:
        outlier_values = [-9999.0]  # Default outlier value

    # Replace outlier values with NaN
    df["Value"] = df["Value"].replace(outlier_values, np.nan)
    return df

def standardize_porosity(df):
    """
    Convert porosity values > 1 into fractions and replace negative values with NaN.

    Parameters:
        df (pd.DataFrame): DataFrame with porosity values.

    Returns:
        pd.DataFrame: Updated DataFrame with standardized porosity.
    """
    # Convert percentages (>1) to fractions
    df["Value"] = df["Value"].apply(lambda x: x / 100 if x > 1 else x)

    # Replace negative values with NaN
    df["Value"] = df["Value"].apply(lambda x: np.nan if x < 0 else x)

    return df


def process_log(df, log_name, porosity_logs):
    """
    Apply cleaning and standardization steps to a single log.

    Parameters:
        df (pd.DataFrame): DataFrame to process.
        log_name (str): Name of the log being processed.
        porosity_logs (list): List of log names corresponding to porosity.

    Returns:
        pd.DataFrame: Cleaned and standardized DataFrame.
    """
    # Standardize depth
    df = standardize_depth(df)

    # Replace outliers
    df = replace_outliers(df)

    # Standardize porosity if applicable
    if log_name in porosity_logs:
        df = standardize_porosity(df)

    return df


def standardize_all_logs(log_data, property_mapping):
    """
    Standardize and clean all logs in a dataset based on property mapping.

    Parameters:
        log_data (dict): Dictionary of logs for each well.
        property_mapping (dict): Mapping of properties to their corresponding log names.

    Returns:
        dict: Dictionary with standardized and cleaned logs for all wells.
    """
    porosity_logs = property_mapping["Porosity"]  # Extract porosity logs
    cleaned_log_data = {}

    for well_name, logs in log_data.items():
        cleaned_log_data[well_name] = {}

        for log_name, df in logs.items():
            # Process each log
            cleaned_log_data[well_name][log_name] = process_log(df.copy(), log_name, porosity_logs)

    return cleaned_log_data


def select_logs_with_depth(log_data, well_name, selected_logs):
    """
    Select specific logs, file indices, and depth ranges for a given well.

    Parameters:
        log_data (dict): Dictionary containing standardized log DataFrames for each well.
        well_name (str): Name of the well to process.
        selected_logs (dict): Dictionary specifying the logs, files, and depth ranges to keep.

    Returns:
        dict: Dictionary containing the selected logs for the well.
    """
    if well_name not in log_data:
        raise ValueError(f"Well '{well_name}' not found in log data.")

    well_logs = log_data[well_name]
    selected_data = {}

    for log_name, file_filters in selected_logs.items():
        if log_name not in well_logs:
            print(f"Warning: Log '{log_name}' not found for well '{well_name}'. Skipping.")
            continue

        df = well_logs[log_name]
        filtered_dfs = []

        # Process each file index and its associated depth range
        for file_index, filter_info in file_filters.items():
            file_df = df[df['File_Index'] == file_index]

            # Apply depth range filtering if specified
            depth_range = filter_info.get('depth', (None, None))
            min_depth, max_depth = depth_range
            if min_depth is not None:
                file_df = file_df[file_df['Depth'] >= min_depth]
            if max_depth is not None:
                file_df = file_df[file_df['Depth'] <= max_depth]

            if not file_df.empty:
                filtered_dfs.append(file_df)

        # Combine filtered DataFrames for this log
        if filtered_dfs:
            selected_data[log_name] = pd.concat(filtered_dfs, ignore_index=True)
        else:
            print(f"Warning: No data found for log '{log_name}' in the specified files/conditions for well '{well_name}'.")

    return selected_data


def merge_logs_for_well(well_name, selected_logs, property_mapping):
    """
    Merge selected logs for a single well into a unified DataFrame.

    Parameters:
        well_name (str): Name of the well.
        selected_logs (dict): Selected logs for the well, containing DataFrames for each log.
        property_mapping (dict): Mapping of properties to their corresponding log names.

    Returns:
        pd.DataFrame: A unified DataFrame containing the well's logs.
    """
    # Initialize an empty DataFrame with required columns
    merged_df = pd.DataFrame()
    merged_df['Depth'] = []  # Ensure Depth is included

    # Add columns for each property
    for property_name in property_mapping.keys():
        merged_df[property_name] = None  # Create an empty column for each property

    # Iterate through the selected logs and map them to properties
    for property_name, log_names in property_mapping.items():
        for log_name in log_names:
            if log_name in selected_logs:  # Check if this log is available for the well
                log_df = selected_logs[log_name]  # Get the DataFrame for this log
                log_df = log_df[['Depth', 'Value']].rename(columns={'Value': property_name})

                # Merge the log data into the main DataFrame
                if merged_df.empty:
                    merged_df = log_df
                else:
                    merged_df = pd.merge(
                        merged_df, log_df, on='Depth', how='outer'
                    )  # Outer join to preserve all depths

    # Add the 'Well_ID' column
    merged_df['Well_ID'] = well_name

    # Reorder columns
    ordered_columns = ['Well_ID', 'Depth'] + list(property_mapping.keys())
    merged_df = merged_df[ordered_columns]

    return merged_df

def merge_logs_for_all_wells(selected_logs, property_mapping):
    """
    Merge logs for all wells into a single DataFrame.

    Parameters:
        selected_logs (dict): Selected logs for all wells.
        property_mapping (dict): Mapping of properties to their corresponding log names.

    Returns:
        pd.DataFrame: A unified DataFrame containing logs for all wells.
    """
    all_wells_data = []

    for well_name in selected_logs.keys():
        well_df = merge_logs_for_well(well_name, selected_logs[well_name], property_mapping)
        all_wells_data.append(well_df)

    # Concatenate data for all wells
    final_dataset = pd.concat(all_wells_data, ignore_index=True)
    
    # Drop rows where all property columns are NaN
    property_columns = list(property_mapping.keys())
    filtered_dataset = final_dataset.dropna(subset=property_columns, how='all')
    
    filtered_dataset.reset_index(drop=True, inplace=True)
    
    return filtered_dataset
    