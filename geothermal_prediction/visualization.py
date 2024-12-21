import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colormaps
import seaborn as sns
import missingno as msno


def visualize_missing_values(data, figsize=(14, 5), font_size=12, color='cornflowerblue'):
    """
    Visualize missing values in a dataset using bar plots and heatmaps.

    Parameters:
        data (pd.DataFrame): The dataset to visualize.
        figsize (tuple): Figure size for the plots (default: (14, 5)).
        font_size (int): Font size for the bar chart (default: 12).
        color (str): Color for the bar chart (default: 'cornflowerblue').

    Returns:
        None
    """
    # Bar plot of missing values
    fig, axes = plt.subplots(figsize=figsize)
    msno.bar(df=data, ax=axes, fontsize=font_size, color=color)
    axes.set_xlabel('Columns')
    axes.set_ylabel('Non-Missing Values')
    axes.set_title('Total of Observations')
    axes.grid(axis='y', linestyle='--')
    plt.show()

    # Print missing values
    print('Missing values:')
    print(data.isna().sum())

    # Heatmap of missing values
    fig, axes = plt.subplots(figsize=figsize)
    sns.set_context('paper')
    sns.heatmap(data.isna(), yticklabels=False, cbar=False, cmap='Blues', ax=axes)
    axes.set_title('Missingness Pattern in Original Dataset')
    plt.show()


def plot_file_logs(log_data, well_name, logs_to_plot):
    """
    Plot log curves for a specific well in a single row of subplots.

    Parameters:
        log_data (dict): Dictionary containing log DataFrames for each well.
        well_name (str): Well name.
        logs_to_plot (list): List of logs to plot.

    Returns:
        None
    """
    if well_name not in log_data:
        raise ValueError(f"Well '{well_name}' not found in log data.")

    well_logs = log_data[well_name]

    # Calculate overall min and max depth across all logs to plot
    min_depth = float('inf')
    max_depth = float('-inf')

    for log in logs_to_plot:
        if log in well_logs:
            df = well_logs[log]
            min_depth = min(min_depth, df['Depth'].min())
            max_depth = max(max_depth, df['Depth'].max())

    # Create subplots: 1 row, len(logs_to_plot) columns
    num_logs = len(logs_to_plot)
    fig, axes = plt.subplots(1, num_logs, figsize=(6 * num_logs, 10), sharey=True)

    # Ensure axes is iterable (handles single subplot case)
    if num_logs == 1:
        axes = [axes]

    for idx, log in enumerate(logs_to_plot):
        ax = axes[idx]

        # Check if the log exists for the given well
        if log in well_logs:
            df = well_logs[log]

            # Plot data for each File_Index
            for file_index in df['File_Index'].unique():
                file_df = df[df['File_Index'] == file_index]
                ax.plot(file_df['Value'], file_df['Depth'], label=f'File {file_index}')

            # Fetch the unit if available
            unit = df['Unit'].iloc[0] if 'Unit' in df.columns and not df['Unit'].isnull().all() else ''
            unit_label = f' ({unit})' if unit else ''

            # Customize each subplot
            ax.set_ylim(min_depth, max_depth)  # Apply common depth range for all logs
            ax.set_title(f'{log}{unit_label}', fontsize=12)  # Set title with log and unit
            if idx == 0:
                ax.set_ylabel('Depth (m)')
            ax.legend()
            ax.grid(True)
            ax.invert_yaxis()  # Invert the y-axis
        else:
            # If the log does not exist, show an empty plot with a warning
            ax.text(0.5, 0.5, f"No data for {log}", ha="center", va="center", fontsize=12)
            ax.set_title(f"{log}")
            ax.axis("off")  # Hide axes for missing logs

    # Add a main title for the well
    fig.suptitle(f"Log Plots for Well {well_name}", fontsize=16, y=1)

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()


def plot_selected_logs(log_data, well_name):
    """
    Plot all selected log curves for a specific well in a single row of subplots.

    Parameters:
        log_data (dict): Dictionary containing log DataFrames for each well.
        well_name (str): Well name.

    Returns:
        None
    """
    if well_name not in log_data:
        raise ValueError(f"Well '{well_name}' not found in the log data.")

    well_logs = log_data[well_name]
    logs_to_plot = list(well_logs.keys())  # Get all logs available for this well

    # Calculate overall min and max depth across all logs
    min_depth = float('inf')
    max_depth = float('-inf')

    for log in logs_to_plot:
        if log in well_logs:
            df = well_logs[log]
            min_depth = min(min_depth, df['Depth'].min())
            max_depth = max(max_depth, df['Depth'].max())

    # Create subplots: 1 row, len(logs_to_plot) columns
    num_logs = len(logs_to_plot)
    fig, axes = plt.subplots(1, num_logs, figsize=(6 * num_logs, 10), sharey=True)

    # Ensure axes are iterable (handles the case of one log)
    if num_logs == 1:
        axes = [axes]

    for idx, log in enumerate(logs_to_plot):
        ax = axes[idx]

        # Check if the log exists in the selected data
        if log in well_logs:
            df = well_logs[log]

            # Plot data for each File_Index
            for file_index in df['File_Index'].unique():
                file_df = df[df['File_Index'] == file_index]
                ax.plot(file_df['Value'], file_df['Depth'], label=f'File {file_index}')

            # Fetch unit if available
            unit = df['Unit'].iloc[0] if 'Unit' in df.columns and not df['Unit'].isnull().all() else ''
            unit_label = f' ({unit})' if unit else ''

            # Customize each subplot
            ax.set_ylim(min_depth, max_depth)  # Apply common depth range for all logs
            ax.set_title(f'{log}{unit_label}', fontsize=12)
            ax.legend()
            ax.grid(True)
            ax.invert_yaxis()  # Invert the y-axis for depth
            if idx == 0:
                ax.set_ylabel('Depth (m)')
        else:
            # If the log does not exist, display an empty plot with a message
            ax.text(0.5, 0.5, f'No data for {log}', ha='center', va='center', fontsize=12)
            ax.set_title(f'{log}')
            ax.axis('off')

    # Add a main title for the well
    fig.suptitle(f'Selected Log Plots for Well {well_name}', fontsize=16, y=1)

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()


def plot_properties_by_well(dataset, well_name, property_mapping):
    """
    Plot the properties for a specific well in a single row of subplots with markers for individual data points.

    Parameters:
        dataset (pd.DataFrame): Final dataset containing all wells and properties.
        well_name (str): Well name to filter and plot.
        property_columns (list): List of property columns to plot.

    Returns:
        None
    """
    property_columns = list(property_mapping.keys())
    
    # Filter dataset for the specified well
    well_data = dataset[dataset['Well_ID'] == well_name]

    if well_data.empty:
        print(f"No data found for well '{well_name}'.")
        return

    # Create subplots: 1 row, len(property_columns) columns
    num_properties = len(property_columns)
    fig, axes = plt.subplots(1, num_properties, figsize=(6 * num_properties, 10), sharey=True)

    # Ensure axes is iterable (handles single subplot case)
    if num_properties == 1:
        axes = [axes]

    for idx, prop in enumerate(property_columns):
        ax = axes[idx]

        if prop in well_data.columns and not well_data[prop].isna().all():
            # Plot the property against Depth with lines and markers
            ax.plot(well_data[prop], well_data['Depth'], label=prop, linestyle='-', marker='o', markersize=4)

            # Customize each subplot
            ax.set_ylim(well_data['Depth'].min(), well_data['Depth'].max())  # Set depth limits
            ax.set_title(prop, fontsize=12)  # Set title with property name
            if idx == 0:
                ax.set_ylabel("Depth (m)")
            ax.legend()
            ax.grid(True)
            ax.invert_yaxis()  # Invert the y-axis for depth
        else:
            # If the property has no data, show an empty plot with a warning
            ax.text(0.5, 0.5, f"No data for {prop}", ha="center", va="center", fontsize=12)
            ax.set_title(prop)
            ax.axis("off")  # Hide axes for missing properties

    # Add a main title for the well
    fig.suptitle(f"Property Plots for Well {well_name}", fontsize=16, y=1)

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()
    
    
    
def plot_properties_by_well(dataset, well_name, list_properties):
    """
    Plot specified lab properties for a specific well in a single row of subplots with markers for individual data points.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing the lab data.
        well_name (str): Well name to filter and plot.
        list_properties (list): List of property column names to plot.

    Returns:
        None
    """
    # Filter dataset for the specified well
    well_data = dataset[dataset['Well_ID'] == well_name]

    if well_data.empty:
        print(f"No data found for well '{well_name}'.")
        return

    # Create subplots: 1 row, len(list_properties) columns
    num_properties = len(list_properties)
    fig, axes = plt.subplots(1, num_properties, figsize=(6 * num_properties, 10), sharey=True)

    # Ensure axes is iterable (handles single subplot case)
    if num_properties == 1:
        axes = [axes]

    for idx, prop in enumerate(list_properties):
        ax = axes[idx]

        if prop in well_data.columns and not well_data[prop].isna().all():
            # Plot the property against Depth with lines and markers
            ax.scatter(well_data[prop], well_data['Depth'], label=prop)

            # Customize each subplot
            ax.set_ylim(well_data['Depth'].min(), well_data['Depth'].max())  # Set depth limits
            ax.set_title(prop, fontsize=12)  # Set title with property name
            if idx == 0:
                ax.set_ylabel("Depth (m)")
            ax.legend()
            ax.grid(True)
            ax.invert_yaxis()  # Invert the y-axis for depth
        else:
            # If the property has no data, show an empty plot with a warning
            ax.text(0.5, 0.5, f"No data for {prop}", ha="center", va="center", fontsize=12)
            ax.set_title(prop)
            ax.axis("off")  # Hide axes for missing properties

    # Add a main title for the well
    fig.suptitle(f"Lab Property Plots for Well {well_name}", fontsize=16, y=1)

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()



def plot_properties_with_formation(dataset, well_name, property_columns, formation_column='Formation'):
    """
    Plot the properties for a specific well and include geological formation layers.

    Parameters:
        dataset (pd.DataFrame): Dataset containing well data, properties, and formation information.
        well_name (str): Well name to filter and plot.
        property_columns (list): List of property columns to plot.
        formation_column (str): Name of the column with formation information.

    Returns:
        None
    """
    # Filter dataset for the specified well
    well_data = dataset[dataset['Well_ID'] == well_name]

    if well_data.empty:
        print(f"No data found for well '{well_name}'.")
        return

    # Create subplots: 1 row, len(property_columns) columns
    num_properties = len(property_columns)
    fig, axes = plt.subplots(1, num_properties, figsize=(6 * num_properties, 10), sharey=True)

    # Ensure axes is iterable (handles single subplot case)
    if num_properties == 1:
        axes = [axes]

    # Generate a unique pastel color for each formation
    formations = well_data[well_data[formation_column] != 'Unknown_Formation'][formation_column].unique()
    color_map = colormaps.get_cmap('Pastel1')  # Use Pastel1 colormap
    formation_color_dict = {formation: color_map(i / len(formations)) for i, formation in enumerate(formations)}

    for idx, prop in enumerate(property_columns):
        ax = axes[idx]

        if prop in well_data.columns and not well_data[prop].isna().all():
            # Add formation layers first
            for formation in formations:
                # Get depth range for the formation
                formation_depths = well_data[well_data[formation_column] == formation]['Depth']
                ymin, ymax = formation_depths.min(), formation_depths.max()

                # Add shaded area for the formation
                ax.axhspan(
                    ymin, ymax,
                    color=formation_color_dict[formation],
                    alpha=0.4
                )

            # Plot the property against Depth with lines and markers on top of the areas
            ax.scatter(well_data[prop], well_data['Depth'])

            # Customize each subplot
            ax.set_ylim(well_data['Depth'].min(), well_data['Depth'].max())  # Set depth limits
            ax.set_title(prop, fontsize=12)  # Set title with property name
            if idx == 0:
                ax.set_ylabel("Depth (m)")
            ax.grid(True)
            ax.invert_yaxis()  # Invert the y-axis for depth

            # Add legend only for the first subplot
            if idx == 0:
                for formation, color in formation_color_dict.items():
                    ax.scatter([], [], color=color, label=formation)
                ax.legend(title="Formations", loc="upper left")
        else:
            # Skip properties with no data
            ax.axis("off")  # Hide axes for missing properties

    # Add a main title for the well
    fig.suptitle(f"Property Plots for Well {well_name}", fontsize=16, y=1)

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.show()
    
    

    