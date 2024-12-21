import os

def save_to_csv(dataframe, filename, folder='data/processed/'):
    """
    Save a DataFrame to a CSV file in the specified folder, relative to the project root.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to save.
        filename (str): Name of the CSV file (e.g., 'final_dataset'). The '.csv' extension will be added automatically if not included.
        folder (str): Folder path where the file will be saved (default: 'data/processed/').

    Returns:
        None
    """
    # Get the current working directory of the notebook
    notebook_dir = os.getcwd()
    
    # Calculate the project root based on the structure
    project_root = os.path.abspath(os.path.join(notebook_dir, "../"))  # Adjust according to structure
    folder_path = os.path.join(project_root, folder)

    # Ensure the filename has the .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Construct the full file path
    file_path = os.path.join(folder_path, filename)

    try:
        os.makedirs(folder_path, exist_ok=True)  # Ensure the directory exists
        dataframe.to_csv(file_path, index=False)
        print(f"File saved successfully")
    except Exception as e:
        print(f"Error saving file: {e}")

