import pandas as pd

def load_raw_data(filepath, sep='\t'):
    """
    Loads raw data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        sep (str): The separator used in the CSV file (default is tab).

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if loading fails.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=sep)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return