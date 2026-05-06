import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file and return it as a pandas DataFrame.

    Parameters:
    -----------
    path : str
        The path to the CSV file.

    Returns:
    --------
    pd.DataFrame
        The loaded dataset.
        
    Raises:
    -------
    FileNotFoundError:
        If the file does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)

    print(f"[INFO] Loaded data from '{path}'")
    print(f"[INFO] Shape: {df.shape}")
    return df