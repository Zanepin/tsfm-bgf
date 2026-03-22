import numpy as np
import pandas as pd
from pathlib import Path

def load_glucose_series(csv_path: str, limit: int = None) -> np.ndarray:
    """
    Basic file reading method: Reads a single subject's CSV into a continuous 1D numpy array (float32 structure)
    :param csv_path: Path to the data file
    :param limit: Whether to truncate the sequence length to the first `limit` size, None means reading the full sequence
    :return: Corresponding numpy array, returns None if invalid or column is insufficient
    """
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 1: 
            return None
        # Safely read the first column and convert to standard numeric format, removing garbled code
        vals = pd.to_numeric(df.iloc[:, 0], errors='coerce').values.astype("float32")
        
        # Remove leading and trailing NaNs
        vals = vals[~np.isnan(vals)]
        
        if limit and len(vals) > limit:
            vals = vals[:limit]
            
        return vals
    except Exception as e:
        return None
