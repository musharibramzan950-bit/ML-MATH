"""Parse user input: equation strings, CSV data, JSON arrays."""
import re, io
import numpy as np
import pandas as pd

def parse_input(raw: str, input_type: str = 'auto'):
    """Returns {'X': ndarray, 'y': ndarray|None, 'labels': list, 'raw': str, 'type': str}"""
    raw = raw.strip()

    # Try CSV-style multi-line
    if '\n' in raw or ',' in raw:
        try:
            df = pd.read_csv(io.StringIO(raw), header=None)
            arr = df.values.astype(float)
            if arr.ndim == 1 or arr.shape[1] == 1:
                return {'X': arr.flatten(), 'y': None, 'labels': [], 'raw': raw, 'type': 'array'}
            X, y = arr[:, :-1], arr[:, -1]
            return {'X': X, 'y': y, 'labels': [], 'raw': raw, 'type': 'tabular'}
        except Exception:
            pass

    # Space or semicolon separated numbers
    nums = re.findall(r'[-+]?\d*\.?\d+', raw)
    if len(nums) >= 2:
        arr = np.array([float(n) for n in nums])
        return {'X': arr, 'y': None, 'labels': [], 'raw': raw, 'type': 'array'}

    return {'X': np.array([]), 'y': None, 'labels': [], 'raw': raw, 'type': 'equation'}


def parse_csv_file(file_obj):
    """Parse uploaded CSV file."""
    df = pd.read_csv(file_obj)
    df = df.dropna()
    labels = list(df.columns)
    arr = df.values.astype(float)
    X = arr[:, :-1]
    y = arr[:, -1]
    return {'X': X, 'y': y, 'labels': labels, 'df': df}
