import numpy as np

def apply_snv(X):
    """
    Applies Standard Normal Variate (SNV) normalization to HSI data.
    Assumes input shape is (N, H, W, B) or (H, W, B).
    """
    if len(X.shape) == 4: # Batch of cubes
        mean = np.mean(X, axis=3, keepdims=True)
        std = np.std(X, axis=3, keepdims=True)
    elif len(X.shape) == 3: # Single cube
        mean = np.mean(X, axis=2, keepdims=True)
        std = np.std(X, axis=2, keepdims=True)
    else:
        raise ValueError("Input must have 3 or 4 dimensions.")
    return (X - mean) / (std + 1e-8)
