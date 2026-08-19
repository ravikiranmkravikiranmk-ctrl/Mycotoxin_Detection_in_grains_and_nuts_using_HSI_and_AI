import h5py
import json
import numpy as np

def load_h5_dataset(path):
    """Loads HSI data, labels, wavelengths, and class names from an H5 file."""
    with h5py.File(path, "r") as f:
        X = f["data"][:].astype("float32")
        y = f["labels"][:].astype("int64")
        wavelengths = f["wavelengths"][:] if "wavelengths" in f else None
        class_names = json.loads(f.attrs["class_names"])
    # CORRECTED LINE
    print(f"-> Dataset loaded: {X.shape[0]} samples, {len(class_names)} classes.")
    return X, y, wavelengths, class_names
