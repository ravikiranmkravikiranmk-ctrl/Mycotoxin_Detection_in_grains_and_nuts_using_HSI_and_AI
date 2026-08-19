import numpy as np
import h5py
import os

# --- Configuration ---
# File name for the final, guaranteed-clean colored box
OUTPUT_FILENAME = "clean_sample_colored_box.h5" 
HSI_HEIGHT = 60    
HSI_WIDTH = 60     
NUM_BANDS = 150    
DATASET_NAME = "sample" 

# --- Simulation Function (UNIFORM CURVED SPECTRA + GUARANTEED CLEAN) ---

def generate_robust_clean_hsi(h, w, bands):
    """
    Generates an HSI cube that is spectrally realistic (curved) and mathematically
    guaranteed to be classified as CLEAN (uniform across space).
    This uniformity prevents interpolation errors in the dashboard.
    """
    
    print(f"Generating robustly CLEAN and colored HSI data...")
    
    wavelengths = np.linspace(400, 1000, bands, dtype=np.float32) 

    # 1. Create a Realistic Curved Spectral Base (Crucial for COLOR)
    w_norm = (wavelengths - 400) / (1000 - 400)
    BASE_SPECTRUM = 0.3 + 0.5 * np.exp(w_norm * 1.2) / np.exp(1.2) 
    BASE_SPECTRUM = np.clip(BASE_SPECTRUM, 0.2, 0.95).astype(np.float32)

    # 2. Apply the same base spectrum to the entire cube (Uniform in space)
    hsi_cube = np.zeros((h, w, bands), dtype=np.float32)
    hsi_cube[:, :, :] = BASE_SPECTRUM * 1.0 
    
    # 3. Add global, minimal noise
    noise = np.random.uniform(-1e-6, 1e-6, hsi_cube.shape)
    hsi_cube += noise
    
    hsi_cube = np.clip(hsi_cube, 0.0, 1.0)
    return hsi_cube, wavelengths

def save_hsi_to_h5(hsi_data, wavelengths, filename, dataset_name):
    """Saves the HSI data and metadata to an HDF5 (.h5) file."""
    try:
        with h5py.File(filename, 'w') as f:
            f.create_dataset(dataset_name, data=hsi_data, compression="gzip")
            f.create_dataset("wavelengths", data=wavelengths, compression="gzip")
            
        print(f"\n--- SUCCESS ---")
        print(f"Successfully created: {filename}")
        print(f"Data shape (H x W x Bands): {hsi_data.shape}")
        
    except Exception as e:
        print(f"An error occurred while saving the H5 file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    hsi_data, wavelengths = generate_robust_clean_hsi(HSI_HEIGHT, HSI_WIDTH, NUM_BANDS)
    save_hsi_to_h5(hsi_data, wavelengths, OUTPUT_FILENAME, DATASET_NAME)