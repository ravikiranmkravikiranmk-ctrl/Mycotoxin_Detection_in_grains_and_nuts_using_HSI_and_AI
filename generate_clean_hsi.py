import numpy as np
import h5py
import os

# --- Configuration (CRITICAL: DATASET_NAME is set to 'sample' and file is clean) ---
OUTPUT_FILENAME = "clean_peanut_sample.h5" 
HSI_HEIGHT = 60    
HSI_WIDTH = 60     
NUM_BANDS = 150    
DATASET_NAME = "sample" # This MUST match the key your project uses: f['sample'][:]

# --- Simulation Function (ULTRA-UNIFORM DATA for SNV) ---

def generate_snv_neutral_hsi(h, w, bands):
    """
    Generates a near-perfectly uniform HSI cube. When this data goes through 
    your project's apply_snv function, it will result in a cube of near-zero values, 
    which your model recognizes as "Clean."
    """
    
    print(f"Generating SNV-neutral HSI data...")
    
    wavelengths = np.linspace(400, 1000, bands, dtype=np.float32) 
    CONSTANT_REFLECTANCE = 0.65 

    hsi_cube = np.full((h, w, bands), CONSTANT_REFLECTANCE, dtype=np.float32)

    # Add minimal noise (1e-6) to satisfy SNV's non-zero standard deviation check
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
        print(f"Internal Dataset Key: '{dataset_name}'")
        
    except Exception as e:
        print(f"An error occurred while saving the H5 file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    hsi_data, wavelengths = generate_snv_neutral_hsi(HSI_HEIGHT, HSI_WIDTH, NUM_BANDS)
    save_hsi_to_h5(hsi_data, wavelengths, OUTPUT_FILENAME, DATASET_NAME)