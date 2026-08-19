import os
import re
import cv2
import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_png_folder(input_folder, output_path):
    """
    Reads a folder of PNG slices (e.g., 'sample_123nm.png'), stacks them in
    wavelength order, and saves them as a dashboard-compatible .h5 file.
    """
    input_path = Path(input_folder)
    if not input_path.is_dir():
        print(f"Error: Input folder not found at '{input_folder}'")
        return

    print(f"-> Scanning for PNG files in '{input_folder}'...")
    
    image_files = []
    # Regular expression to extract the wavelength number from the filename
    wavelength_pattern = re.compile(r'(\d+)\s*nm')

    for filename in os.listdir(input_path):
        if filename.lower().endswith(".png"):
            match = wavelength_pattern.search(filename)
            if match:
                wavelength = int(match.group(1))
                image_files.append((wavelength, input_path / filename))
            else:
                print(f"Warning: Could not parse wavelength from '{filename}'. Skipping.")

    if not image_files:
        print(f"Error: No valid PNG files with wavelength information found in the folder.")
        return

    # IMPORTANT: Sort the files numerically by wavelength
    image_files.sort()
    
    print(f"-> Found {len(image_files)} spectral bands. Stacking them into a 3D cube...")

    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_files[0][1]), cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        print(f"Error: Could not read the first image file: {image_files[0][1]}")
        return
    height, width = first_image.shape
    
    # Create an empty 3D numpy array
    num_bands = len(image_files)
    hsi_cube = np.zeros((height, width, num_bands), dtype=np.float32)
    
    wavelengths = []

    # Loop through the sorted files and stack them
    for i, (wavelength, filepath) in enumerate(tqdm(image_files, desc="Processing Bands")):
        # Read the image as grayscale (single channel)
        band_image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        
        # Normalize pixel values to be between 0 and 1
        if band_image.max() > 1:
            band_image = band_image / 255.0
            
        hsi_cube[:, :, i] = band_image
        wavelengths.append(wavelength)
        
    print(f"-> HSI cube created with shape: {hsi_cube.shape} (Height, Width, Bands)")

    # Save to the H5 format that the dashboard understands
    print(f"-> Saving to dashboard-compatible .h5 file at '{output_path}'...")
    with h5py.File(output_path, 'w') as f:
        # The dashboard expects the data under a key named 'sample'
        f.create_dataset('sample', data=hsi_cube)
        # Also save the wavelength information
        f.create_dataset('wavelengths', data=np.array(wavelengths, dtype=np.float32))

    print(f"\nSuccess! Your sample has been converted.")
    print(f"You can now upload '{output_path}' to the dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of PNG slices into a dashboard-compatible .h5 file.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing the PNG image slices (e.g., 'dataset/coffee').")
    parser.add_argument("--output_file", default="converted_sample.h5", help="Name of the output .h5 file to be created.")
    
    args = parser.parse_args()
    convert_png_folder(args.input_folder, args.output_file)