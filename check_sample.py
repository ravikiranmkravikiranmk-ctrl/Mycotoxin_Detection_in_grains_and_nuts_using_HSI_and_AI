import os
import numpy as np
import tensorflow as tf
import h5py
import argparse

# --- Import project-specific functions ---
from mycotoxin_hsi.preprocess import apply_snv

# --- Configuration ---
MODEL_PATH = "data/models/mycotoxin_detector.h5"
CLASS_NAMES = ["Clean", "Aflatoxin B1", "Ochratoxin A", "Fumonisin", "DON"]

def check_single_sample(sample_path):
    """
    Loads a single HSI sample, preprocesses it, and predicts if it's contaminated.
    This is the main function of your working project.
    """
    # 1. Check if the model and sample exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained model not found at '{MODEL_PATH}'.")
        print("Please run 'python train.py' first to train and save the model.")
        return

    if not os.path.exists(sample_path):
        print(f"ERROR: Sample file not found at '{sample_path}'.")
        print("Please provide a valid path to an HSI sample file (.h5 format).")
        return

    # 2. Load the trained AI model
    print(f"-> Loading trained model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 3. Load the single HSI sample to be checked
    print(f"-> Loading sample for analysis from '{sample_path}'...")
    with h5py.File(sample_path, 'r') as f:
        # Assumes the .h5 file contains a dataset named 'sample'
        hsi_cube = f['sample'][:]
    
    print(f"-> Sample dimensions: {hsi_cube.shape}")

    # 4. Preprocess the sample exactly as done during training
    print("-> Preprocessing sample (applying SNV)...")
    processed_sample = apply_snv(hsi_cube)
    
    # Add the necessary dimensions for the model (batch and channel)
    # Model expects: (batch_size, height, width, bands, channels)
    input_tensor = processed_sample[np.newaxis, ..., np.newaxis]

    # 5. Make the prediction
    print("-> Running detection with the AI model...")
    probabilities = model.predict(input_tensor, verbose=0)[0]
    
    # 6. Interpret the results
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]
    predicted_class_name = CLASS_NAMES[predicted_index]

    print("\n" + "="*30)
    print("      DETECTION RESULT")
    print("="*30)

    # The final, simple answer
    if predicted_class_name == "Clean":
        print("\nSTATUS: Sample is CLEAN (Not Contaminated)")
    else:
        print("\nSTATUS: Sample is CONTAMINATED")

    print(f"\nDETAILS:")
    print(f"  - Detected Class: {predicted_class_name}")
    print(f"  - Confidence: {confidence:.2%}")
    print("="*30)


if __name__ == "__main__":
    # This allows you to run the script from the command line with an argument
    parser = argparse.ArgumentParser(description="Check a single HSI sample for mycotoxin contamination.")
    parser.add_argument("--sample_path", type=str, required=True, help="Path to the H5 file of the HSI sample to check.")
    
    args = parser.parse_args()
    
    check_single_sample(args.sample_path)