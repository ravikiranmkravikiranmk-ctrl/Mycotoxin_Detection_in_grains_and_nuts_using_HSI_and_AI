import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from scipy.interpolate import interp1d

# --- Import project-specific functions ---
from mycotoxin_hsi.preprocess import apply_snv

# --- Page Configuration ---
st.set_page_config(
    page_title="Mycotoxin Detection Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
MODEL_PATH = "data/models/mycotoxin_detector.h5"
CLASS_NAMES = ["Clean", "Aflatoxin B1", "Ochratoxin A", "Fumonisin", "DON"]
# The exact input shape the model was trained on
MODEL_INPUT_SHAPE = (32, 32, 64) 

# --- Caching the Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at '{MODEL_PATH}'. Please run 'train.py' first.")
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# --- NEW Preprocessing Function ---
def prepare_hsi_for_model(hsi_cube, target_shape):
    """
    Resizes and resamples an HSI cube to match the model's expected input shape.
    """
    target_h, target_w, target_b = target_shape
    original_h, original_w, original_b = hsi_cube.shape

    # 1. Spatial Resize (using OpenCV)
    resized_bands = []
    for i in range(original_b):
        band = hsi_cube[:, :, i]
        # Ensure band is non-empty before resizing
        if band.size == 0:
            continue
        resized_band = cv2.resize(band, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized_bands.append(resized_band)
    
    # Handle the case where the cube might be empty or resizing failed
    if not resized_bands:
        return np.zeros(target_shape, dtype=np.float32)

    resized_cube = np.stack(resized_bands, axis=-1)

    # 2. Spectral Resampling (using interpolation)
    if original_b == target_b:
        resampled_cube = resized_cube
    else:
        resampled_cube = np.zeros((target_h, target_w, target_b), dtype=np.float32)
        
        original_x = np.linspace(0, 1, original_b)
        target_x = np.linspace(0, 1, target_b)
        
        for i in range(target_h):
            for j in range(target_w):
                # Interpolate function for the current pixel's spectrum
                interp_func = interp1d(original_x, resized_cube[i, j, :], kind='linear', fill_value="extrapolate")
                resampled_cube[i, j, :] = interp_func(target_x)

    return resampled_cube

# --- Helper Functions for Analysis & Visualization ---
# MODIFICATION 1: REMOVED redundant apply_snv call
def predict(hsi_cube, model):
    # Data is now assumed to be SNV-processed AND resized/resampled!
    input_tensor = hsi_cube[np.newaxis, ..., np.newaxis] # Add batch and channel dims
    
    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]
    return probabilities, predicted_index, confidence

def create_pseudo_rgb(hsi_cube):
    B = hsi_cube.shape[-1]
    rgb_bands = [int(B * 0.7), int(B * 0.53), int(B * 0.4)]
    rgb_image = hsi_cube[:, :, rgb_bands]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    return (rgb_image * 255).astype(np.uint8)

def plot_spectral_signature(hsi_cube, wavelengths):
    mean_spectrum = np.mean(hsi_cube, axis=(0, 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavelengths, mean_spectrum, color='dodgerblue', linewidth=2)
    ax.set_title("Average Spectral Signature of the Sample", fontsize=14)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Reflectance (Normalized)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    if wavelengths is not None and len(wavelengths) > 0:
        ax.set_xlim(wavelengths.min(), wavelengths.max())
    plt.tight_layout()
    return fig

# --- Main Dashboard UI ---

model = load_model()

st.sidebar.title("🔬 Mycotoxin Detection")
st.sidebar.info(
    "This dashboard uses a 3D-CNN model to detect mycotoxin contamination "
    "in grain/nut samples from Hyperspectral Imaging (HSI) data."
)
st.sidebar.header("Upload Sample")
uploaded_file = st.sidebar.file_uploader(
    "Choose an HSI sample file (.h5 format)", 
    type=["h5", "hdf5"]
)
st.sidebar.markdown("---")
st.sidebar.write("Project by: Srushti H")

st.title("Mycotoxin Detection in Grains & Nuts")

if uploaded_file is None:
    st.info("👈 Please upload an HSI sample file using the sidebar to begin analysis.")
    if os.path.exists('outputs/detection_result.png'):
        st.image('outputs/detection_result.png', caption='Example of a detection result.', use_column_width=True)

if uploaded_file is not None and model is not None:
    with st.spinner('Analyzing sample... This may take a moment.'):
        with h5py.File(uploaded_file, 'r') as f:
            hsi_cube_original = f['sample'][:]
            try:
                wavelengths_original = f['wavelengths'][:]
            except KeyError:
                wavelengths_original = np.linspace(400, 1000, hsi_cube_original.shape[-1])
        
        # --- MODIFICATION 2: CORRECTED ORDER OF PRE-PROCESSING ---
        
        # 1. Apply SNV FIRST (as done in train.py to preserve spectral features)
        st.write(f"Original data shape: `{hsi_cube_original.shape}`")
        st.write("Applying SNV pre-processing...")
        snv_cube = apply_snv(hsi_cube_original)
        
        # 2. THEN, perform Spatial/Spectral Resampling
        st.write(f"Model expects shape: `(Height: {MODEL_INPUT_SHAPE[0]}, Width: {MODEL_INPUT_SHAPE[1]}, Bands: {MODEL_INPUT_SHAPE[2]})`")
        st.write(f"Resizing and resampling data from {snv_cube.shape} to match model input...")
        
        model_ready_cube = prepare_hsi_for_model(snv_cube, MODEL_INPUT_SHAPE)
        
        st.write(f"Processed data shape for model: `{model_ready_cube.shape}`")

        # Get predictions using the fully processed cube
        probabilities, predicted_index, confidence = predict(model_ready_cube, model)
        predicted_class = CLASS_NAMES[predicted_index]

    st.header("Analysis Results")

    if predicted_class == "Clean":
        st.success(f"**Status: Sample is CLEAN (Not Contaminated)**", icon="✅")
    else:
        st.error(f"**Status: Sample is CONTAMINATED**", icon="⚠️")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Detected Class", value=predicted_class)
        st.metric(label="Confidence Score", value=f"{confidence:.2%}")
        
        st.subheader("Pseudo-RGB Image (Original)")
        rgb_image = create_pseudo_rgb(hsi_cube_original)
        st.image(rgb_image, caption="Visual representation of the original HSI sample.", use_column_width=True)

    with col2:
        st.subheader("Class Probability Distribution")
        prob_df = pd.DataFrame({"Probability": probabilities}, index=CLASS_NAMES)
        st.bar_chart(prob_df)

    st.subheader("Spectral Analysis (Original)")
    fig = plot_spectral_signature(hsi_cube_original, wavelengths_original)
    st.pyplot(fig)