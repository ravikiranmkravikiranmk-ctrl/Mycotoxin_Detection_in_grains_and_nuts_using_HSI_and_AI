import os
import numpy as np
import tensorflow as tf
import cv2
import random

from mycotoxin_hsi.data_utils import load_h5_dataset
from mycotoxin_hsi.preprocess import apply_snv

MODEL_PATH = "data/models/mycotoxin_detector.h5"
DATA_PATH = "data/dataset.h5"
OUTPUT_PATH = "outputs/detection_result.png"

def visualize_detection(hsi_cube, true_label, pred_label, confidence):
    B = hsi_cube.shape[-1]
    rgb_bands = [int(B * 0.7), int(B * 0.5), int(B * 0.3)]
    rgb_image = hsi_cube[:, :, rgb_bands]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    text_pred = f"Predicted: {pred_label} ({confidence:.1%})"
    text_true = f"Ground Truth: {true_label}"
    color = (0, 255, 0) if pred_label == "Clean" else (0, 0, 255)
    cv2.putText(rgb_image, text_pred, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(rgb_image, text_true, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return rgb_image

def main():
    print("--- Mycotoxin Detection Demo ---")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'. Please run 'train.py' first.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    X, y, _, class_names = load_h5_dataset(DATA_PATH)
    idx = random.randint(0, len(X) - 1)
    sample_cube, true_label_idx = X[idx], y[idx]
    print(f"-> Running detection on a random sample (ID: {idx}).")
    print(f"-> Ground Truth: {class_names[true_label_idx]}")
    processed_sample = apply_snv(sample_cube)[np.newaxis, ..., np.newaxis]
    probabilities = model.predict(processed_sample, verbose=0)[0]
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
    print(f"-> Prediction: {class_names[predicted_idx]} with {confidence:.2%} confidence.")
    vis_image = visualize_detection(sample_cube, class_names[true_label_idx], class_names[predicted_idx], confidence)
    cv2.imwrite(OUTPUT_PATH, vis_image)
    # CORRECTED LINE
    print(f"-> Detection visualization saved to '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    main()
