import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from mycotoxin_hsi.data_utils import load_h5_dataset
from mycotoxin_hsi.preprocess import apply_snv
from mycotoxin_hsi.model import build_mycotoxin_3d_cnn

DATA_PATH = "data/dataset.h5"
MODEL_SAVE_PATH = "data/models/mycotoxin_detector.h5"
OUTPUT_DIR = "outputs"
EPOCHS = 25
BATCH_SIZE = 16

def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True)
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    # CORRECTED LINE
    print(f"-> Training history plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    # CORRECTED LINE
    print(f"-> Confusion matrix saved to {save_path}")

def main():
    print("--- Mycotoxin Detection Model Training ---")
    X, y, _, class_names = load_h5_dataset(DATA_PATH)
    print("-> Applying SNV preprocessing...")
    X_processed = apply_snv(X)[..., np.newaxis]
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"-> Data split: {len(X_train)} training, {len(X_val)} validation samples.")
    model = build_mycotoxin_3d_cnn(input_shape=X_train.shape[1:], num_classes=len(class_names))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    ]
    print(f"-> Starting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1
    )
    print("-> Evaluating model on validation set...")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    y_pred = np.argmax(best_model.predict(X_val), axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, target_names=class_names))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_history(history, os.path.join(OUTPUT_DIR, "training_history.png"))
    plot_confusion_matrix(y_val, y_pred, class_names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    # CORRECTED LINE
    print(f"\n-> Training complete. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
