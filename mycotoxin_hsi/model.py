import tensorflow as tf
from tensorflow.keras import layers, models

def build_mycotoxin_3d_cnn(input_shape, num_classes):
    """Builds a 3D CNN model for HSI classification."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(16, kernel_size=(3, 3, 7), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Conv3D(32, kernel_size=(3, 3, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name="Mycotoxin3D_CNN")
    return model
