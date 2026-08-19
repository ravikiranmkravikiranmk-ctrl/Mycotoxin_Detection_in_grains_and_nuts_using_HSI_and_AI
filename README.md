# Real-Time Mycotoxin Detection using HSI and AI

This is a fully-fledged, ready-to-run project for a Major Project submission. It demonstrates the detection of five mycotoxin classes using Hyperspectral Imaging (HSI) and a 3D Convolutional Neural Network (CNN).

## Features
- **End-to-End Workflow**: From data loading to training, evaluation, and detection simulation.
- **Built-in Dataset**: Includes a small, synthetic dataset, so it's runnable out-of-the-box.
- **State-of-the-Art Model**: Uses a 3D CNN, ideal for HSI data.
- **Clear & Professional Structure**: Organized for easy understanding and extension.
- **Visualization**: Generates output plots and images for your project report.

## Quick Start Guide

1.  **Unzip the folder.**
2.  **Open a terminal** in the `Mycotoxin-HSI-AI-Project` directory.
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Train the model:**
    ```bash
    python train.py
    ```
6.  **Run a detection demo:**
    ```bash
    python detect.py
    ```
