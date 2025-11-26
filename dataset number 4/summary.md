# Dataset 4: Indian Sign Language

## Overview
This dataset contains images of Indian Sign Language alphabet gestures (A-Z, excluding some letters). It is sourced from Kaggle.

## Structure
-   `archive/ISL_Dataset/`: Contains subdirectories for each letter class with images.
-   `train.py`: Script to train the CNN model.
-   `deploy.py`: Script to run real-time inference using the trained model.
-   `summary.md`: This file.

## Training
The model is a Convolutional Neural Network (CNN) trained on images resized to 28x28 grayscale.
-   **Input Shape**: (28, 28, 1)
-   **Classes**: 23 (Letters A-Z, excluding H, J, Y)
-   **Optimizer**: Adam
-   **Loss**: Categorical Crossentropy

## Usage
To use this dataset:
1.  Train the model: `python3 train.py`
2.  Run inference: `python3 deploy.py`
    -   Or use the main interface: `python3 ../main.py` and select Option 4.
