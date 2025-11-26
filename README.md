# Sign Language Recognition Model

This project aims to develop a sign language recognition model capable of identifying alphabet characters (A-Z) and numbers (0-9) from images. The ultimate goal is to integrate this model with a live camera feed for real-time sign language interpretation.

## Project Structure

- `datasets.py`: Defines the `BenchmarkDataset` abstract base class and the `DatasetRegistry` for managing multiple datasets.
- `sign_mnist_dataset.py`: Implements the `SignMnistDataset` by inheriting from `BenchmarkDataset` and registers it with the `DatasetRegistry`.
- `npy_dataset.py`: Implements the `NpyDataset` by inheriting from `BenchmarkDataset` and registers it with the `DatasetRegistry` for datasets stored in `.npy` format.
- `README.md`: This file.
- `trained_models/`: Contains all trained models (`sign_language_model_sign_mnist.h5`, `sign_language_model_digits.h5`, `sign_language_model_ardamavi.h5`).
- `dataset number 1/`: Contains all code and data for the Sign MNIST dataset.
    - `archive/`: Contains the raw dataset files (`sign_mnist_train.csv`, `sign_mnist_test.csv`).
    - `train.py`: Script to train the model for this dataset.
    - `deploy.py`: Script to deploy the trained model for real-time prediction.
    - `summary.md`: A detailed outline of the model training steps for this dataset.
- `dataset number 2/`: Contains all code and data for the NPY-based sign language digits dataset.
    - `archive/Sign-language-digits-dataset/`: Contains the raw dataset files (`X.npy`, `Y.npy`).
    - `train.py`: Script to train the model for this dataset.
    - `deploy.py`: Script to deploy the trained model for real-time prediction.
    - `summary.md`: A detailed outline of the model training steps for this dataset.
- `dataset number 3/`: Contains all code and data for the ArdaMavi sign language digits dataset.
    - `archive/`: Contains the raw dataset files (`X.npy`, `Y.npy`).
    - `train.py`: Script to train the model for this dataset.
    - `deploy.py`: Script to deploy the trained model for real-time prediction.
    - `summary.md`: A detailed outline of the model training steps for this dataset.

## Dataset Management

This project uses a `DatasetRegistry` to manage various `BenchmarkDataset` implementations. The core dataset logic (`datasets.py`, `sign_mnist_dataset.py`, `npy_dataset.py`) remains in the root directory to be shared by all dataset-specific scripts.

### Current Datasets

- **SignMnistDataset**: This dataset is based on the Sign MNIST dataset, containing grayscale images of hand gestures representing alphabet characters (A-Z, excluding J and Z due to motion). Its code and data are located in `dataset number 1/`.
- **NpyDataset**: This dataset is designed for sign language digits, loading data from `.npy` files. Its code and data are located in `dataset number 2/`.
- **ArdaMaviDataset**: This dataset is also for sign language digits (0-9), sourced from ArdaMavi. Its code and data are located in `dataset number 3/`.

### Adding New Datasets

To add a new dataset:
1.  Create a new directory for your dataset (e.g., `my_new_dataset/`).
2.  Place your raw data files within `my_new_dataset/archive/`.
3.  Copy `train.py`, `deploy.py`, and `summary.md` from an existing dataset folder (e.g., `dataset number 1/`) into `my_new_dataset/`.
4.  Create a `trained_models/` directory inside `my_new_dataset/`.
5.  Create a new Python file (e.g., `my_new_dataset_impl.py`) that defines a class inheriting from `BenchmarkDataset` (from `datasets.py`). This file should be placed in the root directory.
6.  Implement the `load_data()` method within your new class to handle loading, preprocessing, reshaping, normalizing, and one-hot encoding of your dataset's data. Ensure `X_train`, `y_train`, `X_test`, `y_test`, and `label_mapping` attributes are populated. The `data_dir` attribute of your dataset instance will be set by the `train.py` and `deploy.py` scripts within `my_new_dataset/`.
7.  Decorate your new dataset class with `@DatasetRegistry.register_dataset` to make it available to the system.
8.  Ensure your new dataset implementation file (e.g., `my_new_dataset_impl.py`) is imported in the `train.py` and `deploy.py` scripts within `my_new_dataset/` (and any other relevant scripts) so it gets registered.
9.  Modify the `train.py` and `deploy.py` scripts within `my_new_dataset/` to hardcode the `SELECTED_DATASET`, `DATA_DIR`, `MODEL_FILENAME`, and `MODEL_SAVE_DIR` variables to be specific to your new dataset.

## Getting Started

To get started with this project, follow the steps below.

### Prerequisites

*   Python 3.x
*   Jupyter Notebook or a Python IDE
*   Required Python libraries (will be installed in the setup)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd sign-language-model
    ```
    (Note: Replace `<repository_url>` with the actual URL if this project is hosted on Git.)

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pandas numpy tensorflow keras scikit-learn matplotlib opencv-python
    ```

## Usage

### Training the Model

To train the model for a specific dataset (e.g., Sign MNIST):
1.  Navigate to the dataset's directory:
    ```bash
    cd "dataset number 1"
    ```
2.  Run the training script:
    ```bash
    python3 train.py
    ```
    ```
    This will train the model and save it to `trained_models/sign_language_model_sign_mnist.h5`.

To train the model for the NPY-based digits dataset:
1.  Navigate to the dataset's directory:
    ```bash
    cd "dataset number 2"
    ```
2.  Run the training script:
    ```bash
    python3 train.py
    ```
    ```
    This will train the model and save it to `trained_models/sign_language_model_digits.h5`.

### Real-time Prediction

To use a trained model for real-time predictions with your camera (e.g., for Sign MNIST):
1.  Navigate to the dataset's directory:
    ```bash
    cd "dataset number 1"
    ```
2.  Run the deployment script:
    ```bash
    python3 deploy.py
    ```
    A window will open displaying your camera feed with real-time predictions. Press 'q' to quit.

To use a trained model for real-time predictions with your camera (e.g., for NPY-based digits):
1.  Navigate to the dataset's directory:
    ```bash
    cd "dataset number 2"
    ```
2.  Run the deployment script:
    ```bash
    python3 deploy.py
    ```
    A window will open displaying your camera feed with real-time predictions. Press 'q' to quit.

    A window will open displaying your camera feed with real-time predictions. Press 'q' to quit.

To use a trained model for real-time predictions with your camera (e.g., for ArdaMavi digits):
1.  Navigate to the dataset's directory:
    ```bash
    cd "dataset number 3"
    ```
2.  Run the deployment script:
    ```bash
    python3 deploy.py
    ```
    A window will open displaying your camera feed with real-time predictions. Press 'q' to quit.

### Main Interface

You can also use the centralized interface to select any model:
```bash
python3 main.py
```

## Training Steps (High-Level)

The training process involves the following key stages:

1.  **Data Loading and Preprocessing:** Handled by the selected `BenchmarkDataset` implementation (e.g., `SignMnistDataset` or `NpyDataset`).
2.  **Model Architecture:** A Convolutional Neural Network (CNN) is defined in the respective `train.py` script.
3.  **Model Compilation:** Configured with an optimizer, loss function, and metrics.
4.  **Model Training:** The CNN is trained on the preprocessed data.
5.  **Model Evaluation:** Model performance is assessed on the test set.
6.  **Deployment with Camera Integration:** The trained model is integrated with a live camera feed for real-time predictions.
