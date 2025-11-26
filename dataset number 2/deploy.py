import sys
import os

# Add the parent directory to the Python path to allow importing modules from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import run_realtime_prediction
# Import the dataset to ensure registration
import npy_dataset

# --- Hardcoded Dataset Configuration for NpyDataset ---
SELECTED_DATASET = "NpyDataset"
DATA_DIR = "archive/Sign-language-digits-dataset" # Relative to the current script's directory (dataset number 2/)
MODEL_FILENAME = "sign_language_model_digits.h5"
MODEL_SAVE_DIR = "../trained_models" # Relative to the current script's directory (dataset number 2/)

# Construct absolute paths for the inference function
current_dir = os.path.dirname(os.path.abspath(__file__))
full_model_path = os.path.join(current_dir, MODEL_SAVE_DIR, MODEL_FILENAME)
full_data_dir = os.path.join(current_dir, DATA_DIR)

if __name__ == "__main__":
    run_realtime_prediction(full_model_path, SELECTED_DATASET, full_data_dir)