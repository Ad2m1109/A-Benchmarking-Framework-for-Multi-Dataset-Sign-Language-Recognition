import sys
import os

# Add the parent directory to the Python path to allow importing modules from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Import the DatasetRegistry and ensure ArdaMaviDataset is registered
from datasets import DatasetRegistry
from ardamavi_dataset import ArdaMaviDataset # This import registers the dataset

# --- Hardcoded Dataset Configuration for ArdaMaviDataset ---
SELECTED_DATASET = "ArdaMaviDataset"
DATA_DIR = "archive" # Relative to the current script's directory (dataset number 3/)
MODEL_FILENAME = "sign_language_model_ardamavi.h5"
MODEL_SAVE_DIR = "../trained_models" # Relative to the current script's directory (dataset number 3/)

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
FULL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

# --- Training Configuration ---
EPOCHS = 10
BATCH_SIZE = 64

# --- Data Loading and Preprocessing ---
print(f"Loading dataset: {SELECTED_DATASET} from {DATA_DIR}...")
dataset_instance = DatasetRegistry.get_dataset(SELECTED_DATASET)
# Pass the data directory to the dataset instance
dataset_instance.data_dir = DATA_DIR
dataset_instance.load_data()

X_train = dataset_instance.X_train
y_train = dataset_instance.y_train
X_test = dataset_instance.X_test
y_test = dataset_instance.y_test
label_mapping = dataset_instance.label_mapping
num_classes = dataset_instance.get_dataset_info()['num_classes']

print("Data preprocessing complete.")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of classes: {num_classes}")

# --- Define the CNN model ---
print("Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# --- Compile the model ---
print("Compiling model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train the model ---
print("Training model...")
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test))
print("Model training complete.")

# --- Evaluate the model ---
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- Save the model ---
print(f"Saving model to {FULL_MODEL_PATH}...")
model.save(FULL_MODEL_PATH)
print("Model saved.")
