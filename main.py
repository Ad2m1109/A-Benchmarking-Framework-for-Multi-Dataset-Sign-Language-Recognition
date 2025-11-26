import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from inference import run_realtime_prediction
# Import dataset modules to ensure they are registered
import sign_mnist_dataset
import npy_dataset
import ardamavi_dataset
import indian_sign_language_dataset

def main():
    while True:
        print("\n--- Sign Language Recognition System ---")
        print("1. Test Sign MNIST Model (Alphabet A-Z)")
        print("2. Test Sign Digits Model (0-9)")
        print("3. Test ArdaMavi Digits Model (0-9)")
        print("4. Test Indian Sign Language Model (Alphabet)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            model_path = os.path.join("trained_models", "sign_language_model_sign_mnist.h5")
            data_dir = os.path.join("dataset number 1", "archive")
            dataset_name = "SignMnistDataset"
            run_realtime_prediction(model_path, dataset_name, data_dir)
        elif choice == '2':
            model_path = os.path.join("trained_models", "sign_language_model_digits.h5")
            data_dir = os.path.join("dataset number 2", "archive", "Sign-language-digits-dataset")
            dataset_name = "NpyDataset"
            run_realtime_prediction(model_path, dataset_name, data_dir)
        elif choice == '3':
            model_path = os.path.join("trained_models", "sign_language_model_ardamavi.h5")
            data_dir = os.path.join("dataset number 3", "archive")
            dataset_name = "ArdaMaviDataset"
            run_realtime_prediction(model_path, dataset_name, data_dir)
        elif choice == '4':
            model_path = os.path.join("trained_models", "sign_language_model_indian.h5")
            data_dir = os.path.join("dataset number 4", "archive", "ISL_Dataset")
            dataset_name = "IndianSignLanguageDataset"
            run_realtime_prediction(model_path, dataset_name, data_dir)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
