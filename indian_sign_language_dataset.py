import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from datasets import BenchmarkDataset, DatasetRegistry

@DatasetRegistry.register_dataset
class IndianSignLanguageDataset(BenchmarkDataset):
    def __init__(self, name="IndianSignLanguageDataset"):
        super().__init__(name)
        self.data_dir = None  # Will be set by the training/deployment script

    def load_data(self):
        if self.data_dir is None:
            raise ValueError("data_dir must be set before calling load_data()")

        print(f"Loading {self.name} from {self.data_dir}...")
        
        # The dataset has folders A-Z (excluding H, J, Y based on what we saw)
        # Each folder contains images for that letter
        images = []
        labels = []
        
        # Get all class folders
        class_folders = sorted([f for f in os.listdir(self.data_dir) 
                               if os.path.isdir(os.path.join(self.data_dir, f))])
        
        print(f"Found {len(class_folders)} classes: {class_folders}")
        
        for class_name in class_folders:
            class_path = os.path.join(self.data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Loading {len(image_files)} images from class {class_name}...")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Resize to 28x28
                resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                
                images.append(resized)
                labels.append(class_name)
        
        print(f"Loaded {len(images)} images total")
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)
        
        # Normalize pixel values
        X = X / 255.0
        
        # Reshape to add channel dimension
        X = X.reshape(-1, 28, 28, 1)
        
        # Split into train/test
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # One-hot encode labels
        self.label_binarizer = LabelBinarizer()
        self.y_train = self.label_binarizer.fit_transform(y_train_raw)
        self.y_test = self.label_binarizer.transform(y_test_raw)
        
        self.X_train = X_train
        self.X_test = X_test
        
        # Create label mapping
        self.label_mapping = {i: label for i, label in enumerate(self.label_binarizer.classes_)}
        
        print(f"{self.name} Data preprocessing complete.")
        print(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        print(f"Classes: {self.label_mapping}")

    def get_dataset_info(self):
        num_classes = len(self.label_mapping) if self.label_mapping else 0
        return {"num_classes": num_classes}
