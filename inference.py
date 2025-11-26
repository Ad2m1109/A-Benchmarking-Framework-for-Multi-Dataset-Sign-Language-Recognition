import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from datasets import DatasetRegistry

def detect_hand_region(frame):
    """
    Detect hand region using skin color detection.
    Returns bounding box coordinates (x, y, w, h) or None if no hand detected.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour (assumed to be the hand)
    max_contour = max(contours, key=cv2.contourArea)
    
    # Filter out small contours
    if cv2.contourArea(max_contour) < 5000:
        return None
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(max_contour)
    
    return (x, y, w, h)

def run_realtime_prediction(model_path, dataset_name, data_dir):
    """
    Runs real-time prediction using the specified model and dataset.
    Uses OpenCV skin detection to focus on the hand region.

    Args:
        model_path (str): Path to the trained .h5 model file.
        dataset_name (str): Name of the registered dataset class.
        data_dir (str): Path to the dataset directory (for loading metadata/labels).
    """

    # Load the trained model
    print(f"Loading trained model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = load_model(model_path)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get the dataset instance to retrieve label mapping
    print(f"Retrieving label mapping for {dataset_name}...")
    try:
        dataset_instance = DatasetRegistry.get_dataset(dataset_name)
        dataset_instance.data_dir = data_dir
        dataset_instance.load_data() 
        label_mapping = dataset_instance.label_mapping
        print("Label mapping retrieved.")
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    print("Camera initialized.")

    print("Starting real-time prediction. Press 'q' to quit.")
    print("Show your hand to the camera for detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        predicted_character = "No hand detected"
        confidence = 0.0
        
        # Detect hand region
        hand_bbox = detect_hand_region(frame)
        
        if hand_bbox is not None:
            x, y, bbox_w, bbox_h = hand_bbox
            
            # Add padding
            padding = 30
            x = max(0, x - padding)
            y = max(0, y - padding)
            bbox_w = min(w - x, bbox_w + 2 * padding)
            bbox_h = min(h - y, bbox_h + 2 * padding)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + bbox_w, y + bbox_h), (0, 255, 0), 2)
            
            # Extract hand region
            hand_roi = frame[y:y+bbox_h, x:x+bbox_w]
            
            if hand_roi.size > 0:
                try:
                    # Preprocess hand ROI
                    gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                    resized_hand = cv2.resize(gray_hand, (28, 28), interpolation=cv2.INTER_AREA)
                    normalized_hand = resized_hand / 255.0
                    model_input = normalized_hand.reshape(1, 28, 28, 1)

                    # Make prediction
                    predictions = model.predict(model_input, verbose=0)
                    predicted_class_index = np.argmax(predictions)
                    confidence = predictions[0][predicted_class_index]
                    
                    # Map predicted index to character
                    predicted_character = label_mapping.get(predicted_class_index, "Unknown")
                except Exception as e:
                    predicted_character = f"Error: {str(e)[:20]}"

        # Display the prediction with confidence
        if confidence > 0:
            display_text = f"Prediction: {predicted_character} ({confidence:.2f})"
        else:
            display_text = f"Prediction: {predicted_character}"
            
        cv2.putText(frame, display_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (10, h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")
