"""
Prediction Module for PCOS Detection
Handles single image predictions and batch predictions
"""

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os


def preprocess_image(img_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction

    Args:
        img_path: Path to image file
        target_size: Target image size (default: 224x224)

    Returns:
        Preprocessed image array ready for prediction
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_single_image(model, img_path, class_names=None, threshold=0.5):
    """
    Make prediction on a single image

    Args:
        model: Trained Keras model
        img_path: Path to image file
        class_names: List of class names (optional)
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary with prediction results
    """
    if class_names is None:
        class_names = ['Not Infected', 'PCOS Infected']

    # Preprocess image
    img_array = preprocess_image(img_path)

    # Get prediction
    prediction_prob = model.predict(img_array, verbose=0)[0][0]

    # Determine class
    predicted_class = 1 if prediction_prob > threshold else 0

    # Calculate confidence
    confidence = prediction_prob if predicted_class == 1 else 1 - prediction_prob

    result = {
        'predicted_class': class_names[predicted_class],
        'confidence': float(confidence),
        'probability': float(prediction_prob),
        'class_index': int(predicted_class),
        'threshold': float(threshold)
    }

    return result


def predict_batch(model, image_paths, class_names=None, threshold=0.5):
    """
    Make predictions on multiple images

    Args:
        model: Trained Keras model
        image_paths: List of image file paths
        class_names: List of class names (optional)
        threshold: Classification threshold (default: 0.5)

    Returns:
        List of prediction result dictionaries
    """
    results = []
    for img_path in image_paths:
        try:
            result = predict_single_image(model, img_path, class_names, threshold)
            result['image_path'] = img_path
            results.append(result)
        except Exception as e:
            results.append({
                'image_path': img_path,
                'error': str(e)
            })
    return results


def validate_image_file(file_path):
    """
    Validate if a file is a valid image

    Args:
        file_path: Path to file

    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Try to open and verify image
        img = Image.open(file_path)
        img.verify()
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in valid_extensions:
            return False, f"Invalid file extension: {ext}"
        
        return True, None
    except Exception as e:
        return False, str(e)









