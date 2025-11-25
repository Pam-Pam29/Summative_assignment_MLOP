"""
Data Preprocessing Module for PCOS Detection
Handles image preprocessing, data augmentation, and data generators
"""

import numpy as np
from pathlib import Path
from PIL import Image
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pandas as pd


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


def create_data_generators(train_dir, test_dir, img_size=(224, 224), batch_size=64, validation_split=0.2, seed=42):
    """
    Create data generators for training, validation, and testing

    Args:
        train_dir: Directory containing training data
        test_dir: Directory containing test data
        img_size: Target image size
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_generator, validation_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=seed
    )

    validation_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=seed
    )

    test_generator = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def count_images(directory):
    """
    Count images in each class directory

    Args:
        directory: Directory to count images in

    Returns:
        Dictionary with counts per class
    """
    counts = {}
    total = 0

    if Path(directory).exists():
        for class_dir in Path(directory).iterdir():
            if class_dir.is_dir():
                image_files = list(class_dir.glob('*.jpg')) + \
                             list(class_dir.glob('*.png')) + \
                             list(class_dir.glob('*.jpeg'))
                counts[class_dir.name] = len(image_files)
                total += len(image_files)
        counts['total'] = total

    return counts


def find_problematic_images(directory):
    """
    Find corrupted or invalid image files

    Args:
        directory: Directory to scan

    Returns:
        List of tuples (file_path, error_message)
    """
    problematic_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except Exception as e:
                problematic_files.append((file_path, str(e)))
    return problematic_files


def create_image_dataframe(base_path, categories):
    """
    Create a DataFrame with image paths and labels

    Args:
        base_path: Base path to dataset
        categories: List of category names

    Returns:
        DataFrame with image_path and label columns
    """
    image_paths = []
    labels = []

    for category in categories:
        for subset in ['test', 'train']:
            subset_path = os.path.join(base_path, subset, category)
            if os.path.exists(subset_path):
                for image_name in os.listdir(subset_path):
                    image_path = os.path.join(subset_path, image_name)
                    image_paths.append(image_path)
                    labels.append(category)

    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })

    return df









