"""
Model Creation and Training Module for PCOS Detection
Handles model building, training, and saving
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout,
                                     BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                       ReduceLROnPlateau)
from tensorflow.keras.regularizers import l2
from pathlib import Path
import json
from datetime import datetime
import numpy as np


def build_model(img_height=224, img_width=224, learning_rate=1e-4):
    """
    Build the PCOS detection model using MobileNetV2 transfer learning
    MobileNetV2 is much lighter than ResNet50 (~3.4M params vs ~25M)
    Perfect for deployment on resource-constrained environments

    Args:
        img_height: Image height
        img_width: Image width
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 base model (much lighter than ResNet50!)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3),
        alpha=1.0  # Width multiplier (1.0 = full width, smaller = less params)
    )
    base_model.trainable = False

    # Build custom classification head with stronger regularization to prevent overfitting
    # Reduced capacity + higher dropout + stronger L2 regularization
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Add dropout after pooling
        Dense(128, kernel_regularizer=l2(0.01)),  # Increased L2 from 0.001 to 0.01
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),  # Increased from 0.3 to 0.5
        Dense(64, kernel_regularizer=l2(0.01)),  # Reduced from 128 to 64, increased L2
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),  # Increased from 0.2 to 0.4
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn')
        ]
    )

    return model


def train_model(model, train_generator, validation_generator, 
                epochs=20, model_save_path='models/pcos_model.keras', 
                patience=5, verbose=1):
    """
    Train the model with callbacks

    Args:
        model: Keras model to train
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Maximum number of epochs
        model_save_path: Path to save the best model (default: .keras format)
        patience: Early stopping patience
        verbose: Verbosity level

    Returns:
        Training history
    """
    # Ensure models directory exists
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert .h5 to .keras if needed (for backward compatibility)
    if model_save_path.endswith('.h5'):
        model_save_path = model_save_path.replace('.h5', '.keras')
        if verbose:
            print(f"⚠️ Note: Changed save path to .keras format: {model_save_path}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=verbose
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_format='keras',  # Use native Keras format (no HDF5 warnings!)
            verbose=verbose
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=verbose
        )
    ]

    # Train model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    return history


def save_model_properly(model, base_path='models/pcos_model', save_weights=True):
    """
    Save model in multiple formats for maximum compatibility
    This ensures the model can be loaded reliably on different systems
    
    Args:
        model: Trained Keras model to save
        base_path: Base path for saving (without extension)
        save_weights: Whether to also save weights separately
    
    Returns:
        List of saved file paths
    """
    saved_paths = []
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Save in .keras format (newer, most compatible with TensorFlow 2.x+)
        # This is the recommended format - no warnings!
        keras_path = f"{base_path}.keras"
        print(f"Saving model in .keras format to {keras_path}...")
        model.save(keras_path)  # No save_format needed - .keras extension is detected automatically
        saved_paths.append(keras_path)
        print(f"✅ Model saved successfully to {keras_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save .keras format: {str(e)}")
    
    try:
        # 2. Save in .h5 format (legacy support - only if needed)
        # Note: This will show a warning, but we keep it for backward compatibility
        h5_path = f"{base_path}.h5"
        print(f"Saving model in .h5 format to {h5_path} (legacy backup)...")
        model.save(h5_path, save_format='h5')
        saved_paths.append(h5_path)
        print(f"✅ Model saved successfully to {h5_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save .h5 format: {str(e)}")
    
    if save_weights:
        try:
            # 3. Save weights separately (lightweight, can rebuild model)
            weights_path = f"{base_path}.weights.h5"
            print(f"Saving model weights to {weights_path}...")
            model.save_weights(weights_path)
            saved_paths.append(weights_path)
            print(f"✅ Weights saved successfully to {weights_path}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save weights: {str(e)}")
    
    if not saved_paths:
        raise Exception("Failed to save model in any format!")
    
    print(f"✅ Model saved successfully in {len(saved_paths)} format(s)")
    return saved_paths


def save_training_history(history, model, test_metrics, save_path='models/training_history.json'):
    """
    Save training history and metadata

    Args:
        history: Training history object
        model: Trained model
        test_metrics: Dictionary of test set metrics
        save_path: Path to save history JSON
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'precision': [float(x) for x in history.history['precision']],
        'val_precision': [float(x) for x in history.history.get('val_precision', [])],
        'recall': [float(x) for x in history.history['recall']],
        'val_recall': [float(x) for x in history.history.get('val_recall', [])],
        'auc': [float(x) for x in history.history['auc']],
        'val_auc': [float(x) for x in history.history.get('val_auc', [])]
    }

    training_metadata = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'training',
        'epochs_trained': len(history.history['loss']),
        'best_epoch': int(np.argmax(history.history['val_accuracy']) + 1) if 'val_accuracy' in history.history else 0,
        'history': history_dict,
        'final_metrics': test_metrics,
        'model_config': {
            'base_model': 'MobileNetV2',
            'pretrained': 'ImageNet',
            'input_shape': [224, 224, 3],
            'total_params': int(total_params),
            'trainable_params': int(trainable_params)
        }
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(training_metadata, f, indent=4)


def load_saved_model(model_path=None):
    """
    Load a saved model (supports multiple formats and paths)

    Args:
        model_path: Path to saved model file (optional, will try multiple paths if None)

    Returns:
        Loaded Keras model or None if not found
    """
    # Try multiple model paths if not specified
    if model_path is None:
        model_paths = [
            'models/pcos_model.keras',
            'models/pcos_model.h5',
            'models/pcos_model'
        ]
    else:
        model_paths = [model_path]
    
    for path in model_paths:
        try:
            if not Path(path).exists() and not Path(path).is_dir():
                continue
            
            # Try loading with tf.keras (more compatible)
            model = tf.keras.models.load_model(path, compile=False)
            print(f"✅ Model loaded from {path}")
            return model
        except Exception as e1:
            try:
                # Try standard load_model
                model = load_model(path)
                print(f"✅ Model loaded from {path}")
                return model
            except Exception as e2:
                # Try with custom_objects for MobileNetV2
                try:
                    from tensorflow.keras.applications import MobileNetV2
                    model = tf.keras.models.load_model(
                        path,
                        custom_objects={'MobileNetV2': MobileNetV2},
                        compile=False
                    )
                    print(f"✅ Model loaded from {path} (with MobileNetV2 custom_objects)")
                    return model
                except Exception as e3:
                    continue
    
    print("❌ Could not load model from any path")
    return None

