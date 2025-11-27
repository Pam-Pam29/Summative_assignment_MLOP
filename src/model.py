"""
Model Creation and Training Module for PCOS Detection
Handles model building, training, and saving
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D, Dropout,
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
    Uses Functional API with alpha=0.5 for lighter model
    
    Args:
        img_height: Image height (default: 224)
        img_width: Image width (default: 224)
        learning_rate: Learning rate for optimizer (default: 1e-4)

    Returns:
        Compiled Keras model
    """
    # Create input layer
    inputs = Input(shape=(img_height, img_width, 3))
    
    # Load pre-trained MobileNetV2 base model with alpha=0.5 (lighter)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_height, img_width, 3),
        alpha=0.5  # Width multiplier (0.5 = half width, lighter model)
    )
    base_model.trainable = False
    
    # Pass inputs through base model
    x = base_model(inputs, training=False)
    
    # Add custom classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model using Functional API
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'AUC']
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
            # Format is determined by file extension (.keras = Keras format, .h5 = HDF5 format)
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
    # Note: workers and use_multiprocessing are not supported in newer TensorFlow versions
    # Memory optimization is handled via batch_size reduction instead
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    return history


def save_model_for_render(model, base_name='pcos_model', models_dir='models'):
    """
    Save model optimized for Render deployment
    Saves in multiple formats with Render-friendly naming
    
    Args:
        model: Trained Keras model to save
        base_name: Base name for model files (without extension)
        models_dir: Directory to save models in
    
    Returns:
        Dictionary with saved file paths
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    saved_formats = {}
    
    try:
        # 1. Save in .keras format (primary - most compatible)
        keras_path = models_path / f"{base_name}.keras"
        print(f"💾 Saving model in .keras format to {keras_path}...")
        model.save(str(keras_path))
        saved_formats['keras'] = str(keras_path)
        print(f"✅ Model saved successfully to {keras_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save .keras format: {str(e)}")
    
    try:
        # 2. Save weights separately (lightweight backup)
        weights_path = models_path / f"{base_name}.weights.h5"
        print(f"💾 Saving model weights to {weights_path}...")
        model.save_weights(str(weights_path))
        saved_formats['weights'] = str(weights_path)
        print(f"✅ Weights saved successfully to {weights_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save weights: {str(e)}")
    
    try:
        # 3. Save in .h5 format (legacy backup - may show warning but that's okay)
        h5_path = models_path / f"{base_name}.h5"
        print(f"💾 Saving model in .h5 format to {h5_path} (backup)...")
        model.save(str(h5_path), save_format='h5')
        saved_formats['h5'] = str(h5_path)
        print(f"✅ Model saved successfully to {h5_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save .h5 format: {str(e)}")
    
    if not saved_formats:
        raise Exception("Failed to save model in any format!")
    
    print(f"\n✅ Model saved successfully in {len(saved_formats)} format(s) for Render deployment")
    return saved_formats


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


def load_model_with_fallback(models_dir='models', base_name='pcos_model'):
    """
    Load model with automatic fallback - optimized for Render
    Tries multiple formats and paths automatically
    
    Args:
        models_dir: Directory containing model files
        base_name: Base name of model files (without extension)
    
    Returns:
        Loaded Keras model or None if not found
    """
    models_path = Path(models_dir)
    
    # Priority order: .h5 > .keras > weights (rebuild)
    # Changed to prioritize .h5 first since .keras files can hang on Render
    # Note: Files with spaces in names are handled but should be renamed
    load_attempts = [
        (models_path / f"{base_name}.h5", "H5 format"),
        (models_path / f"{base_name} .h5", "H5 format (with space)"),  # Handle spaces in filenames (should be renamed)
        (models_path / f"{base_name}.keras", "keras format"),  # Try .keras last since it can hang
    ]
    
    # Try loading full model files first
    for model_path, format_name in load_attempts:
        if model_path.exists():
            try:
                print(f"🔄 Attempting to load model from {format_name}: {model_path}...")
                print(f"   File size: {model_path.stat().st_size / (1024*1024):.2f} MB")
                
                # Try with compile=False first (more compatible)
                # For .keras files, try with safe_mode=False to avoid hanging
                try:
                    print(f"   Step 1/3: Loading model structure (compile=False)...")
                    import time
                    start_time = time.time()
                    
                    # For .keras files, try with safe_mode=False (newer TF versions)
                    if str(model_path).endswith('.keras'):
                        try:
                            model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
                        except TypeError:
                            # Older TensorFlow versions don't have safe_mode parameter
                            model = tf.keras.models.load_model(str(model_path), compile=False)
                    else:
                        model = tf.keras.models.load_model(str(model_path), compile=False)
                    
                    load_time = time.time() - start_time
                    print(f"✅ Model loaded from {format_name} (compile=False) in {load_time:.2f}s")
                except Exception as e1:
                    error_msg = str(e1)
                    print(f"   ⚠️ Loading with compile=False failed: {error_msg[:200]}")
                    # Try with compile=True
                    try:
                        print(f"   Step 1/3: Retrying with compile=True...")
                        start_time = time.time()
                        
                        # For .keras files, try with safe_mode=False
                        if str(model_path).endswith('.keras'):
                            try:
                                model = tf.keras.models.load_model(str(model_path), safe_mode=False)
                            except TypeError:
                                model = tf.keras.models.load_model(str(model_path))
                        else:
                            model = tf.keras.models.load_model(str(model_path))
                        
                        load_time = time.time() - start_time
                        print(f"✅ Model loaded from {format_name} (compile=True) in {load_time:.2f}s")
                    except Exception as e2:
                        error_msg = str(e2)
                        print(f"   ❌ Loading with compile=True also failed: {error_msg[:200]}")
                        import traceback
                        traceback.print_exc()
                        raise e2
                
                # Recompile with standard metrics
                print(f"   Step 2/3: Compiling model with metrics...")
                start_time = time.time()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc')
                    ]
                )
                compile_time = time.time() - start_time
                print(f"   Step 3/3: Model compilation complete in {compile_time:.2f}s")
                print(f"✅ Model compiled and ready! Total time: {load_time + compile_time:.2f}s")
                return model
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed to load {format_name}: {error_msg[:500]}")
                import traceback
                traceback.print_exc()
                continue
    
    # Fallback: Try loading from weights (requires rebuilding model)
    # Handle both with and without spaces in filenames
    weights_attempts = [
        models_path / f"{base_name}.weights.h5",
        models_path / f"{base_name}.weights .h5",  # Handle spaces
        models_path / f"{base_name}_weights.h5",
    ]
    
    for weights_path in weights_attempts:
        if weights_path.exists():
            try:
                print(f"🔄 Attempting to rebuild model from weights: {weights_path}...")
                print("⚠️ This will download MobileNetV2 base weights (~9MB) - may take a moment...")
                model = build_model(img_height=224, img_width=224, learning_rate=1e-4)
                model.load_weights(str(weights_path))
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc')
                    ]
                )
                print(f"✅ Model rebuilt and loaded from weights!")
                return model
            except Exception as e:
                print(f"❌ Failed to rebuild from weights {weights_path}: {str(e)}")
                continue
    
    print(f"❌ Could not load model from any format in {models_dir}")
    return None


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

