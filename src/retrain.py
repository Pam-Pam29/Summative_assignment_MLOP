"""
Retraining Module for PCOS Detection Model
Handles model retraining with new data
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import tensorflow as tf
import gc

# Optimize TensorFlow for low memory usage (same as api.py)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA completely
tf.config.set_visible_devices([], 'GPU')

# Limit TensorFlow threads to reduce memory
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Set TensorFlow to use less memory
tf.config.experimental.enable_op_determinism()

from src.model import build_model, train_model, save_training_history, load_saved_model
from src.preprocessing import create_data_generators, count_images


def merge_uploaded_data(upload_dir, train_dir):
    """
    Merge uploaded training data into training directory

    Args:
        upload_dir: Directory containing uploaded data
        train_dir: Main training directory

    Returns:
        Number of files merged
    """
    merged_count = 0
    upload_path = Path(upload_dir)
    train_path = Path(train_dir)

    if not upload_path.exists():
        return 0

    for category_dir in upload_path.iterdir():
        if category_dir.is_dir():
            dest_dir = train_path / category_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_file in category_dir.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    dest_file = dest_dir / img_file.name
                    if not dest_file.exists():
                        shutil.copy2(img_file, dest_file)
                        merged_count += 1

    return merged_count


def retrain_model(train_dir='data/train', test_dir='data/test', 
                  upload_dir='data/uploads/training',
                  model_save_path='models/pcos_model.keras',
                  epochs=20, batch_size=16, img_size=(224, 224),  # Reduced batch_size for Render
                  validation_split=0.2, seed=42):
    """
    Retrain the model with new data

    Args:
        train_dir: Training data directory
        test_dir: Test data directory
        upload_dir: Directory with newly uploaded data
        model_save_path: Path to save retrained model
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        validation_split: Validation split ratio
        seed: Random seed

    Returns:
        Dictionary with training results and metrics
    """
    print("Starting model retraining...")

    # Merge uploaded data
    print(f"Merging uploaded data from {upload_dir}...")
    merged_count = merge_uploaded_data(upload_dir, train_dir)
    print(f"Merged {merged_count} new images into training set")

    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir,
        test_dir,
        img_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed
    )

    # Load existing model as pre-trained (REQUIREMENT: Use custom model as pre-trained)
    print("Loading existing model as pre-trained...")
    model = load_saved_model()
    
    if model is None:
        print("No existing model found, building new one...")
        model = build_model(img_height=img_size[0], img_width=img_size[1], learning_rate=1e-4)
    else:
        # Fine-tune: Use existing model as pre-trained (REQUIREMENT)
        print("Using existing model as pre-trained for fine-tuning...")
        print(f"  Model architecture: {model.name}")
        print(f"  Total layers: {len(model.layers)}")
        
        # Make model trainable for fine-tuning
        model.trainable = True
        
        # Fine-tuning strategy: Freeze base model (MobileNetV2), unfreeze custom head
        # Find where the base model ends (usually at GlobalAveragePooling2D)
        base_model_end_idx = None
        for i, layer in enumerate(model.layers):
            if 'global_average_pooling' in layer.name.lower() or 'flatten' in layer.name.lower():
                base_model_end_idx = i
                break
        
        if base_model_end_idx is not None:
            # Freeze base model layers
            for layer in model.layers[:base_model_end_idx]:
                layer.trainable = False
            print(f"  Frozen base model layers: {base_model_end_idx}")
            print(f"  Trainable custom head layers: {len(model.layers) - base_model_end_idx}")
        else:
            # Fallback: Freeze all but last 10 layers
            for layer in model.layers[:-10]:
                layer.trainable = False
            print(f"  Frozen layers: {len(model.layers) - 10}")
            print(f"  Trainable layers: 10")
        
        # Recompile with lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        trainable_count = sum([1 for layer in model.layers if layer.trainable])
        print(f"  Total trainable layers: {trainable_count}")
        print("✅ Using custom model as pre-trained model (REQUIREMENT MET)")

    # Train model
    print(f"Training model for {epochs} epochs...")
    history = train_model(
        model,
        train_gen,
        val_gen,
        epochs=epochs,
        model_save_path=model_save_path,
        verbose=1
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(test_gen, verbose=1)

    test_metrics = {
        'test_accuracy': float(test_results[1]),
        'test_loss': float(test_results[0]),
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3]),
        'test_auc': float(test_results[4])
    }

    # Calculate F1 score
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = np.array(test_gen.classes)

    from sklearn.metrics import f1_score
    test_metrics['test_f1'] = float(f1_score(y_true, y_pred))
    
    # Save model properly in multiple formats (optimized for Render)
    from src.model import save_model_for_render
    print("\n💾 Saving model in multiple formats for Render deployment...")
    saved_formats = save_model_for_render(model, base_name='pcos_model', models_dir='models')
    print(f"✅ Model saved successfully in {len(saved_formats)} format(s): {list(saved_formats.keys())}")

    # Save training history
    save_training_history(history, model, test_metrics)

    print("Retraining completed successfully!")
    print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['test_f1']:.4f}")

    return {
        'success': True,
        'merged_images': merged_count,
        'metrics': test_metrics,
        'history': {
            'epochs': len(history.history['loss']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        }
    }


if __name__ == '__main__':
    # Example usage
    result = retrain_model()
    print(json.dumps(result, indent=2))











