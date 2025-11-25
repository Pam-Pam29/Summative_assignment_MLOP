"""
Retrain model with MobileNetV2
This script retrains the PCOS detection model using the lighter MobileNetV2 architecture
"""
import os
import sys
from pathlib import Path
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import build_model, train_model, save_training_history
from src.preprocessing import create_data_generators
from tensorflow.keras.models import load_model
import numpy as np

def main():
    """Retrain model with MobileNetV2"""
    print("=" * 60)
    print("Retraining PCOS Detection Model with MobileNetV2")
    print("=" * 60)
    
    # Data directories
    train_dir = Path('data/train')
    test_dir = Path('data/test')
    model_save_path = 'models/pcos_model.h5'
    
    # Check if data directories exist
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        print("Please ensure your training data is in the correct location.")
        return
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print("Please ensure your test data is in the correct location.")
        return
    
    print(f"\n📁 Data directories:")
    print(f"   Train: {train_dir}")
    print(f"   Test: {test_dir}")
    
    # Create data generators
    print("\n📊 Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir,
        test_dir,
        img_size=(224, 224),
        batch_size=32,  # Smaller batch for MobileNetV2
        validation_split=0.2,
        seed=42
    )
    
    print(f"   Training samples: {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Test samples: {test_gen.samples}")
    
    # Build MobileNetV2 model
    print("\n🏗️  Building MobileNetV2 model...")
    model = build_model(img_height=224, img_width=224, learning_rate=1e-4)
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]) if hasattr(tf, 'keras') else model.count_params()
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024*1024):.1f}MB (estimated)")
    
    # Train model
    print("\n🚀 Starting training...")
    print("   This may take 1-2 hours depending on your hardware.")
    print("   MobileNetV2 trains faster than ResNet50!")
    
    history = train_model(
        model,
        train_gen,
        val_gen,
        epochs=20,
        model_save_path=model_save_path,
        patience=5,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_results = model.evaluate(test_gen, verbose=1)
    
    test_metrics = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_precision': float(test_results[2]) if len(test_results) > 2 else None,
        'test_recall': float(test_results[3]) if len(test_results) > 3 else None,
        'test_auc': float(test_results[4]) if len(test_results) > 4 else None,
    }
    
    print("\n✅ Training completed!")
    print(f"\n📊 Test Results:")
    for key, value in test_metrics.items():
        if value is not None:
            print(f"   {key}: {value:.4f}")
    
    # Save training history
    print("\n💾 Saving training history...")
    save_training_history(history, model, test_metrics)
    
    # Extract weights
    print("\n💾 Extracting weights...")
    weights_path = 'models/pcos_model.weights.h5'
    model.save_weights(weights_path)
    
    # Check file sizes
    model_size = os.path.getsize(model_save_path) / (1024*1024)
    weights_size = os.path.getsize(weights_path) / (1024*1024)
    
    print(f"\n📦 Model files:")
    print(f"   Full model: {model_size:.1f}MB")
    print(f"   Weights: {weights_size:.1f}MB")
    print(f"\n🎉 Model saved to: {model_save_path}")
    print(f"🎉 Weights saved to: {weights_path}")
    print(f"\n💡 Your API will automatically use the weights file for lower memory usage!")

if __name__ == '__main__':
    main()

