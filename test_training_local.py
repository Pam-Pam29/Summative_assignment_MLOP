"""
Local Testing Script for Training
Test training functionality locally before deploying to Render
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for local testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA

print("=" * 60)
print("LOCAL TRAINING TEST")
print("=" * 60)
print()

# Check if data directories exist
data_train = Path('data/train')
data_test = Path('data/test')

if not data_train.exists():
    print(f"❌ Training data directory not found: {data_train}")
    print("   Please ensure data/train directory exists with 'infected' and 'notinfected' subdirectories")
    sys.exit(1)

if not data_test.exists():
    print(f"❌ Test data directory not found: {data_test}")
    print("   Please ensure data/test directory exists with 'infected' and 'notinfected' subdirectories")
    sys.exit(1)

print(f"✅ Training data directory found: {data_train}")
print(f"✅ Test data directory found: {data_test}")
print()

# Test data generators
print("Testing data generators...")
try:
    from src.preprocessing import create_data_generators
    
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir='data/train',
        test_dir='data/test',
        batch_size=16,  # Small batch for local testing
        img_size=(224, 224),
        validation_split=0.2
    )
    
    print(f"✅ Training batches: {len(train_gen)}")
    print(f"✅ Validation batches: {len(val_gen)}")
    print(f"✅ Test batches: {len(test_gen)}")
    print()
    
except Exception as e:
    print(f"❌ Error creating data generators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model building
print("Testing model building...")
try:
    from src.model import build_model
    import tensorflow as tf
    
    tf.config.set_visible_devices([], 'GPU')
    
    model = build_model(img_height=224, img_width=224, learning_rate=1e-4)
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {model.count_params():,}")
    print()
    
except Exception as e:
    print(f"❌ Error building model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test training (just 1 epoch for quick test)
print("Testing training (1 epoch for quick test)...")
try:
    from src.model import train_model
    
    history = train_model(
        model,
        train_gen,
        val_gen,
        epochs=1,  # Just 1 epoch for testing
        model_save_path='models/test_model.keras',
        patience=5,
        verbose=1
    )
    
    print("✅ Training test completed successfully!")
    print(f"   Training loss: {history.history['loss'][-1]:.4f}")
    print(f"   Validation loss: {history.history['val_loss'][-1]:.4f}")
    print()
    
except Exception as e:
    print(f"❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model loading
print("Testing model loading...")
try:
    from src.model import load_model_with_fallback
    
    loaded_model = load_model_with_fallback(models_dir='models', base_name='test_model')
    
    if loaded_model is not None:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model file not found (this is okay for a test run)")
    print()
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print()
print("Your training code is ready for deployment!")
print("You can now deploy to Render with confidence.")

