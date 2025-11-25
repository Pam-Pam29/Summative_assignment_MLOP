"""
Extract model weights from full model to reduce size
This creates a much smaller weights file (~20-30MB vs ~93MB)
"""
import os
from tensorflow import keras

def extract_weights():
    """Extract weights from full model"""
    model_path = 'models/pcos_model.h5'
    weights_path = 'models/pcos_model.weights.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available files:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                print(f"  - {f}")
        return
    
    print(f"📦 Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print(f"💾 Saving weights to: {weights_path}")
    model.save_weights(weights_path)
    
    # Check sizes
    model_size = os.path.getsize(model_path) / (1024*1024)
    weights_size = os.path.getsize(weights_path) / (1024*1024)
    
    print(f"\n✅ Success!")
    print(f"   Model size: {model_size:.1f}MB")
    print(f"   Weights size: {weights_size:.1f}MB")
    print(f"   Saved: {model_size - weights_size:.1f}MB ({((model_size - weights_size) / model_size * 100):.1f}% reduction)")
    print(f"\n💡 Your code already tries to load weights first!")
    print(f"   Commit and push {weights_path} to use it.")

if __name__ == '__main__':
    extract_weights()


