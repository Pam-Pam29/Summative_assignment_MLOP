"""
Script to create training_history.json from notebook training data
Run this after training your model in the notebook
"""
import json
from pathlib import Path
from datetime import datetime

# Example training history based on your MobileNetV2 training
# Replace these values with your actual training history from the notebook

# Based on your training logs showing 100% validation accuracy at epoch 5
training_history = {
    "timestamp": datetime.now().isoformat(),
    "mode": "training",
    "epochs_trained": 20,  # Update with your actual number of epochs
    "best_epoch": 5,  # Epoch where you got 100% validation accuracy
    "history": {
        "accuracy": [
            0.6171, 0.9592, 0.9882, 0.9948, 0.9885, 0.9947, 0.9980, 1.0000, 1.0000, 0.9989,
            0.9976, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ],
        "val_accuracy": [
            0.8307, 0.9505, 0.9844, 0.9974, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ],
        "loss": [
            1.2645, 0.8330, 0.7205, 0.6688, 0.6607, 0.6379, 0.6191, 0.6103, 0.5978, 0.5938,
            0.5806, 0.5728, 0.5584, 0.5508, 0.5440, 0.5380, 0.5320, 0.5260, 0.5200, 0.5140
        ],
        "val_loss": [
            1.0382, 0.8633, 0.7778, 0.7191, 0.6665, 0.6351, 0.6138, 0.6007, 0.5878, 0.5752,
            0.5649, 0.5546, 0.5480, 0.5397, 0.5334, 0.5280, 0.5230, 0.5180, 0.5130, 0.5080
        ],
        "precision": [
            0.7184, 0.9714, 0.9934, 0.9951, 0.9859, 0.9914, 0.9973, 1.0000, 1.0000, 1.0000,
            0.9960, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ],
        "val_precision": [
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ],
        "recall": [
            0.6072, 0.9593, 0.9869, 0.9959, 0.9941, 0.9992, 0.9992, 1.0000, 1.0000, 0.9982,
            1.0000, 0.9996, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ],
        "val_recall": [
            0.7149, 0.9167, 0.9737, 0.9956, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ],
        "auc": [
            0.6659, 0.9926, 0.9992, 0.9998, 0.9997, 1.0000, 0.9998, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ],
        "val_auc": [
            0.9930, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
        ]
    },
    "final_metrics": {
        "test_accuracy": 1.0,  # 100% - from your training
        "test_precision": 1.0,  # 100%
        "test_recall": 1.0,  # 100%
        "test_f1": 1.0,  # 100%
        "test_auc": 1.0,  # 100%
        "test_loss": 0.5  # Example value
    },
    "model_config": {
        "base_model": "MobileNetV2",
        "pretrained": "ImageNet",
        "input_shape": [224, 224, 3],
        "total_params": 3500000,  # Approximate for MobileNetV2
        "trainable_params": 2000000
    }
}

# Save to file
output_path = Path('models/training_history.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(training_history, f, indent=4)

print(f"✅ Created {output_path}")
print(f"   - {training_history['epochs_trained']} epochs")
print(f"   - Best epoch: {training_history['best_epoch']}")
print(f"   - Final accuracy: {training_history['final_metrics']['test_accuracy']:.2%}")
print("\n💡 Update the values in this script with your actual training history from the notebook!")
print("   Then run: python create_training_history.py")

