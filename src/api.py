"""
Flask API for PCOS Detection Model
Provides endpoints for prediction and model management
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import threading
from werkzeug.utils import secure_filename
import tensorflow as tf

# Optimize TensorFlow for low memory usage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging
# Limit GPU memory growth (even though we're on CPU, this helps)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Set TensorFlow to use less memory
tf.config.experimental.enable_op_determinism()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.prediction import predict_single_image, validate_image_file, preprocess_image
from src.model import load_saved_model, build_model, train_model, save_training_history
from src.preprocessing import create_data_generators, count_images

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# Try multiple model file formats
MODEL_PATHS = [
    'models/pcos_model.keras',  # Newer format - most compatible
    'models/pcos_model.h5',     # Legacy H5 format
    'models/pcos_model'         # SavedModel format
]
# Try multiple weights file names
WEIGHTS_PATHS = [
    'models/pcos_model.weights.h5',  # Common naming
    'models/pcos_model_weights.h5',   # Alternative naming
]
WEIGHTS_PATH = WEIGHTS_PATHS[0]  # Default
MODEL_PATH = 'models/pcos_model.h5'  # Default/fallback
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure directories exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)

# Global variables
model = None
model_loaded_at = None
model_loading = False  # Track if model is currently being loaded
is_training = False
training_status = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'started_at': None,
    'completed_at': None
}

# Lazy load model (only when needed, not at startup)
def load_model():
    global model, model_loaded_at, MODEL_PATH, model_loading
    
    # If already loading, wait
    if model_loading:
        return None
    
    # If already loaded, return
    if model is not None:
        return model
    
    model_loading = True
    try:
        # Clear any existing TensorFlow sessions to free memory
        tf.keras.backend.clear_session()
        
        # IMPORTANT: Try weights first (much smaller memory footprint!)
        # Weights-only loading uses less memory than full model
        if model is None:
            for weights_path in WEIGHTS_PATHS:
                if os.path.exists(weights_path):
                    print(f"Attempting to rebuild model from weights: {weights_path}...")
                    try:
                        from src.model import build_model
                        # Rebuild the architecture (lightweight)
                        model = build_model(img_height=224, img_width=224, learning_rate=1e-4)
                        # Load the weights (much smaller than full model)
                        model.load_weights(weights_path)
                        # Compile the model
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
                        model_loaded_at = datetime.now().isoformat()
                        MODEL_PATH = weights_path
                        print(f"✅ Model rebuilt and loaded from weights: {weights_path}")
                        return model
                    except Exception as e:
                        print(f"❌ Error rebuilding from weights {weights_path}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        model = None  # Reset for next attempt
                        continue
        
        # If weights loading failed, try full model (uses more memory)
        # Try loading from different formats
        for model_path in MODEL_PATHS:
            try:
                if os.path.exists(model_path) or os.path.isdir(model_path):
                    print(f"Attempting to load model from {model_path}...")
                    
                    # Try loading with compile=False first (for compatibility)
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False)
                        # Recompile with the same metrics
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
                        model_loaded_at = datetime.now().isoformat()
                        MODEL_PATH = model_path  # Update to the working path
                        print(f"✅ Model loaded successfully from {model_path}")
                        return
                    except Exception as e1:
                        print(f"  Error loading {model_path} with compile=False: {str(e1)[:100]}")
                        # Try with compile=True
                        try:
                            model = tf.keras.models.load_model(model_path)
                            model_loaded_at = datetime.now().isoformat()
                            MODEL_PATH = model_path
                            print(f"✅ Model loaded successfully from {model_path}")
                            return
                        except Exception as e2:
                            print(f"  Error loading {model_path}: {str(e2)[:100]}")
                            continue
            except Exception as e:
                print(f"  Error checking {model_path}: {str(e)[:100]}")
                continue
        
        # If we get here, no model loaded
        if model is None:
            print("❌ Could not load model from any of the following paths:")
            for path in MODEL_PATHS:
                exists = os.path.exists(path) or os.path.isdir(path)
                print(f"   - {path}: {'✅ exists' if exists else '❌ not found'}")
            for weights_path in WEIGHTS_PATHS:
                if os.path.exists(weights_path):
                    print(f"   - {weights_path}: ✅ exists (can rebuild from weights)")
                else:
                    print(f"   - {weights_path}: ❌ not found")
            print("\n💡 Solution: Save weights only in Colab and download them.")
            print("   See COLAB_SAVE_WEIGHTS_ONLY.md for instructions.")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        model_loading = False
    
    return model

# Don't load model at startup - use lazy loading instead
# load_model()  # Commented out to save memory at startup


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - doesn't load model to save memory"""
    # Check if model files exist without loading
    model_exists = False
    model_file_path = None
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path) or os.path.isdir(model_path):
            model_exists = True
            model_file_path = model_path
            break
    
    # Check weights files
    if not model_exists:
        for weights_path in WEIGHTS_PATHS:
            if os.path.exists(weights_path):
                model_exists = True
                model_file_path = weights_path
                break
    
    model_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_loaded_at': model_loaded_at,
        'model_file_exists': model_exists,
        'model_file_path': model_file_path,
        'uptime': 'active'
    }
    
    # Add message if model not loaded yet (lazy loading)
    if model is None:
        if model_exists:
            model_status['message'] = 'Model file exists but not loaded yet. Will load on first prediction request.'
        else:
            model_status['error'] = f'Model file not found. Please train the model first.'
    
    return jsonify(model_status)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for single image"""
    # Lazy load model on first prediction request
    if model is None:
        load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    # Handle case where filename might be None or empty
    filename = file.filename if file.filename else 'uploaded_image.jpg'
    if filename == '' or filename is None:
        # Try to get filename from Content-Disposition header
        content_disposition = request.headers.get('Content-Disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"\'')
        else:
            filename = 'uploaded_image.jpg'
    
    # Check file extension
    if not allowed_file(filename):
        return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}. Got: {filename}'}), 400

    filepath = None
    try:
        # Save uploaded file temporarily
        filename = secure_filename(filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        file.save(filepath)

        # Validate image
        is_valid, error_msg = validate_image_file(filepath)
        if not is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Invalid image: {error_msg}'}), 400

        # Make prediction
        result = predict_single_image(model, filepath)

        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            'success': True,
            'prediction': result
        })

    except Exception as e:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({'error': str(e)}), 500


@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload training data for retraining"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    category = request.form.get('category', 'unknown')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    if category not in ['infected', 'notinfected', 'noninfected']:
        return jsonify({'error': 'Invalid category. Must be "infected" or "notinfected"'}), 400

    # Normalize category name
    if category == 'noninfected':
        category = 'notinfected'

    uploaded_files = []
    errors = []

    # Create category directory
    category_dir = Path(UPLOAD_FOLDER) / 'training' / category
    category_dir.mkdir(parents=True, exist_ok=True)

    for idx, file in enumerate(files):
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filepath = category_dir / f"{timestamp}_{idx}_{filename}"
                file.save(str(filepath))

                # Validate image
                is_valid, error_msg = validate_image_file(str(filepath))
                if not is_valid:
                    if os.path.exists(str(filepath)):
                        os.remove(str(filepath))
                    errors.append(f"{filename}: {error_msg}")
                else:
                    uploaded_files.append(str(filepath))
            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")

    return jsonify({
        'success': True,
        'uploaded_count': len(uploaded_files),
        'error_count': len(errors),
        'uploaded_files': uploaded_files,
        'errors': errors
    })


@app.route('/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining"""
    global is_training, training_status

    if is_training:
        return jsonify({'error': 'Training already in progress'}), 400

    # Check if training data exists
    training_data_dir = Path(UPLOAD_FOLDER) / 'training'
    if not training_data_dir.exists():
        return jsonify({'error': 'No training data uploaded'}), 400

    # Start training in background thread
    thread = threading.Thread(target=train_model_background)
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'message': 'Training started',
        'status': training_status
    })


def train_model_background():
    """Background training function"""
    global is_training, training_status, model, model_loaded_at

    is_training = True
    training_status = {
        'status': 'training',
        'progress': 0,
        'message': 'Initializing training...',
        'started_at': datetime.now().isoformat(),
        'completed_at': None
    }

    try:
        # Get training data paths
        train_dir = Path('data/train')
        upload_dir = Path(UPLOAD_FOLDER) / 'training'

        # Merge uploaded data with existing training data
        if upload_dir.exists():
            for category_dir in upload_dir.iterdir():
                if category_dir.is_dir():
                    dest_dir = train_dir / category_dir.name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    # Copy files
                    for img_file in category_dir.glob('*'):
                        if img_file.is_file():
                            import shutil
                            shutil.copy2(img_file, dest_dir / img_file.name)

        # Create data generators
        training_status['message'] = 'Creating data generators...'
        train_gen, val_gen, test_gen = create_data_generators(
            train_dir,
            Path('data/test'),
            img_size=(224, 224),
            batch_size=64,
            validation_split=0.2
        )

        # Build model
        training_status['message'] = 'Building model...'
        new_model = build_model()

        # Train model
        training_status['message'] = 'Training model...'
        history = train_model(
            new_model,
            train_gen,
            val_gen,
            epochs=20,
            model_save_path=MODEL_PATH,
            verbose=0
        )

        # Evaluate on test set
        training_status['message'] = 'Evaluating model...'
        # test_gen already created above

        test_results = new_model.evaluate(test_gen, verbose=0)
        test_metrics = {
            'test_accuracy': float(test_results[1]),
            'test_loss': float(test_results[0]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'test_auc': float(test_results[4])
        }

        # Save training history
        save_training_history(history, new_model, test_metrics)

        # Update global model
        model = new_model
        model_loaded_at = datetime.now().isoformat()
        
        # Save the newly trained model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        training_status = {
            'status': 'completed',
            'progress': 100,
            'message': 'Training completed successfully',
            'started_at': training_status['started_at'],
            'completed_at': datetime.now().isoformat(),
            'metrics': test_metrics
        }

    except Exception as e:
        training_status = {
            'status': 'failed',
            'progress': 0,
            'message': f'Training failed: {str(e)}',
            'started_at': training_status['started_at'],
            'completed_at': datetime.now().isoformat()
        }
    finally:
        is_training = False


@app.route('/training_status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 404

    info = {
        'model_loaded': True,
        'model_loaded_at': model_loaded_at,
        'model_path': MODEL_PATH,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params()
    }

    # Try to load training history
    history_path = 'models/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            info['training_history'] = history

    return jsonify(info)


@app.route('/dataset_stats', methods=['GET'])
def dataset_stats():
    """Get dataset statistics"""
    train_dir = Path('data/train')
    test_dir = Path('data/test')
    upload_dir = Path(UPLOAD_FOLDER) / 'training'

    stats = {
        'train': count_images(train_dir) if train_dir.exists() else {},
        'test': count_images(test_dir) if test_dir.exists() else {},
        'uploaded': count_images(upload_dir) if upload_dir.exists() else {}
    }

    return jsonify(stats)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

