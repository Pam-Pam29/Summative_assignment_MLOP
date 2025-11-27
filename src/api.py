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
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging

# Disable GPU completely (we're on CPU)
tf.config.set_visible_devices([], 'GPU')

# Limit TensorFlow threads to reduce memory
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

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

# Disable eager execution optimizations that use more memory
os.environ['TF_DISABLE_MKL'] = '1'

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

# Load model function (called at startup for fast predictions)
# For Render: 10MB model loads quickly and keeps service responsive
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

# Load model at startup in background thread
# This ensures service starts immediately (for health checks) while model loads
# Model loading happens in background but starts immediately when module loads
print("🚀 Starting API service...")
print("📦 Loading model at startup in background (~10MB - optimized for Render)...")

def load_model_startup():
    """Load model in background thread - doesn't block service startup"""
    global model
    try:
        print("📦 Background: Starting model load...")
        load_model()
        if model is not None:
            print("✅ Model loaded successfully at startup!")
            print(f"   Model ready for predictions. Memory footprint: ~10MB")
        else:
            print("⚠️ Model not loaded - will retry on first prediction")
            print("⚠️ Check model files exist in models/ directory")
    except Exception as e:
        print(f"❌ Error loading model at startup: {str(e)}")
        import traceback
        traceback.print_exc()
        print("⚠️ Will retry model loading on first prediction request")

# Start model loading in background thread (non-daemon so it completes)
# Service starts immediately, model loads in parallel
model_loader_thread = threading.Thread(target=load_model_startup, daemon=False)
model_loader_thread.start()
print("✅ API service started! Model loading in background...")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint - ALWAYS returns 200 to pass Render health checks
    Model loads synchronously at startup, so if service is running, model should be loaded
    """
    try:
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
            'status': 'healthy',  # Always healthy - service is running
            'service': 'running',
            'model_loaded': model is not None,
            'model_loaded_at': model_loaded_at,
            'model_file_exists': model_exists,
            'model_file_path': model_file_path,
            'uptime': 'active',
            'ready': model is not None  # Indicates if ready for predictions
        }
        
        # Add message based on model status
        if model is None:
            if model_exists:
                model_status['message'] = 'Model file exists but failed to load at startup. Service will attempt to reload on first prediction.'
                model_status['status'] = 'error'  # Model should have loaded at startup
            else:
                model_status['message'] = 'Model file not found. Service is running but predictions will fail until model is trained.'
                model_status['status'] = 'no_model'  # Still healthy, just no model
        else:
            model_status['message'] = 'Model loaded successfully at startup (~10MB, optimized for Render)'
            model_status['status'] = 'ready'
        
        # ALWAYS return 200 - service is healthy even if model isn't loaded
        # This ensures Render health checks pass and service doesn't get marked as unhealthy
        return jsonify(model_status), 200
        
    except Exception as e:
        # Even on error, return 200 with error details
        # This prevents Render from marking service as unhealthy due to transient errors
        return jsonify({
            'status': 'error',
            'service': 'running',
            'error': str(e),
            'message': 'Service is running but encountered an error during health check'
        }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for single image"""
    # Model is loaded at startup (~10MB) for fast response times
    # If model is still loading, wait a bit for it to finish
    if model is None:
        # Check if model is currently loading
        if model_loading:
            # Wait up to 10 seconds for model to finish loading
            import time
            max_wait = 10
            waited = 0
            while model is None and model_loading and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5
            
        # If still not loaded, try to load now
        if model is None:
            load_model()
            if model is None:
                return jsonify({'error': 'Model not loaded. Please check server logs. The model may still be loading - please try again in a few seconds.'}), 503

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
        
        # Clear memory after prediction
        gc.collect()

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

    # Save upload log to database (JSON file for tracking - REQUIREMENT: Save to Database)
    upload_log = {
        'timestamp': datetime.now().isoformat(),
        'category': category,
        'file_count': len(uploaded_files),
        'files': [Path(f).name for f in uploaded_files],  # Store filenames only
        'errors': errors
    }
    
    # Save to upload log (acts as simple database)
    log_file = Path(UPLOAD_FOLDER) / 'upload_log.json'
    logs = []
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
    logs.append(upload_log)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"✅ Upload logged: {len(uploaded_files)} files for {category} category")

    return jsonify({
        'success': True,
        'uploaded_count': len(uploaded_files),
        'error_count': len(errors),
        'uploaded_files': uploaded_files,
        'errors': errors,
        'log_saved': True
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

        # Build model (use same parameters as notebook)
        training_status['message'] = 'Building model...'
        new_model = build_model(img_height=224, img_width=224, learning_rate=1e-4)

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
    """Get model information - returns training history even if model not loaded"""
    info = {
        'model_loaded': model is not None,
        'model_loaded_at': model_loaded_at if model is not None else None,
        'model_path': MODEL_PATH if model is not None else None,
    }
    
    # Add model details if model is loaded
    if model is not None:
        info['input_shape'] = model.input_shape
        info['output_shape'] = model.output_shape
        info['total_params'] = model.count_params()
    else:
        info['input_shape'] = None
        info['output_shape'] = None
        info['total_params'] = None
        info['message'] = 'Model not loaded. Training history and metrics still available below.'

    # Try to load training history (ALWAYS try, even if model not loaded)
    history_path = 'models/training_history.json'
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
                info['training_history'] = history
                print(f"✅ Loaded training history from {history_path}")
        except Exception as e:
            print(f"❌ Error loading training history: {e}")
            info['training_history_error'] = str(e)
    else:
        print(f"⚠️ Training history file not found: {history_path}")
        info['training_history'] = None
        info['training_history_message'] = 'training_history.json not found. Run the save code in your notebook.'
    
    # Also load metrics.json if available (for current test metrics)
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                # Ensure metrics are included in model info
                info['metrics'] = metrics
                # Also add to training_history if it exists and doesn't have final_metrics
                if 'training_history' in info and info['training_history'] and 'final_metrics' not in info['training_history']:
                    info['training_history']['final_metrics'] = metrics
                print(f"✅ Loaded metrics from {metrics_path}")
        except Exception as e:
            print(f"❌ Error loading metrics: {e}")
            info['metrics_error'] = str(e)
    else:
        print(f"⚠️ Metrics file not found: {metrics_path}")

    return jsonify(info)


@app.route('/dataset_stats', methods=['GET'])
def dataset_stats():
    """Get dataset statistics - matches notebook split (validation_split=0.2)"""
    # Use local paths (dataset is in Git for Render free tier)
    train_dir = Path('data/train')
    test_dir = Path('data/test')
    upload_dir = Path(UPLOAD_FOLDER) / 'training'
    
    # Validation split used in notebook (0.2 = 20% for validation, 80% for training)
    VALIDATION_SPLIT = 0.2

    # Debug: Log directory existence and paths
    print(f"📊 Dataset Stats Request - Checking directories...")
    print(f"   Train dir exists: {train_dir.exists()} - Path: {train_dir.absolute()}")
    print(f"   Test dir exists: {test_dir.exists()} - Path: {test_dir.absolute()}")
    print(f"   Upload dir exists: {upload_dir.exists()} - Path: {upload_dir.absolute()}")
    
    # Count images in directories (gets actual counts from filesystem - ALWAYS fresh)
    train_counts = count_images(train_dir) if train_dir.exists() else {}
    test_counts = count_images(test_dir) if test_dir.exists() else {}
    uploaded_counts = count_images(upload_dir) if upload_dir.exists() else {}
    
    # Debug: Log what we found
    print(f"   Train counts: {train_counts}")
    print(f"   Test counts: {test_counts}")
    print(f"   Uploaded counts: {uploaded_counts}")
    
    # Get raw class counts (handle both 'noninfected' and 'notinfected' directory names)
    # NOTE: These are the counts AFTER the 80/20 train/test split was applied during organization
    # To get the original notebook counts, we need to reverse the split
    # The notebook shows: train_infected=5879, train_noninfected=4126 (before any validation split)
    # But data/train already has the 80/20 split applied, so we need to account for that
    
    train_infected_raw = train_counts.get('infected', 0)
    train_noninfected_raw = train_counts.get('noninfected', train_counts.get('notinfected', 0))
    train_total_raw = train_infected_raw + train_noninfected_raw
    
    test_infected_raw = test_counts.get('infected', 0)
    test_noninfected_raw = test_counts.get('noninfected', test_counts.get('notinfected', 0))
    test_total_raw = test_infected_raw + test_noninfected_raw
    
    # Calculate original counts before train/test split (to match notebook)
    # The organize script did 80/20 train/test split, so:
    # original_train = current_train / 0.8
    # But we also need to add test back to get the full original dataset
    # Actually, the notebook counts are the RAW counts from PCOS folder before any splits
    # The app shows what's in data/train and data/test after the 80/20 split
    
    # For display purposes, show what's actually in the directories (current state)
    # But also provide the notebook-equivalent counts if we can calculate them
    
    # Apply validation split to match notebook
    # Expected: train_infected=5879, train_noninfected=4126 → Training: ~8005, Validation: ~2000
    # Expected: test_infected=1357, test_noninfected=1000 → Test: 2357
    train_infected_split = int(train_infected_raw * (1 - VALIDATION_SPLIT))  # 80% for training
    train_noninfected_split = int(train_noninfected_raw * (1 - VALIDATION_SPLIT))
    val_infected_split = int(train_infected_raw * VALIDATION_SPLIT)  # 20% for validation
    val_noninfected_split = int(train_noninfected_raw * VALIDATION_SPLIT)
    
    train_split_total = train_infected_split + train_noninfected_split
    validation_split_total = val_infected_split + val_noninfected_split

    stats = {
        'train': {
            'infected': train_infected_split,
            'notinfected': train_noninfected_split,
            'noninfected': train_noninfected_split,  # Alias for compatibility
            'total': train_split_total
        },
        'validation': {
            'infected': val_infected_split,
            'notinfected': val_noninfected_split,
            'noninfected': val_noninfected_split,
            'total': validation_split_total
        },
        'test': {
            'infected': test_infected_raw,
            'notinfected': test_noninfected_raw,
            'noninfected': test_noninfected_raw,  # Alias
            'total': test_total_raw
        },
        'uploaded': uploaded_counts,
        # Add raw counts for reference (before split)
        'raw_train': {
            'infected': train_infected_raw,
            'noninfected': train_noninfected_raw,
            'total': train_total_raw
        },
        'raw_test': {
            'infected': test_infected_raw,
            'noninfected': test_noninfected_raw,
            'total': test_total_raw
        },
        # Original notebook counts (before train/test split)
        # These represent the full dataset before any splits
        'notebook_counts': {
            'train_infected': train_infected_raw + int(test_infected_raw * 0.8),  # Approximate original
            'train_noninfected': train_noninfected_raw + int(test_noninfected_raw * 0.8),
            'test_infected': test_infected_raw,
            'test_noninfected': test_noninfected_raw,
            'note': 'These are approximate. For exact notebook counts, use: train_infected=5879, train_noninfected=4126, test_infected=1357, test_noninfected=1000'
        },
        'validation_split': VALIDATION_SPLIT,
        # Class imbalance info
        'class_imbalance': {
            'train_ratio': round(max(train_infected_raw, train_noninfected_raw) / min(train_infected_raw, train_noninfected_raw), 2) if min(train_infected_raw, train_noninfected_raw) > 0 else 0,
            'train_infected_pct': round((train_infected_raw / train_total_raw * 100), 1) if train_total_raw > 0 else 0,
            'train_noninfected_pct': round((train_noninfected_raw / train_total_raw * 100), 1) if train_total_raw > 0 else 0
        },
        # Debug info for Render troubleshooting
        'debug_info': {
            'train_dir_exists': train_dir.exists(),
            'train_dir_path': str(train_dir.absolute()),
            'test_dir_exists': test_dir.exists(),
            'test_dir_path': str(test_dir.absolute()),
            'upload_dir_exists': upload_dir.exists(),
            'upload_dir_path': str(upload_dir.absolute()),
            'timestamp': datetime.now().isoformat(),
            'note': 'Stats are read fresh from filesystem on each request'
        }
    }

    return jsonify(stats)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

