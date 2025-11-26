# Testing Guide - PCOS Detection MLOPs System

This guide will help you verify that all components of your MLOPs system are working correctly.

## 🧪 Quick Test Checklist

- [ ] API server starts without errors
- [ ] Model loads successfully
- [ ] Health check endpoint works
- [ ] Prediction endpoint works with test images
- [ ] UI loads and connects to API
- [ ] Data upload works
- [ ] Retraining can be triggered
- [ ] Load testing with Locust works

---

## Step 1: Test API Server

### 1.1 Start the API Server

```bash
# Option 1: Using run_api.py
python run_api.py

# Option 2: Using Flask directly
python src/api.py

# Option 3: Using gunicorn (production)
gunicorn src.api:app --bind 0.0.0.0:5000
```

**Expected Output:**
```
Starting PCOS Detection API Server...
API will be available at http://localhost:5000
Health check: http://localhost:5000/health
 * Running on http://0.0.0.0:5000
```

### 1.2 Test Health Check Endpoint

Open a new terminal and run:

```bash
# Using curl
curl http://localhost:5000/health

# Or visit in browser
# http://localhost:5000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_loaded_at": "2024-01-01T12:00:00",
  "model_file_exists": true,
  "model_file_path": "models/pcos_model.h5",
  "uptime": "active",
  "message": "Model loaded successfully at startup (MobileNetV2 - 13MB)"
}
```

**✅ Success Criteria:**
- Status code: 200
- `model_loaded`: true (or `model_file_exists`: true)
- No error messages

---

## Step 2: Test Prediction Endpoint

### 2.1 Test with a Sample Image

```bash
# Using curl (replace with actual image path)
curl -X POST http://localhost:5000/predict \
  -F "file=@data/test/infected/image11877.jpg"

# Or using Python
python test_prediction.py
```

**Expected Response:**
```json
{
  "success": true,
  "prediction": {
    "predicted_class": "Infected",
    "confidence": 0.9991,
    "probability": 0.9991,
    "class_index": 0,
    "threshold": 0.5
  }
}
```

**✅ Success Criteria:**
- Status code: 200
- `success`: true
- `predicted_class` is either "Infected" or "Non-infected"
- `confidence` is between 0 and 1

### 2.2 Test with Multiple Images

Test with both infected and non-infected images:

```bash
# Test infected image
curl -X POST http://localhost:5000/predict \
  -F "file=@data/test/infected/image11877.jpg"

# Test non-infected image  
curl -X POST http://localhost:5000/predict \
  -F "file=@data/test/notinfected/Image_SetB868.jpg"
```

**✅ Success Criteria:**
- Infected images → predicted_class: "Infected"
- Non-infected images → predicted_class: "Non-infected"
- High confidence scores (>0.8)

---

## Step 3: Test UI (Streamlit)

### 3.1 Start the UI

```bash
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### 3.2 Test Each Page

#### Home Page
- [ ] API status shows "Online"
- [ ] Model status shows "✅ Loaded" or "⏳ Not Loaded (Lazy Loading)"
- [ ] Training/Test image counts are displayed

#### Predict Page
1. Upload a test image from `data/test/infected/` or `data/test/notinfected/`
2. Click "Predict" button
3. Verify:
   - [ ] Prediction shows correct class ("Infected" or "Non-infected")
   - [ ] Confidence score is displayed
   - [ ] Progress bar shows confidence level
   - [ ] Color coding: Red for "Infected", Green for "Non-infected"

#### Visualizations Page
- [ ] Dataset distribution charts load
- [ ] Training history charts display (if available)
- [ ] Model performance metrics show
- [ ] Feature interpretations are displayed

#### Upload Data Page
1. Select category: "infected" or "notinfected"
2. Upload 2-3 test images
3. Click "Upload Files"
4. Verify:
   - [ ] Success message appears
   - [ ] Upload count is correct
   - [ ] Files are saved to `data/uploads/training/`

#### Retrain Model Page
1. After uploading data, go to "Retrain Model" page
2. Click "Start Retraining" button
3. Verify:
   - [ ] Training status shows "training"
   - [ ] Progress updates (may take several minutes)
   - [ ] Eventually shows "completed" with metrics

#### Model Status Page
- [ ] API status shows "Online"
- [ ] Model information displays
- [ ] Dataset statistics show correct counts

---

## Step 4: Test Data Upload Endpoint

### 4.1 Test Upload API

```bash
# Upload single image
curl -X POST http://localhost:5000/upload_training_data \
  -F "category=infected" \
  -F "files=@data/test/infected/image11877.jpg"

# Upload multiple images
curl -X POST http://localhost:5000/upload_training_data \
  -F "category=notinfected" \
  -F "files=@data/test/notinfected/Image_SetB868.jpg" \
  -F "files=@data/test/notinfected/Image_585.jpg"
```

**Expected Response:**
```json
{
  "success": true,
  "uploaded_count": 2,
  "error_count": 0,
  "uploaded_files": ["path/to/file1.jpg", "path/to/file2.jpg"],
  "errors": [],
  "log_saved": true
}
```

**✅ Success Criteria:**
- Status code: 200
- `uploaded_count` > 0
- Files appear in `data/uploads/training/{category}/`
- Upload log exists at `data/uploads/upload_log.json`

### 4.2 Verify Files Are Saved

```bash
# Check uploaded files
ls -la data/uploads/training/infected/
ls -la data/uploads/training/notinfected/

# Check upload log
cat data/uploads/upload_log.json
```

---

## Step 5: Test Retraining

### 5.1 Trigger Retraining via API

```bash
curl -X POST http://localhost:5000/retrain
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Training started",
  "status": {
    "status": "training",
    "progress": 0,
    "message": "Initializing training...",
    "started_at": "2024-01-01T12:00:00"
  }
}
```

### 5.2 Check Training Status

```bash
curl http://localhost:5000/training_status
```

**Expected Response (during training):**
```json
{
  "status": "training",
  "progress": 50,
  "message": "Training model...",
  "started_at": "2024-01-01T12:00:00"
}
```

**Expected Response (after completion):**
```json
{
  "status": "completed",
  "progress": 100,
  "message": "Training completed successfully",
  "started_at": "2024-01-01T12:00:00",
  "completed_at": "2024-01-01T12:05:00",
  "metrics": {
    "test_accuracy": 0.9525,
    "test_loss": 0.2594,
    "test_precision": 0.9014,
    "test_recall": 0.9970,
    "test_auc": 0.9984,
    "test_f1": 0.9468
  }
}
```

**✅ Success Criteria:**
- Training starts without errors
- Status progresses from "training" to "completed"
- Metrics are returned after completion
- New model is saved to `models/pcos_model.h5`

---

## Step 6: Test Other API Endpoints

### 6.1 Model Info

```bash
curl http://localhost:5000/model_info
```

**Expected Response:**
```json
{
  "model_loaded": true,
  "model_loaded_at": "2024-01-01T12:00:00",
  "model_path": "models/pcos_model.h5",
  "input_shape": [null, 224, 224, 3],
  "output_shape": [null, 1],
  "total_params": 3400000
}
```

### 6.2 Dataset Statistics

```bash
curl http://localhost:5000/dataset_stats
```

**Expected Response:**
```json
{
  "train": {
    "infected": 781,
    "notinfected": 1143,
    "total": 1924
  },
  "test": {
    "infected": 787,
    "notinfected": 1145,
    "total": 1932
  },
  "uploaded": {
    "infected": 2,
    "notinfected": 3,
    "total": 5
  }
}
```

---

## Step 7: Test Load Testing with Locust

### 7.1 Start API Server

```bash
python run_api.py
```

### 7.2 Start Locust

```bash
locust -f locustfile.py --host=http://localhost:5000
```

### 7.3 Access Locust Web UI

Open browser: `http://localhost:8089`

### 7.4 Configure and Run Test

1. **Number of users**: 50 (start small)
2. **Spawn rate**: 5 users/second
3. **Host**: http://localhost:5000
4. Click "Start Swarming"

### 7.5 Monitor Results

**✅ Success Criteria:**
- No failures (0%)
- Response times are reasonable (< 2 seconds)
- Requests per second > 5
- All endpoints respond successfully

**Expected Results:**
- GET /health: ~100-500ms average
- POST /predict: ~200-1000ms average
- GET /dataset_stats: ~50-200ms average

---

## Step 8: Test with Docker (Optional)

### 8.1 Build and Start Services

```bash
docker-compose up --build
```

### 8.2 Test API

```bash
curl http://localhost:5000/health
```

### 8.3 Test UI

Open browser: `http://localhost:8501`

### 8.4 Test with Multiple Containers

```bash
# Scale to 3 API containers
docker-compose up --scale api=3

# Test load balancing
for i in {1..10}; do curl http://localhost:5000/health; done
```

---

## Step 9: Verify Model Predictions Match Notebook

### 9.1 Test with Known Images

Use the same test images from your notebook and verify predictions match:

```python
# Quick Python test script
import requests

# Test infected image
with open('data/test/infected/image11877.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    print("Infected image:", response.json())

# Test non-infected image
with open('data/test/notinfected/Image_SetB868.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'file': f}
    )
    print("Non-infected image:", response.json())
```

**✅ Success Criteria:**
- Predictions match notebook results
- Class names are correct ("Infected" or "Non-infected")
- Confidence scores are similar

---

## 🐛 Troubleshooting

### API Won't Start

**Error**: `ModuleNotFoundError`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Error**: `Model not found`
- **Solution**: Ensure model file exists in `models/` directory
- Run the notebook to generate the model

### Predictions Are Wrong

**Issue**: Predictions don't match notebook
- **Check**: Class names in `src/prediction.py` match notebook
- **Check**: Model file is the same one from notebook
- **Check**: Image preprocessing matches notebook

### UI Can't Connect to API

**Error**: "Cannot connect to API"
- **Check**: API server is running on port 5000
- **Check**: `API_BASE_URL` environment variable is correct
- **Check**: Firewall/antivirus isn't blocking port 5000

### Retraining Fails

**Error**: "No training data uploaded"
- **Solution**: Upload data first via UI or API
- **Check**: Files are in `data/uploads/training/`

**Error**: "Training failed"
- **Check**: Sufficient disk space
- **Check**: Model file exists and is valid
- **Check**: Training data has both classes

---

## ✅ Final Verification Checklist

Before submitting, verify:

- [ ] API starts without errors
- [ ] Model loads successfully
- [ ] Predictions work correctly (test with 5+ images)
- [ ] UI loads and all pages work
- [ ] Data upload works (both infected and notinfected)
- [ ] Retraining completes successfully
- [ ] Load testing runs without errors
- [ ] All API endpoints return correct responses
- [ ] Class names match notebook ("Infected", "Non-infected")
- [ ] Model uses existing model as pre-trained during retraining

---

## 📝 Test Results Template

Document your test results:

```
Test Date: ___________
Tester: ___________

API Tests:
- Health Check: [ ] Pass [ ] Fail
- Prediction: [ ] Pass [ ] Fail
- Upload: [ ] Pass [ ] Fail
- Retrain: [ ] Pass [ ] Fail

UI Tests:
- Home: [ ] Pass [ ] Fail
- Predict: [ ] Pass [ ] Fail
- Visualizations: [ ] Pass [ ] Fail
- Upload: [ ] Pass [ ] Fail
- Retrain: [ ] Pass [ ] Fail

Load Testing:
- Users: _____
- Response Time: _____ ms
- Failures: _____ %
- RPS: _____

Issues Found:
1. ___________
2. ___________
```

---

**Need Help?** Check the logs:
- API logs: Terminal where `run_api.py` is running
- UI logs: Terminal where `streamlit run app.py` is running
- Docker logs: `docker-compose logs`

