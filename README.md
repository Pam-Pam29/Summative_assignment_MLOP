# PCOS Detection MLOPs System

A complete Machine Learning Operations (MLOPs) pipeline for Polycystic Ovary Syndrome (PCOS) detection using ultrasound images. This system includes model training, API deployment, web UI, retraining capabilities, and load testing.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Load Testing](#load-testing)
- [Deployment](#deployment)
- [Video Demo](#video-demo)
- [Results](#results)

## 🎯 Project Overview

This project implements an end-to-end MLOPs pipeline for PCOS detection from ultrasound images using:
- **Model**: ResNet50 transfer learning for binary classification
- **Dataset**: PCOS Detection using Ultrasound Images from Kaggle
- **Framework**: TensorFlow/Keras
- **API**: Flask REST API
- **UI**: Streamlit web application
- **Deployment**: Docker containers

## ✨ Features

### Core Functionality
- ✅ **Image Prediction**: Upload ultrasound images for PCOS detection
- ✅ **Model Training**: Train ResNet50-based model with transfer learning
- ✅ **Model Retraining**: Upload new data and retrain the model
- ✅ **Data Visualization**: Interactive visualizations of dataset and model performance
- ✅ **Model Monitoring**: Track model uptime and performance metrics

### MLOPs Features
- ✅ **Data Acquisition**: Automated dataset download from Kaggle
- ✅ **Data Processing**: Image preprocessing and augmentation
- ✅ **Model Creation**: Transfer learning with ResNet50
- ✅ **Model Testing**: Comprehensive evaluation with multiple metrics
- ✅ **API Creation**: RESTful API for predictions and model management
- ✅ **UI Creation**: Web interface for all functionalities
- ✅ **Cloud Deployment**: Docker-based deployment ready
- ✅ **Load Testing**: Locust-based performance testing

## 📁 Project Structure

```
Summative_assignment_MLOP/
│
├── README.md
│
├── notebook/
│   └── PCOS_Detection_MLOP.ipynb    # Jupyter notebook with model development
│
├── src/
│   ├── preprocessing.py              # Data preprocessing functions
│   ├── model.py                      # Model building and training
│   ├── prediction.py                 # Prediction functions
│   ├── api.py                        # Flask API server
│   └── retrain.py                    # Model retraining module
│
├── data/
│   ├── train/                        # Training images
│   │   ├── infected/
│   │   └── notinfected/
│   ├── test/                         # Test images
│   │   ├── infected/
│   │   └── notinfected/
│   └── uploads/                      # Uploaded training data
│       └── training/
│
├── models/
│   ├── pcos_model.h5                 # Trained model file
│   └── training_history.json         # Training history and metrics
│
├── app.py                            # Streamlit UI application
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker image configuration
├── docker-compose.yml                # Docker Compose configuration
└── locustfile.py                     # Locust load testing script
```

## 🔧 Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- Git
- Kaggle API credentials (for dataset download)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Summative_assignment_MLOP.git
cd Summative_assignment_MLOP
```

**GitHub Repository**: [Add your GitHub repository URL here]

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

The dataset will be automatically downloaded when running the notebook. Alternatively, you can download it manually from [Kaggle](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images).

### 4. Train the Model

Run the Jupyter notebook `notebook/PCOS_Detection_MLOP.ipynb` to train the initial model. The trained model will be saved to `models/pcos_model.h5`.

## 🚀 Usage

### Option 1: Run Locally

#### Start the API Server

```bash
python src/api.py
```

The API will be available at `http://localhost:5000`

#### Start the Streamlit UI

```bash
streamlit run app.py
```

The UI will be available at `http://localhost:8501`

### Option 2: Run with Docker

#### Build and Start Services

```bash
docker-compose up --build
```

This will start both the API server and the Streamlit UI.

#### Access Services

- API: `http://localhost:5000`
- UI: `http://localhost:8501`

### Option 3: Run with Multiple Docker Containers (Load Testing)

To test with multiple API containers:

```bash
# Scale API service
docker-compose up --scale api=3

# Or use docker-compose with specific configuration
docker-compose -f docker-compose.yml up --scale api=5
```

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns API health status and model information.

### Predict
```
POST /predict
Content-Type: multipart/form-data
Body: file (image file)
```
Upload an image and get PCOS prediction.

**Response:**
```json
{
  "success": true,
  "prediction": {
    "predicted_class": "PCOS Infected",
    "confidence": 0.9876,
    "probability": 0.9876,
    "class_index": 1
  }
}
```

### Upload Training Data
```
POST /upload_training_data
Content-Type: multipart/form-data
Body: files (multiple image files), category (infected/notinfected)
```
Upload images for model retraining.

### Retrain Model
```
POST /retrain
```
Trigger model retraining with uploaded data.

### Training Status
```
GET /training_status
```
Get current training status and progress.

### Model Info
```
GET /model_info
```
Get model information and training history.

### Dataset Statistics
```
GET /dataset_stats
```
Get dataset statistics (train/test/uploaded counts).

## 📊 Load Testing

### Using Locust

1. **Start the API server** (if not already running):
```bash
python src/api.py
```

2. **Run Locust**:
```bash
locust -f locustfile.py --host=http://localhost:5000
```

3. **Access Locust Web UI**:
Open `http://localhost:8089` in your browser

4. **Configure Test**:
- Number of users: 100
- Spawn rate: 10 users/second
- Host: http://localhost:5000

5. **Run Load Test**:
Click "Start Swarming" to begin the test

### Testing with Multiple Containers

To test with different numbers of Docker containers:

```bash
# Test with 1 container
docker-compose up --scale api=1

# Test with 3 containers
docker-compose up --scale api=3

# Test with 5 containers
docker-compose up --scale api=5
```

Record latency and response times for each configuration.

### Expected Results

- **Single Container**: ~200-500ms average response time
- **3 Containers**: ~100-300ms average response time
- **5 Containers**: ~50-200ms average response time

## ☁️ Deployment

### Recommended: Render (Easiest & Free Public URLs)

**Render is recommended** because it provides free public URLs, which meets the assignment requirements for "Excellent" grade.

#### Quick Deploy to Render

1. **Push code to GitHub** (required for Render):
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Sign up at Render**: https://render.com (free account)

3. **Deploy API Service**:
   - Go to Render Dashboard → "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `pcos-api`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn src.api:app --bind 0.0.0.0:$PORT`
     - **Environment Variable**: `FLASK_ENV=production`
   - Click "Create Web Service"
   - Copy the API URL (e.g., `https://pcos-api.onrender.com`)

4. **Deploy UI Service**:
   - Go to Render Dashboard → "New +" → "Web Service"
   - Connect same GitHub repository
   - Configure:
     - **Name**: `pcos-ui`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
     - **Environment Variable**: `API_BASE_URL=https://pcos-api.onrender.com` (use your actual API URL)
   - Click "Create Web Service"
   - Copy the UI URL (e.g., `https://pcos-ui.onrender.com`)

**Detailed Guide**: See [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md) for step-by-step instructions.

**Deployment URLs**: 
- **API**: [Add your Render API URL here after deployment]
- **UI**: [Add your Render UI URL here after deployment]

**Note**: Free tier services may take 30 seconds to wake up after inactivity (cold start).

### Alternative: Other Cloud Platforms

#### AWS
- Use ECS (Elastic Container Service) with Fargate
- Use EC2 with Docker
- Use Elastic Beanstalk

#### Google Cloud Platform
- Use Cloud Run
- Use GKE (Google Kubernetes Engine)
- Use Compute Engine with Docker

#### Azure
- Use Azure Container Instances
- Use Azure Kubernetes Service
- Use App Service

#### Heroku
```bash
heroku create pcos-detection-app
heroku container:push web
heroku container:release web
```

**To deploy to Heroku:**
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Login: `heroku login`
3. Create app: `heroku create pcos-detection-mlops`
4. Deploy: `git push heroku main`

### Environment Variables

Set the following environment variables for production:

```bash
FLASK_ENV=production
API_BASE_URL=http://your-api-url:5000
```

## 📹 Video Demo

**Video Demo Link**: [Add your YouTube link here after recording]

The video demonstrates:
- Model prediction process (upload image, get prediction)
- Data upload functionality (bulk upload for retraining)
- Model retraining process (trigger retraining, monitor progress)
- Data visualizations (dataset stats, training history, feature interpretations)
- Load testing results (Locust performance testing)

**Note**: Please record a video with:
- Camera on (as required)
- Clear demonstration of prediction and retraining
- Good audio quality
- Upload to YouTube and update this link

## 📈 Results

### Model Performance

- **Test Accuracy**: 99.17%
- **Test Precision**: 98.62%
- **Test Recall**: 100.00%
- **Test F1-Score**: 99.30%
- **Test AUC**: 1.0000

### Evaluation Metrics

The model was evaluated using:
1. Accuracy
2. Loss
3. Precision
4. Recall
5. F1-Score
6. AUC-ROC
7. Confusion Matrix
8. Classification Report

### Load Testing Results

Results from Locust load testing:

**Option A: With Docker (Multiple Containers)**
| Containers | Avg Response Time | Min Response Time | Max Response Time | Requests/sec |
|------------|-------------------|-------------------|-------------------|--------------|
| 1          | TBD               | TBD               | TBD               | TBD          |
| 3          | TBD               | TBD               | TBD               | TBD          |
| 5          | TBD               | TBD               | TBD               | TBD          |

**Option B: Without Docker (Different Load Levels)**
| Load Level | Users | Avg Response Time | Min Response Time | Max Response Time | Requests/sec |
|------------|-------|-------------------|-------------------|-------------------|--------------|
| Light      | 10    | TBD               | TBD               | TBD               | TBD          |
| Medium     | 50    | TBD               | TBD               | TBD               | TBD          |
| Heavy      | 100   | TBD               | TBD               | TBD               | TBD          |

**Note**: Run actual load tests and update this table with your results. Use:
- 100 users, spawn rate 10
- Test for 2-3 minutes per container configuration
- Record average, min, max response times
- Note requests per second

**To run load tests:**
1. Start API server: `python run_api.py`
2. Run Locust: `locust -f locustfile.py --host=http://localhost:5000`
3. Open browser: http://localhost:8089
4. Configure: 100 users, spawn rate 10
5. Run for 2-3 minutes and record results

## 🎓 Features Demonstrated

### 1. Model Prediction
- ✅ Single image upload and prediction
- ✅ Confidence scores
- ✅ Class probabilities

### 2. Data Visualizations
- ✅ Dataset distribution (train/test)
- ✅ Training history curves
- ✅ Model performance metrics
- ✅ Feature interpretations

### 3. Upload Data
- ✅ Bulk image upload
- ✅ Category selection (infected/notinfected)
- ✅ File validation

### 4. Retraining
- ✅ Data upload and storage
- ✅ Preprocessing of uploaded data
- ✅ Model retraining trigger
- ✅ Training progress monitoring

### 5. Model Monitoring
- ✅ Model uptime tracking
- ✅ API health checks
- ✅ Performance metrics

## 🛠️ Troubleshooting

### Model Not Loading
- Ensure `models/pcos_model.h5` exists
- Check file permissions
- Verify model file is not corrupted

### API Connection Issues
- Verify API server is running on port 5000
- Check firewall settings
- Ensure CORS is enabled

### Docker Issues
- Ensure Docker is running
- Check port availability (5000, 8501)
- Review Docker logs: `docker-compose logs`

## 📝 License

This project is for educational purposes as part of the MLOPs assignment.

## 👤 Author

[Your Name]

## 🙏 Acknowledgments

- Dataset: [PCOS Detection using Ultrasound Images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)
- TensorFlow/Keras for deep learning framework
- Streamlit for UI development
- Flask for API development

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a demonstration project for MLOPs. For production medical applications, additional validation, regulatory compliance, and clinical testing would be required.

