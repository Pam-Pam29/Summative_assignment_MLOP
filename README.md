# PCOS Detection MLOPs System

A complete Machine Learning Operations (MLOPs) pipeline for Polycystic Ovary Syndrome (PCOS) detection using ultrasound images. This system includes model training, API deployment, web UI, retraining capabilities, and load testing.


## 🎯 Project Overview

This project implements an end-to-end MLOPs pipeline for PCOS detection from ultrasound images using:
- **Model**: MobileNetV2 transfer learning for binary classification (lightweight, ~3.4M parameters)
- **Dataset**: PCOS-XAI Ultrasound Dataset from Kaggle
- **Framework**: TensorFlow/Keras
- **API**: Flask REST API
- **UI**: Streamlit web application
- **Deployment**: Render

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
├── Notebook/
│   └── Victoria_Fakunle__PCOS_Assignment_MLOPs.ipynb    # Jupyter notebook with model development
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
├── render.yaml                       # Render deployment configuration
├── locustfile.py                     # Locust load testing script
├── run_api.py                        # Helper script to run API
├── start_local_api.bat/.sh           # Quick start scripts for local API
├── run_locust.bat/.sh                # Quick start scripts for Locust
├── test_training_local.py            # Local training test script
├── LOCUST_TESTING_GUIDE.md           # Comprehensive Locust testing guide
├── DEMO_SETUP.md                     # Local demo setup guide
└── TESTING_GUIDE.md                  # General testing guide
```

## 🔧 Prerequisites

- Python 3.11 or higher
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

Run the Jupyter notebook `Notebook/Victoria_Fakunle__PCOS_Assignment_MLOPs.ipynb` to train the initial model. The trained model will be saved to `models/pcos_model.h5` (or `models/pcos_model.keras`).


#### Manual Start

**Start the API Server:**
```bash
python src/api.py
# Or use the helper script:
python run_api.py
```

The API will be available at `http://localhost:5000`

**Start the Streamlit UI:**
```bash
streamlit run app.py
```

The UI will be available at `http://localhost:8501`

**Note**: For local demo, the UI will automatically connect to `http://localhost:5000`. No configuration needed!

### Option 2: Run with Render

Clone the repo
create 2 projects(one for api and one for ui)

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



**Deployment URLs**: 
- **API**: https://pcos-api-1fce.onrender.com
- **UI**: https://pcos-ui.onrender.com
- **API Health Check**: https://pcos-api-1fce.onrender.com/health


## 📹 Video Demo

**Video Demo Link**:https://youtu.be/uIP90MUGJP8?si=qaAmJF9h-gOBnwYc



## 📈 Results

### Model Performance

- **Test Accuracy**: 97.50%
- **Test Precision**: 94.68%
- **Test Recall**: 99.70%
- **Test F1-Score**: 97.12%
- **Test AUC**: 0.9984
- **Model Architecture**: MobileNetV2 (alpha=0.5, lightweight)
- **Model Size**: ~5-10 MB

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



## 🙏 Acknowledgments

- Dataset: [PCOS Detection using Ultrasound Images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)
- TensorFlow/Keras for deep learning framework
- Streamlit for UI development
- Flask for API development


For questions or issues, please open an issue on GitHub.

---

**Note**: This is a demonstration project for MLOPs. For production medical applications, additional validation, regulatory compliance, and clinical testing would be required.

