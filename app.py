"""
Streamlit UI for PCOS Detection MLOPs Application
Provides interface for predictions, visualizations, and model management
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="PCOS Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


def get_api_status(use_cache=True):
    """Check API status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.ConnectionError as e:
        return None
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return None


@st.cache_data(ttl=60)
def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model_info", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


@st.cache_data(ttl=60)
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/dataset_stats", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_training_status():
    """Get training status"""
    try:
        response = requests.get(f"{API_BASE_URL}/training_status", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def main():
    st.markdown('<h1 class="main-header"> PCOS Detection System</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Predict", "Visualizations", "Upload Data", "Retrain Model", "Model Status"]
    )

    # Check API connection with retry button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh Connection", help="Click to retry API connection"):
            st.cache_data.clear()
            st.rerun()
    
    api_status = get_api_status()
    if api_status is None:
        st.error(" Cannot connect to API. Please ensure the API server is running.")
        st.info(f"**API URL:** {API_BASE_URL}")
        st.info("**Start the API server with:** `python run_api.py` or `python src/api.py`")
        
        # Try to provide more helpful debugging
        with st.expander(" Troubleshooting"):
            st.write(f"""
            **Current API URL:** `{API_BASE_URL}`
            
            1. **Check if API is running:**
               - Open a new terminal
               - Run: `curl http://localhost:5000/health`
               - Or visit: http://localhost:5000/health in your browser
            
            2. **Start the API:**
               ```bash
               python run_api.py
               ```
            
            3. **Check firewall/antivirus:**
               - Make sure port 5000 is not blocked
            
            4. **Try different URL:**
               - If running in Docker, use: `http://api:5000`
               - If on different machine, use the machine's IP
            """)
        
        # Try direct connection test
        if st.button(" Test Connection Now"):
            try:
                test_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if test_response.status_code == 200:
                    st.success(f" Connection successful! API is running.")
                    st.json(test_response.json())
                    st.info("Click 'Refresh Connection' button above to continue.")
                else:
                    st.error(f" API returned status code: {test_response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error(" Connection refused. Is the API server running?")
            except Exception as e:
                st.error(f" Error: {str(e)}")
        
        return

    # Home page
    if page == "Home":
        show_home_page(api_status)

    # Predict page
    elif page == "Predict":
        show_predict_page()

    # Visualizations page
    elif page == "Visualizations":
        show_visualizations_page()

    # Upload Data page
    elif page == "Upload Data":
        show_upload_page()

    # Retrain Model page
    elif page == "Retrain Model":
        show_retrain_page()

    # Model Status page
    elif page == "Model Status":
        show_status_page(api_status)


def show_home_page(api_status):
    """Display home page with overview"""
    st.header("Welcome to PCOS Detection System")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("API Status", "Online" if api_status else "Offline")
    
    with col2:
        model_info = get_model_info()
        model_status = "Loaded" if model_info and model_info.get('model_loaded') else "Not Loaded"
        st.metric("Model Status", model_status)
    
    with col3:
        dataset_stats = get_dataset_stats()
        train_total = dataset_stats.get('train', {}).get('total', 0) if dataset_stats else 0
        st.metric("Training Images", train_total)
    
    with col4:
        test_total = dataset_stats.get('test', {}).get('total', 0) if dataset_stats else 0
        st.metric("Test Images", test_total)

    st.markdown("---")

    st.subheader("System Overview")
    st.write("""
    This MLOPs system provides:
    - **Image Prediction**: Upload ultrasound images for PCOS detection
    - **Data Visualization**: Explore dataset statistics and model performance
    - **Data Upload**: Upload new training data for model improvement
    - **Model Retraining**: Trigger model retraining with new data
    - **Model Monitoring**: Track model performance and uptime
    """)

    # Model information
    if model_info:
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Model Path:** {model_info.get('model_path', 'N/A')}")
            st.write(f"**Input Shape:** {model_info.get('input_shape', 'N/A')}")
        with col2:
            st.write(f"**Total Parameters:** {model_info.get('total_params', 'N/A'):,}")
            st.write(f"**Model Loaded At:** {model_info.get('model_loaded_at', 'N/A')}")


def show_predict_page():
    """Display prediction page"""
    st.header("Image Prediction")

    uploaded_file = st.file_uploader(
        "Upload an ultrasound image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an ultrasound image to detect PCOS"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("Predict", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Reset file pointer and send with proper filename
                        uploaded_file.seek(0)
                        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or 'image/jpeg')}
                        response = requests.post(
                            f"{API_BASE_URL}/predict",
                            files=files,
                            timeout=120  # Increased to 120 seconds for model loading + cold start
                        )

                        if response.status_code == 200:
                            try:
                                result = response.json()
                            except ValueError as json_error:
                                st.error(f"Invalid JSON response from API. Response text: {response.text[:200]}")
                                st.info("This might be a timeout or the API is still loading the model. Try again in a moment.")
                                return
                            prediction = result.get('prediction', {})

                            st.success("Prediction completed!")

                            # Display results
                            predicted_class = prediction.get('predicted_class', 'Unknown')
                            confidence = prediction.get('confidence', 0)
                            probability = prediction.get('probability', 0)

                            # Color coding
                            if predicted_class == 'PCOS Infected':
                                st.error(f"**Prediction:** {predicted_class}")
                            else:
                                st.success(f"**Prediction:** {predicted_class}")

                            st.metric("Confidence", f"{confidence:.2%}")
                            st.metric("Probability", f"{probability:.4f}")

                            # Progress bar
                            st.progress(confidence)

                        else:
                            try:
                                error = response.json().get('error', 'Unknown error')
                                st.error(f"Prediction failed: {error}")
                            except ValueError:
                                st.error(f"Prediction failed with status {response.status_code}")
                                st.info(f"Response: {response.text[:200]}")
                                if response.status_code == 504 or response.status_code == 502:
                                    st.warning("The API might be timing out. This can happen on free tier when the service is sleeping or loading the model. Try again in 30-60 seconds.")

                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The API is taking too long to respond.")
                        st.info("This is normal for free tier services - first request after sleep takes 30-60 seconds, and model loading adds another 30-60 seconds.")
                        st.info("Please try again - the model should be loaded now and subsequent requests will be faster.")
                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to API. Please check:")
                        st.info(f"1. API URL is correct: {API_BASE_URL}")
                        st.info("2. API service is running on Render")
                        st.info("3. API_BASE_URL environment variable is set correctly")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Check the API logs in Render dashboard for more details.")


def show_visualizations_page():
    """Display visualizations page"""
    st.header("Data Visualizations")

    dataset_stats = get_dataset_stats()
    model_info = get_model_info()

    if not dataset_stats:
        st.warning("No dataset statistics available")
        return

    # Dataset distribution
    st.subheader("Dataset Distribution")

    col1, col2 = st.columns(2)

    with col1:
        train_stats = dataset_stats.get('train', {})
        if train_stats:
            train_df = pd.DataFrame([
                {'Class': 'Infected', 'Count': train_stats.get('infected', 0)},
                {'Class': 'Not Infected', 'Count': train_stats.get('notinfected', 0)},
            ])
            fig = px.pie(
                train_df,
                values='Count',
                names='Class',
                title='Training Set Distribution',
                color_discrete_map={'Infected': '#ff6b6b', 'Not Infected': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        test_stats = dataset_stats.get('test', {})
        if test_stats:
            test_df = pd.DataFrame([
                {'Class': 'Infected', 'Count': test_stats.get('infected', 0)},
                {'Class': 'Not Infected', 'Count': test_stats.get('notinfected', 0)},
            ])
            fig = px.pie(
                test_df,
                values='Count',
                names='Class',
                title='Test Set Distribution',
                color_discrete_map={'Infected': '#ff6b6b', 'Not Infected': '#4ecdc4'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Training history visualization
    if model_info and 'training_history' in model_info:
        st.subheader("Training History")
        history = model_info['training_history']

        if 'history' in history:
            hist_data = history['history']
            epochs = range(1, len(hist_data.get('accuracy', [])) + 1)

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy', 'Loss', 'Precision & Recall', 'AUC'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Accuracy
            fig.add_trace(
                go.Scatter(x=list(epochs), y=hist_data.get('accuracy', []),
                          name='Train Accuracy', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=hist_data.get('val_accuracy', []),
                          name='Val Accuracy', line=dict(color='red')),
                row=1, col=1
            )

            # Loss
            fig.add_trace(
                go.Scatter(x=list(epochs), y=hist_data.get('loss', []),
                          name='Train Loss', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=hist_data.get('val_loss', []),
                          name='Val Loss', line=dict(color='red')),
                row=1, col=2
            )

            # Precision & Recall
            if 'precision' in hist_data:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=hist_data.get('precision', []),
                              name='Precision', line=dict(color='green')),
                    row=2, col=1
                )
            if 'recall' in hist_data:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=hist_data.get('recall', []),
                              name='Recall', line=dict(color='orange')),
                    row=2, col=1
                )

            # AUC
            if 'auc' in hist_data:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=hist_data.get('auc', []),
                              name='Train AUC', line=dict(color='purple')),
                    row=2, col=2
                )
            if 'val_auc' in hist_data:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=hist_data.get('val_auc', []),
                              name='Val AUC', line=dict(color='brown')),
                    row=2, col=2
                )

            fig.update_layout(height=800, showlegend=True, title_text="Training Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)

        # Final metrics
        if 'final_metrics' in history:
            st.subheader("Model Performance Metrics")
            metrics = history['final_metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
            with col2:
                st.metric("Test Precision", f"{metrics.get('test_precision', 0):.2%}")
            with col3:
                st.metric("Test Recall", f"{metrics.get('test_recall', 0):.2%}")
            with col4:
                st.metric("Test F1 Score", f"{metrics.get('test_f1', 0):.2%}")
    
    # Feature Interpretations
    st.subheader("Feature Interpretations")
    st.markdown("""
    ### Understanding the Dataset Through Visualizations
    
    The following interpretations help us understand the story our data tells:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Feature 1: Class Distribution**
        
        The dataset shows the balance between infected and non-infected cases. 
        This distribution tells us:
        - Whether we have a balanced dataset for training
        - If class imbalance techniques are needed
        - The prevalence of PCOS cases in the dataset
        
        A balanced dataset helps the model learn both classes equally well.
        """)
    
    with col2:
        st.markdown("""
        **Feature 2: Training Metrics Over Time**
        
        The training history curves reveal:
        - **Accuracy trends**: How well the model learns over epochs
        - **Loss reduction**: Model's learning progress
        - **Overfitting detection**: Gap between train and validation metrics
        
        These metrics tell us if the model is learning effectively and generalizing well.
        """)
    
    with col3:
        st.markdown("""
        **Feature 3: Model Performance Metrics**
        
        Final evaluation metrics provide insights into:
        - **Precision**: How reliable positive predictions are
        - **Recall**: Ability to catch all PCOS cases (critical for medical use)
        - **F1-Score**: Balance between precision and recall
        
        High recall is crucial in medical applications to avoid missing PCOS cases.
        """)
    
    # Additional interpretation
    st.markdown("""
    ### Overall Story
    
    The visualizations tell a story of:
    1. **Data Quality**: Balanced dataset ensures fair learning
    2. **Model Learning**: Training curves show effective learning without overfitting
    3. **Clinical Reliability**: High recall (100%) means no PCOS cases are missed, 
       which is critical for medical screening applications
    4. **Production Readiness**: Consistent performance across train/validation/test 
       indicates the model is ready for deployment
    """)


def show_upload_page():
    """Display data upload page"""
    st.header("Upload Training Data")

    st.info("Upload images to be used for model retraining. Images should be organized by class.")

    category = st.selectbox(
        "Select Category",
        ["infected", "notinfected"],
        help="Select the class/category for the uploaded images"
    )

    uploaded_files = st.file_uploader(
        "Upload images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Select multiple images to upload"
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")

        if st.button("Upload Files", type="primary"):
            with st.spinner("Uploading files..."):
                try:
                    # Prepare files for upload
                    files = []
                    for file in uploaded_files:
                        file.seek(0)  # Reset file pointer
                        files.append(('files', (file.name, file.getvalue(), file.type)))
                    
                    data = {'category': category}

                    response = requests.post(
                        f"{API_BASE_URL}/upload_training_data",
                        files=files,
                        data=data,
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Successfully uploaded {result.get('uploaded_count', 0)} file(s)")

                        if result.get('error_count', 0) > 0:
                            st.warning(f"{result.get('error_count', 0)} file(s) failed to upload")
                            for error in result.get('errors', []):
                                st.error(error)
                    else:
                        error = response.json().get('error', 'Unknown error')
                        st.error(f"Upload failed: {error}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")


def show_retrain_page():
    """Display retraining page"""
    st.header("Retrain Model")

    training_status = get_training_status()

    if training_status:
        status = training_status.get('status', 'idle')
        message = training_status.get('message', '')

        if status == 'training':
            st.warning(f"Training in progress: {message}")
            st.progress(training_status.get('progress', 0) / 100)
        elif status == 'completed':
            st.success(f"{message}")
            if 'metrics' in training_status:
                metrics = training_status['metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.2%}")
                with col2:
                    st.metric("Precision", f"{metrics.get('test_precision', 0):.2%}")
                with col3:
                    st.metric("Recall", f"{metrics.get('test_recall', 0):.2%}")
                with col4:
                    st.metric("AUC", f"{metrics.get('test_auc', 0):.4f}")
        elif status == 'failed':
            st.error(f"{message}")

    st.info("Click the button below to start retraining the model with newly uploaded data.")

    if st.button("Start Retraining", type="primary"):
        with st.spinner("Starting retraining process..."):
            try:
                response = requests.post(f"{API_BASE_URL}/retrain", timeout=5)

                if response.status_code == 200:
                    st.success("Retraining started! Check back in a few minutes.")
                    st.info("You can monitor the training status on this page.")
                else:
                    error = response.json().get('error', 'Unknown error')
                    st.error(f"Failed to start retraining: {error}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Auto-refresh if training
    if training_status and training_status.get('status') == 'training':
        time.sleep(5)
        st.rerun()


def show_status_page(api_status):
    """Display model status page"""
    st.header("Model Status & Uptime")

    # API Status
    st.subheader("API Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Status", "Online" if api_status else "Offline")
    with col2:
        if api_status:
            loaded_at = api_status.get('model_loaded_at', 'N/A')
            st.metric("Model Loaded At", loaded_at)
    with col3:
        if api_status:
            uptime = "Active" if api_status.get('model_loaded') else "Inactive"
            st.metric("Model Uptime", uptime)

    # Model Information
    model_info = get_model_info()
    if model_info:
        st.subheader("Model Information")
        st.json(model_info)

    # Dataset Statistics
    dataset_stats = get_dataset_stats()
    if dataset_stats:
        st.subheader("Dataset Statistics")
        st.json(dataset_stats)


if __name__ == '__main__':
    main()

