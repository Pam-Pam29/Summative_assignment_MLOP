"""
Locust load testing script for PCOS Detection API
Simulates multiple users making prediction requests
"""

from locust import HttpUser, task, between
import random
import os
from pathlib import Path


class PCOSDetectionUser(HttpUser):
    """Simulates a user making prediction requests"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Check API health
        response = self.client.get("/health")
        if response.status_code != 200:
            print("API is not healthy!")
    
    @task(3)
    def predict_image(self):
        """Simulate image prediction request"""
        # Use a sample image if available, otherwise skip
        sample_images = list(Path("data/test").rglob("*.jpg")) + \
                       list(Path("data/test").rglob("*.png"))
        
        if sample_images:
            image_path = random.choice(sample_images)
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                with self.client.post(
                    "/predict",
                    files=files,
                    catch_response=True,
                    name="predict_image"
                ) as response:
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            response.success()
                        else:
                            response.failure("Prediction failed")
                    else:
                        response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check API health"""
        self.client.get("/health", name="health_check")
    
    @task(1)
    def model_info(self):
        """Get model information"""
        self.client.get("/model_info", name="model_info")
    
    @task(1)
    def dataset_stats(self):
        """Get dataset statistics"""
        self.client.get("/dataset_stats", name="dataset_stats")


class LoadTestUser(HttpUser):
    """Heavy load testing user"""
    
    wait_time = between(0.5, 1.5)
    
    @task(10)
    def rapid_predictions(self):
        """Make rapid prediction requests"""
        sample_images = list(Path("data/test").rglob("*.jpg")) + \
                       list(Path("data/test").rglob("*.png"))
        
        if sample_images:
            image_path = random.choice(sample_images)
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                self.client.post("/predict", files=files, name="rapid_predict")












