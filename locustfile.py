"""
Locust load testing script for PCOS Detection API
Simulates multiple users making prediction requests

Usage:
    # Test local API
    locust -f locustfile.py --host=http://localhost:5000
    
    # Test Render API
    locust -f locustfile.py --host=https://pcos-api-1fce.onrender.com
    
    # Headless mode (no UI)
    locust -f locustfile.py --host=https://pcos-api-1fce.onrender.com --headless -u 50 -r 10 -t 60s
"""

from locust import HttpUser, task, between, events
import random
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache sample images to avoid repeated file system access
_sample_images = None


def get_sample_images():
    """Get list of sample images for testing (cached)"""
    global _sample_images
    if _sample_images is None:
        test_dir = Path("data/test")
        if test_dir.exists():
            _sample_images = (
                list(test_dir.rglob("*.jpg")) +
                list(test_dir.rglob("*.jpeg")) +
                list(test_dir.rglob("*.png")) +
                list(test_dir.rglob("*.JPG")) +
                list(test_dir.rglob("*.JPEG")) +
                list(test_dir.rglob("*.PNG"))
            )
            logger.info(f"Found {len(_sample_images)} sample images for testing")
        else:
            _sample_images = []
            logger.warning(f"Test directory not found: {test_dir}")
    return _sample_images


class PCOSDetectionUser(HttpUser):
    """
    Simulates a normal user making prediction requests
    Realistic behavior: checks health, views info, makes predictions
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests (realistic user behavior)
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Check API health first
        with self.client.get("/health", catch_response=True, name="startup_health_check") as response:
            if response.status_code == 200:
                response.success()
                logger.info("API is healthy - user starting")
            else:
                response.failure(f"API health check failed: {response.status_code}")
                logger.warning(f"API health check failed: {response.status_code}")
    
    @task(5)  # Most common task - making predictions
    def predict_image(self):
        """Simulate image prediction request (most common user action)"""
        sample_images = get_sample_images()
        
        if not sample_images:
            logger.warning("No sample images available for prediction")
            return
        
        image_path = random.choice(sample_images)
        
        # Determine content type based on file extension
        ext = image_path.suffix.lower()
        content_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, content_type)}
                with self.client.post(
                    "/predict",
                    files=files,
                    catch_response=True,
                    timeout=90,  # Increased timeout for Render cold starts
                    name="predict_image"
                ) as response:
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            if result.get('success') or 'prediction' in result:
                                response.success()
                            else:
                                response.failure(f"Prediction failed: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            response.failure(f"Failed to parse response: {str(e)}")
                    elif response.status_code == 503:
                        response.failure("Service temporarily unavailable (cold start)")
                    else:
                        response.failure(f"HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error in predict_image: {str(e)}")
    
    @task(2)  # Second most common - checking health
    def health_check(self):
        """Check API health"""
        with self.client.get("/health", catch_response=True, name="health_check") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)  # Less common - viewing model info
    def model_info(self):
        """Get model information"""
        with self.client.get("/model_info", catch_response=True, name="model_info", timeout=10) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")
    
    @task(1)  # Less common - viewing dataset stats
    def dataset_stats(self):
        """Get dataset statistics"""
        with self.client.get("/dataset_stats", catch_response=True, name="dataset_stats", timeout=10) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dataset stats failed: {response.status_code}")


class LoadTestUser(HttpUser):
    """
    Heavy load testing user - simulates stress testing
    Makes rapid requests with minimal wait time
    """
    
    wait_time = between(0.5, 1.5)  # Faster requests for stress testing
    
    def on_start(self):
        """Quick health check on start"""
        self.client.get("/health", name="load_test_health_check")
    
    @task(10)  # Heavy focus on predictions
    def rapid_predictions(self):
        """Make rapid prediction requests (stress test)"""
        sample_images = get_sample_images()
        
        if not sample_images:
            return
        
        image_path = random.choice(sample_images)
        ext = image_path.suffix.lower()
        content_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, content_type)}
                with self.client.post(
                    "/predict",
                    files=files,
                    catch_response=True,
                    timeout=90,
                    name="rapid_predict"
                ) as response:
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            if result.get('success') or 'prediction' in result:
                                response.success()
                            else:
                                response.failure(f"Prediction failed: {result.get('error', 'Unknown')}")
                        except:
                            response.failure("Failed to parse JSON response")
                    else:
                        response.failure(f"HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error in rapid_predictions: {str(e)}")
    
    @task(1)  # Occasional health checks
    def health_check(self):
        """Quick health check"""
        self.client.get("/health", name="load_test_health")


# Event hooks for better reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    sample_count = len(get_sample_images())
    logger.info(f"🚀 Load test starting with {sample_count} sample images available")
    logger.info(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    logger.info("🛑 Load test stopped")













