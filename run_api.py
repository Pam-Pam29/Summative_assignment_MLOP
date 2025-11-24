"""
Simple script to run the API server
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables
os.environ['FLASK_APP'] = 'src.api'
os.environ['FLASK_ENV'] = 'development'

if __name__ == '__main__':
    from src.api import app
    print("Starting PCOS Detection API Server...")
    print("API will be available at http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=True)






