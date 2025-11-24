FROM python:3.11-slim

WORKDIR /app

# Note: DNS configuration in docker-compose.yml handles runtime
# For build-time, we rely on Docker Desktop's network settings

# Install system dependencies with retry logic and timeout
RUN apt-get update --fix-missing || apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with retry logic
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/train data/test data/uploads/training models

# Expose ports
EXPOSE 5000 8501

# Default command (can be overridden)
CMD ["python", "src/api.py"]




