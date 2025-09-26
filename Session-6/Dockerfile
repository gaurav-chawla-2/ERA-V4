# Use Python 3.9 slim image as base for optimal size and performance
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Set environment variables for optimal Python behavior in containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for PyTorch and scientific computing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with no cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY model1.py .
COPY model2.py .
COPY model3.py .

COPY train.py .
COPY README.md .

# Create data directory for MNIST dataset
RUN mkdir -p data

# Create directory for model outputs
RUN mkdir -p models

# Set default command to run training
CMD ["python", "train.py"]

# Expose port for potential future web interface
EXPOSE 8080

# Add labels for better container management
LABEL maintainer="ERA-V4-Session-6"
LABEL description="Optimized MNIST CNN with <20k parameters achieving 99.4% accuracy"
LABEL version="1.0"