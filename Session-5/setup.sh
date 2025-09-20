#!/bin/bash

# Setup script for MNIST Optimized Model - Session 5
echo "Setting up MNIST Optimized Model Environment..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p models
mkdir -p logs

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip first."
    exit 1
fi

# Install requirements
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Setup completed successfully!"
echo ""
echo "To start training, run:"
echo "  python3 train.py"
echo ""
echo "To use Docker, run:"
echo "  docker build -t mnist-optimized ."
echo "  docker run --gpus all -v \$(pwd)/data:/app/data mnist-optimized"