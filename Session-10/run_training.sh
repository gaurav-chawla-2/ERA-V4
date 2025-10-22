#!/bin/bash

# ResNet50 ImageNet-mini Training Script
# =====================================

echo "🚀 Starting ResNet50 ImageNet-mini Training"
echo "==========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if dataset exists
if [ ! -d "./data/mini-imagenet" ]; then
    echo "❌ Dataset not found. Setting up dataset..."
    python3 setup_dataset.py
    
    if [ ! -d "./data/mini-imagenet" ]; then
        echo "❌ Dataset setup failed. Please run setup_dataset.py manually."
        exit 1
    fi
fi

# Install requirements
echo "📦 Installing requirements..."
pip3 install -r requirements.txt

# Create necessary directories
mkdir -p models results

# Run training
echo "🏃 Starting training..."
python3 train_resnet50.py

echo "✅ Training completed!"
echo "📊 Check the results/ folder for training analysis"
echo "💾 Check the models/ folder for saved models"