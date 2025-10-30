#!/bin/bash

# Setup script for Tiny-ImageNet-200 dataset
# This script downloads the dataset to /opt/dlami/nvme and prepares it for training

echo "🚀 Setting up Tiny-ImageNet-200 dataset for Session-10"
echo "=================================================="

# Check if we're running with sudo privileges for /opt access
if [ ! -w "/opt" ]; then
    echo "⚠️  This script needs write access to /opt/dlami/nvme"
    echo "💡 Please run with sudo: sudo bash setup_tiny_imagenet.sh"
    exit 1
fi

# Run the Python download script
echo "📥 Starting dataset download..."
python3 download_tiny_imagenet.py

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset setup completed successfully!"
    echo "🎯 You can now run the training script:"
    echo "   python3 train_resnet50.py"
else
    echo ""
    echo "❌ Dataset setup failed!"
    echo "💡 Please check the error messages above and try again"
    exit 1
fi