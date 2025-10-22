"""
Dataset Setup Script for Mini-ImageNet
=====================================

This script helps you download and setup the Mini-ImageNet dataset
for training with the ResNet50 implementation.
"""

import os
import zipfile
import requests
from pathlib import Path
import shutil

# Configuration
DATASET_DIR = "./data/mini-imagenet"
KAGGLE_DATASET = "ctrnngtrung/miniimagenet"

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print("🔧 Creating sample dataset for testing...")
    
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    
    # Create directory structure
    train_dir = Path(DATASET_DIR) / "train"
    val_dir = Path(DATASET_DIR) / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 sample classes
    classes = [f"class_{i:02d}" for i in range(10)]
    
    for class_name in classes:
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        
        # Create sample images for training (50 per class)
        for i in range(50):
            # Generate random image
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(train_dir / class_name / f"img_{i:03d}.jpg")
        
        # Create sample images for validation (20 per class)
        for i in range(20):
            # Generate random image
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(val_dir / class_name / f"img_{i:03d}.jpg")
    
    print(f"✅ Sample dataset created at {DATASET_DIR}")
    print(f"   - 10 classes")
    print(f"   - 500 training images (50 per class)")
    print(f"   - 200 validation images (20 per class)")

def download_with_kaggle_api():
    """Download dataset using Kaggle API"""
    try:
        import kaggle
        print("📥 Downloading Mini-ImageNet dataset using Kaggle API...")
        
        # Download dataset
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path="./temp_download", unzip=True)
        
        # Move to correct location
        temp_path = Path("./temp_download")
        dataset_path = Path(DATASET_DIR)
        
        if temp_path.exists():
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            shutil.move(str(temp_path), str(dataset_path))
            print(f"✅ Dataset downloaded and extracted to {DATASET_DIR}")
            return True
        
    except ImportError:
        print("❌ Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Error downloading with Kaggle API: {e}")
        return False

def check_dataset():
    """Check if dataset exists and is properly structured"""
    dataset_path = Path(DATASET_DIR)
    
    if not dataset_path.exists():
        return False, "Dataset directory does not exist"
    
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    
    if not train_path.exists():
        return False, "Training directory does not exist"
    
    if not val_path.exists():
        return False, "Validation directory does not exist"
    
    # Count classes and images
    train_classes = [d for d in train_path.iterdir() if d.is_dir()]
    val_classes = [d for d in val_path.iterdir() if d.is_dir()]
    
    if len(train_classes) == 0:
        return False, "No training classes found"
    
    if len(val_classes) == 0:
        return False, "No validation classes found"
    
    # Count total images
    train_images = sum(len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png"))) 
                      for class_dir in train_classes)
    val_images = sum(len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png"))) 
                    for class_dir in val_classes)
    
    return True, f"Dataset found: {len(train_classes)} classes, {train_images} train images, {val_images} val images"

def main():
    """Main setup function"""
    print("🚀 Mini-ImageNet Dataset Setup")
    print("=" * 50)
    
    # Check if dataset already exists
    exists, message = check_dataset()
    if exists:
        print(f"✅ {message}")
        print("Dataset is ready for training!")
        return
    
    print(f"❌ {message}")
    print("\nChoose an option to setup the dataset:")
    print("1. Download using Kaggle API (recommended)")
    print("2. Create sample dataset for testing")
    print("3. Manual download instructions")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        if download_with_kaggle_api():
            exists, message = check_dataset()
            if exists:
                print(f"✅ {message}")
            else:
                print(f"❌ Setup failed: {message}")
        else:
            print("❌ Kaggle API download failed. Try option 2 or 3.")
    
    elif choice == "2":
        create_sample_dataset()
        exists, message = check_dataset()
        if exists:
            print(f"✅ {message}")
        else:
            print(f"❌ Sample dataset creation failed: {message}")
    
    elif choice == "3":
        print("\n📋 Manual Download Instructions:")
        print("=" * 40)
        print("1. Go to: https://www.kaggle.com/datasets/ctrnngtrung/miniimagenet")
        print("2. Click 'Download' to get the dataset zip file")
        print("3. Extract the zip file")
        print(f"4. Move the extracted folders to: {DATASET_DIR}")
        print("5. Ensure the structure is:")
        print(f"   {DATASET_DIR}/")
        print("   ├── train/")
        print("   │   ├── class1/")
        print("   │   ├── class2/")
        print("   │   └── ...")
        print("   └── val/")
        print("       ├── class1/")
        print("       ├── class2/")
        print("       └── ...")
        print("\n6. Run this script again to verify the setup")
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()