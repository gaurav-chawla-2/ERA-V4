"""
Download Full ImageNet Dataset
=============================

This script provides instructions and utilities for downloading the full ImageNet dataset
to /opt/dlami/nvme for use with the ResNet50 training script.

Note: Full ImageNet requires registration and manual download from the official website.
"""

import os
import sys
import shutil
from pathlib import Path

# Configuration
DOWNLOAD_DIR = "/data"
DATASET_NAME = "imagenet"

def create_directory(path):
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"✅ Directory created/verified: {path}")
        return True
    except PermissionError:
        print(f"❌ Permission denied: Cannot create directory {path}")
        print("💡 Try running with sudo or check directory permissions")
        return False
    except Exception as e:
        print(f"❌ Error creating directory {path}: {e}")
        return False

def check_imagenet_structure(dataset_path):
    """Check if ImageNet dataset has the correct structure"""
    print("🔍 Checking ImageNet dataset structure...")
    
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    if not os.path.exists(val_dir):
        # Check for alternative validation directory names
        alt_names = ['validation', 'valid', 'test']
        found = False
        for alt in alt_names:
            alt_path = os.path.join(dataset_path, alt)
            if os.path.exists(alt_path):
                print(f"✅ Found validation directory: {alt}")
                found = True
                break
        
        if not found:
            print(f"❌ Validation directory not found")
            return False
    else:
        print("✅ Found validation directory: val")
    
    # Count training classes
    try:
        train_classes = [d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
        print(f"📊 Training classes found: {len(train_classes)}")
        
        if len(train_classes) != 1000:
            print(f"⚠️  Expected 1000 classes, found {len(train_classes)}")
        
    except Exception as e:
        print(f"❌ Error reading training directory: {e}")
        return False
    
    return True

def print_download_instructions():
    """Print instructions for downloading ImageNet"""
    print("📋 ImageNet Download Instructions")
    print("=" * 50)
    print()
    print("ImageNet requires registration and manual download from the official website.")
    print("Here's how to get the dataset:")
    print()
    print("1. 🌐 Visit: https://www.image-net.org/")
    print("2. 📝 Register for an account")
    print("3. 📥 Download the following files:")
    print("   - ILSVRC2012_img_train.tar (~138GB)")
    print("   - ILSVRC2012_img_val.tar (~6.3GB)")
    print("   - ILSVRC2012_devkit_t12.tar.gz (~2.5MB)")
    print()
    print("4. 📁 Extract to the following structure:")
    print(f"   {os.path.join(DOWNLOAD_DIR, DATASET_NAME)}/")
    print("   ├── train/")
    print("   │   ├── n01440764/")
    print("   │   ├── n01443537/")
    print("   │   └── ... (1000 classes)")
    print("   ├── val/")
    print("   │   ├── n01440764/")
    print("   │   ├── n01443537/")
    print("   │   └── ... (1000 classes)")
    print("   └── devkit/")
    print()
    print("5. 🔧 Extraction commands:")
    print("   # Create directories")
    print(f"   sudo mkdir -p {os.path.join(DOWNLOAD_DIR, DATASET_NAME)}")
    print(f"   cd {os.path.join(DOWNLOAD_DIR, DATASET_NAME)}")
    print()
    print("   # Extract training data")
    print("   sudo tar -xf ILSVRC2012_img_train.tar")
    print("   sudo mkdir train")
    print("   sudo mv *.tar train/")
    print("   cd train")
    print("   for f in *.tar; do")
    print("     sudo mkdir ${f%.tar}")
    print("     sudo tar -xf $f -C ${f%.tar}")
    print("     sudo rm $f")
    print("   done")
    print("   cd ..")
    print()
    print("   # Extract validation data")
    print("   sudo tar -xf ILSVRC2012_img_val.tar")
    print("   sudo mkdir val")
    print("   sudo mv *.JPEG val/")
    print()
    print("   # Extract devkit")
    print("   sudo tar -xzf ILSVRC2012_devkit_t12.tar.gz")
    print()
    print("6. 🔧 Organize validation data by class:")
    print("   # Use the provided validation script or organize manually")
    print("   # The validation images need to be sorted into class folders")
    print()

def create_validation_script():
    """Create a script to organize validation data"""
    script_content = '''#!/bin/bash
# Script to organize ImageNet validation data into class folders

VALDIR="/opt/dlami/nvme/imagenet/val"
DEVKIT="/opt/dlami/nvme/imagenet/ILSVRC2012_devkit_t12"

if [ ! -d "$VALDIR" ]; then
    echo "Validation directory not found: $VALDIR"
    exit 1
fi

if [ ! -d "$DEVKIT" ]; then
    echo "Devkit directory not found: $DEVKIT"
    exit 1
fi

# Create class directories
echo "Creating class directories..."
while IFS= read -r line; do
    class_id=$(echo $line | cut -d' ' -f1)
    mkdir -p "$VALDIR/$class_id"
done < "$DEVKIT/data/meta.mat"

# Move images to appropriate class folders
echo "Moving validation images to class folders..."
while IFS= read -r line; do
    img_name=$(echo $line | cut -d' ' -f1)
    class_id=$(echo $line | cut -d' ' -f2)
    if [ -f "$VALDIR/$img_name" ]; then
        mv "$VALDIR/$img_name" "$VALDIR/$class_id/"
    fi
done < "$DEVKIT/data/ILSVRC2012_validation_ground_truth.txt"

echo "Validation data organization complete!"
'''
    
    script_path = "/Users/gc/Documents/code/ERA/ERA-V4/Session-10/organize_imagenet_val.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"📝 Created validation organization script: {script_path}")

def main():
    """Main function"""
    print("🚀 ImageNet Dataset Setup Helper")
    print("=" * 50)
    
    # Check if running with appropriate permissions
    if not os.access('/opt', os.W_OK):
        print("⚠️  Warning: You may need sudo privileges to write to /opt/dlami/nvme")
        print("💡 Consider running: sudo python download_imagenet.py")
    
    # Create download directory
    dataset_path = os.path.join(DOWNLOAD_DIR, DATASET_NAME)
    if not create_directory(dataset_path):
        sys.exit(1)
    
    # Check if dataset already exists
    if check_imagenet_structure(dataset_path):
        print("✅ ImageNet dataset found and appears to be correctly structured!")
        print("🎯 Dataset is ready for training!")
        print(f"📁 Dataset location: {dataset_path}")
        return
    
    # Print download instructions
    print_download_instructions()
    
    # Create validation organization script
    create_validation_script()
    
    print("\n💡 After downloading and extracting:")
    print("1. Run the validation organization script:")
    print("   sudo bash organize_imagenet_val.sh")
    print("2. Verify the setup:")
    print("   python verify_setup.py")
    print("3. Start training:")
    print("   python train_resnet50.py")

if __name__ == "__main__":
    main()