"""
Download Tiny-ImageNet-200 Dataset
=================================

This script downloads the Tiny-ImageNet-200 dataset to /opt/dlami/nvme
for use with the ResNet50 training script.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import sys

# Configuration
DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DOWNLOAD_DIR = "/data"  # Use dedicated /data mount (280GB available)
DATASET_NAME = "tiny-imagenet-200"
ZIP_FILE = "tiny-imagenet-200.zip"

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

def download_file(url, destination):
    """Download file with progress bar"""
    try:
        print(f"📥 Downloading {url}")
        print(f"📁 Destination: {destination}")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                bar_length = 50
                filled_length = (percent * bar_length) // 100
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"\r[{bar}] {percent}% ({downloaded // (1024*1024)}MB/{total_size // (1024*1024)}MB)", end='')
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print("\n✅ Download completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    try:
        print(f"📦 Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extraction completed!")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def verify_dataset(dataset_path):
    """Verify the dataset structure"""
    required_dirs = ['train', 'val', 'test']
    
    print("🔍 Verifying dataset structure...")
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ Found {dir_name} directory")
            if dir_name == 'train':
                num_classes = len([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
                print(f"   📊 Training classes: {num_classes}")
        else:
            print(f"❌ Missing {dir_name} directory")
            return False
    
    return True

def main():
    """Main function to download and setup tiny-imagenet-200"""
    print("🚀 Tiny-ImageNet-200 Dataset Downloader")
    print("=" * 50)
    
    # Check if running with appropriate permissions
    if not os.access('/opt', os.W_OK):
        print("⚠️  Warning: You may need sudo privileges to write to /opt/dlami/nvme")
        print("💡 Consider running: sudo python download_tiny_imagenet.py")
    
    # Create download directory
    if not create_directory(DOWNLOAD_DIR):
        sys.exit(1)
    
    # Set paths
    zip_path = os.path.join(DOWNLOAD_DIR, ZIP_FILE)
    dataset_path = os.path.join(DOWNLOAD_DIR, DATASET_NAME)
    
    # Check if dataset already exists
    if os.path.exists(dataset_path) and verify_dataset(dataset_path):
        print(f"✅ Dataset already exists at {dataset_path}")
        print("🎯 Dataset is ready for training!")
        return
    
    # Download the dataset
    if not download_file(DATASET_URL, zip_path):
        sys.exit(1)
    
    # Extract the dataset
    if not extract_zip(zip_path, DOWNLOAD_DIR):
        sys.exit(1)
    
    # Verify the dataset
    if verify_dataset(dataset_path):
        print("✅ Dataset verification successful!")
        
        # Clean up zip file
        try:
            os.remove(zip_path)
            print("🗑️  Cleaned up zip file")
        except:
            print("⚠️  Could not remove zip file")
        
        print(f"🎯 Dataset ready at: {dataset_path}")
        print("\n📝 Next steps:")
        print("1. Update DATASET_PATH in train_resnet50.py to:")
        print(f"   DATASET_PATH = \"{dataset_path}\"")
        print("2. Run the training script")
        
    else:
        print("❌ Dataset verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()