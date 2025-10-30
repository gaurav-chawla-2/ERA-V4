"""
Verification Script for Tiny-ImageNet-200 Setup
==============================================

This script verifies that the dataset is properly downloaded and
the training environment is ready.
"""

import os
import sys
from pathlib import Path

def check_dataset():
    """Check if the dataset is properly downloaded and structured"""
    # Read dataset path from training script
    dataset_path = None
    try:
        with open('train_resnet50.py', 'r') as f:
            for line in f:
                if line.strip().startswith('DATASET_PATH ='):
                    dataset_path = line.split('=', 1)[1].strip().strip('"\'')
                    break
    except:
        pass
    
    if not dataset_path:
        print("❌ Could not determine dataset path from train_resnet50.py")
        return False
    
    print("🔍 Checking dataset setup...")
    print(f"📁 Looking for dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        if 'tiny-imagenet' in dataset_path.lower():
            print("💡 Run: sudo python3 download_tiny_imagenet.py")
        else:
            print("💡 Run: python3 download_imagenet.py (for instructions)")
        return False
    
    # Detect dataset type
    is_tiny_imagenet = 'tiny-imagenet' in dataset_path.lower()
    dataset_type = "Tiny-ImageNet" if is_tiny_imagenet else "Full ImageNet"
    expected_classes = 200 if is_tiny_imagenet else 1000
    
    print(f"🔍 Detected dataset type: {dataset_type}")
    
    # Check required directories
    if is_tiny_imagenet:
        required_dirs = ['train', 'val', 'test']
    else:
        required_dirs = ['train', 'val']
        # Check for alternative validation directory names
        if not os.path.exists(os.path.join(dataset_path, 'val')):
            alt_names = ['validation', 'valid', 'test']
            for alt in alt_names:
                if os.path.exists(os.path.join(dataset_path, alt)):
                    required_dirs = ['train', alt]
                    break
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ Found {dir_name} directory")
            
            if dir_name == 'train':
                # Count training classes
                try:
                    classes = [d for d in os.listdir(dir_path) 
                              if os.path.isdir(os.path.join(dir_path, d))]
                    print(f"   📊 Training classes: {len(classes)}")
                    
                    if len(classes) != expected_classes:
                        print(f"⚠️  Expected {expected_classes} classes, found {len(classes)}")
                except Exception as e:
                    print(f"⚠️  Error reading training directory: {e}")
                
        else:
            print(f"❌ Missing {dir_name} directory")
            return False
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n🔍 Checking Python dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'matplotlib',
        'numpy',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_training_script():
    """Check if the training script is properly configured"""
    print("\n🔍 Checking training script configuration...")
    
    script_path = "train_resnet50.py"
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    print(f"✅ Found training script: {script_path}")
    
    # Check if dataset path is correctly configured
    with open(script_path, 'r') as f:
        content = f.read()
        if '/opt/dlami/nvme/tiny-imagenet-200' in content:
            print("✅ Dataset path correctly configured")
        else:
            print("⚠️  Dataset path may not be correctly configured")
            print("💡 Check DATASET_PATH in train_resnet50.py")
    
    return True

def main():
    """Main verification function"""
    print("🚀 Tiny-ImageNet-200 Setup Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check dataset
    if not check_dataset():
        all_good = False
    
    # Check dependencies
    if not check_dependencies():
        all_good = False
    
    # Check training script
    if not check_training_script():
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! You're ready to start training!")
        print("🚀 Run: python3 train_resnet50.py")
    else:
        print("❌ Some issues found. Please fix them before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()