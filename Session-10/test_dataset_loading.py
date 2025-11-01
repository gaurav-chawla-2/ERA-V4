#!/usr/bin/env python3
"""
Test Dataset Loading Script
===========================

This script tests whether the Hugging Face ImageNet dataset can be loaded correctly.
Use this to debug dataset loading issues.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dataset_path():
    """Test if the dataset path exists and what files are in it"""
    dataset_path = "/data/imagenet/full_dataset"
    
    print("🔍 Testing dataset path...")
    print(f"📁 Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset path does not exist!")
        print("💡 Make sure you've run the download script and the dataset is saved to the correct location.")
        return False
    
    print("✅ Dataset path exists")
    
    try:
        files = os.listdir(dataset_path)
        print(f"📂 Files in dataset directory ({len(files)} total):")
        for i, file in enumerate(files[:20]):  # Show first 20 files
            file_path = os.path.join(dataset_path, file)
            if os.path.isdir(file_path):
                print(f"   📁 {file}/")
            else:
                print(f"   📄 {file}")
        if len(files) > 20:
            print(f"   ... and {len(files) - 20} more files")
        
        # Check for Hugging Face dataset indicators
        has_dataset_dict = 'dataset_dict.json' in files
        has_arrow_files = any(f.endswith('.arrow') for f in files)
        has_subdirs_with_arrow = any(
            os.path.isdir(os.path.join(dataset_path, f)) and 
            any(sf.endswith('.arrow') for sf in os.listdir(os.path.join(dataset_path, f)))
            for f in files if os.path.isdir(os.path.join(dataset_path, f))
        )
        
        print(f"\n🔍 Hugging Face dataset indicators:")
        print(f"   📄 dataset_dict.json: {'✅' if has_dataset_dict else '❌'}")
        print(f"   🏹 .arrow files: {'✅' if has_arrow_files else '❌'}")
        print(f"   📁 subdirs with .arrow: {'✅' if has_subdirs_with_arrow else '❌'}")
        
        is_hf_dataset = has_dataset_dict or has_arrow_files or has_subdirs_with_arrow
        print(f"\n🎯 Detected as Hugging Face dataset: {'✅' if is_hf_dataset else '❌'}")
        
        return is_hf_dataset
        
    except Exception as e:
        print(f"❌ Error reading dataset directory: {e}")
        return False

def test_datasets_library():
    """Test if the datasets library is available and working"""
    print("\n🔍 Testing Hugging Face datasets library...")
    
    try:
        from datasets import load_from_disk
        print("✅ datasets library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import datasets library: {e}")
        print("💡 Install with: pip install datasets")
        return False

def test_dataset_loading():
    """Test loading the actual dataset"""
    print("\n🔍 Testing dataset loading...")
    
    dataset_path = "/data/imagenet/full_dataset"
    
    try:
        from datasets import load_from_disk
        
        print(f"📁 Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        
        print("✅ Dataset loaded successfully!")
        print(f"📊 Dataset splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"   📂 {split_name}: {len(split_data)} samples")
            
            # Show sample data structure
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"      🔍 Sample keys: {list(sample.keys())}")
                if 'label' in sample:
                    print(f"      🏷️  Label type: {type(sample['label'])}")
                if 'image' in sample:
                    print(f"      🖼️  Image type: {type(sample['image'])}")
        
        # Test class information
        if 'train' in dataset:
            train_data = dataset['train']
            if hasattr(train_data.features, 'label') and hasattr(train_data.features['label'], 'names'):
                class_names = train_data.features['label'].names
                print(f"🏷️  Number of classes: {len(class_names)}")
                print(f"   First 5 classes: {class_names[:5]}")
            else:
                print("⚠️  No class names found in dataset features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def main():
    """Main test function"""
    print("🚀 Dataset Loading Test")
    print("=" * 50)
    
    # Test 1: Dataset path
    path_ok = test_dataset_path()
    
    # Test 2: Datasets library
    library_ok = test_datasets_library()
    
    # Test 3: Dataset loading (only if previous tests pass)
    if path_ok and library_ok:
        loading_ok = test_dataset_loading()
    else:
        loading_ok = False
        print("\n⏭️  Skipping dataset loading test due to previous failures")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   📁 Dataset path: {'✅' if path_ok else '❌'}")
    print(f"   📚 Datasets library: {'✅' if library_ok else '❌'}")
    print(f"   📊 Dataset loading: {'✅' if loading_ok else '❌'}")
    
    if path_ok and library_ok and loading_ok:
        print("\n🎉 All tests passed! Your dataset should work with the training script.")
    else:
        print("\n❌ Some tests failed. Please fix the issues above before training.")
        
        if not path_ok:
            print("\n💡 Dataset path issues:")
            print("   - Make sure you've run your download script")
            print("   - Check that the dataset was saved to /data/imagenet/full_dataset")
            print("   - Verify the path in your download script matches the training script")
        
        if not library_ok:
            print("\n💡 Library issues:")
            print("   - Install the datasets library: pip install datasets")
            print("   - Make sure you're using the correct Python environment")
        
        if not loading_ok and path_ok and library_ok:
            print("\n💡 Dataset loading issues:")
            print("   - The dataset might be corrupted")
            print("   - Try re-running the download script")
            print("   - Check disk space and permissions")

if __name__ == "__main__":
    main()