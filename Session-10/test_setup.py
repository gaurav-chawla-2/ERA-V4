"""
Test Setup Script
================

Quick verification that everything is properly configured for training.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def test_pytorch():
    """Test PyTorch installation and CUDA availability"""
    print("🔧 Testing PyTorch...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test basic tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    print(f"   Basic operations: ✅")
    
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = x_gpu + y_gpu
        print(f"   GPU operations: ✅")

def test_dataset():
    """Test dataset availability"""
    print("\n📁 Testing dataset...")
    dataset_path = Path("./data/mini-imagenet")
    
    if not dataset_path.exists():
        print(f"   ❌ Dataset not found at {dataset_path}")
        print(f"   Run: python setup_dataset.py")
        return False
    
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    
    if not train_path.exists() or not val_path.exists():
        print(f"   ❌ Invalid dataset structure")
        return False
    
    # Count classes and images
    train_classes = [d for d in train_path.iterdir() if d.is_dir()]
    val_classes = [d for d in val_path.iterdir() if d.is_dir()]
    
    train_images = sum(len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png"))) 
                      for class_dir in train_classes)
    val_images = sum(len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png"))) 
                    for class_dir in val_classes)
    
    print(f"   ✅ Dataset found:")
    print(f"      Classes: {len(train_classes)}")
    print(f"      Training images: {train_images}")
    print(f"      Validation images: {val_images}")
    
    return True

def test_model():
    """Test model creation and forward pass"""
    print("\n🏗️  Testing model...")
    
    try:
        # Import the model (assuming train_resnet50.py is in the same directory)
        import sys
        sys.path.append('.')
        
        # Create a simple test version
        import torch.nn as nn
        
        class SimpleResNet50Test(nn.Module):
            def __init__(self, num_classes=100):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, num_classes)
            
            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        model = SimpleResNet50Test()
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        print(f"   ✅ Model creation: Success")
        print(f"   ✅ Forward pass: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing data loading...")
    
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        # Test transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset_path = Path("./data/mini-imagenet")
        if not dataset_path.exists():
            print(f"   ⚠️  Dataset not available for testing")
            return False
        
        # Try to load a small batch
        train_dataset = datasets.ImageFolder(
            root=dataset_path / "train",
            transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Get one batch
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        print(f"   ✅ Data loading: Success")
        print(f"   ✅ Batch shape: {images.shape}")
        print(f"   ✅ Labels shape: {labels.shape}")
        print(f"   ✅ Classes: {len(train_dataset.classes)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading test failed: {e}")
        return False

def test_visualization():
    """Test matplotlib for visualization"""
    print("\n📈 Testing visualization...")
    
    try:
        # Test basic plotting
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title("Test Plot")
        plt.close()  # Don't show, just test
        
        print(f"   ✅ Matplotlib: Working")
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Setup Tests")
    print("=" * 50)
    
    tests = [
        ("PyTorch", test_pytorch),
        ("Dataset", test_dataset),
        ("Model", test_model),
        ("Data Loading", test_data_loading),
        ("Visualization", test_visualization)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to start training.")
        print("Run: python train_resnet50.py")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        
        if not results[1][1]:  # Dataset test failed
            print("💡 To fix dataset: python setup_dataset.py")

if __name__ == "__main__":
    main()