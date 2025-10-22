"""
Test script to verify the ResNet CIFAR-100 setup
Runs basic tests to ensure everything is working correctly
"""

import torch
import sys
import os

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from config import Config
        from resnet import ResNet56, count_parameters
        from data_loader import get_cifar100_loaders
        from trainer import Trainer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model():
    """Test model creation and forward pass"""
    print("\nTesting model...")
    try:
        from resnet import ResNet56, count_parameters
        
        model = ResNet56(num_classes=100)
        num_params = count_parameters(model)
        
        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        
        print(f"✅ Model created successfully")
        print(f"   Parameters: {num_params:,}")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        
        assert y.shape == (4, 100), f"Expected output shape (4, 100), got {y.shape}"
        print("✅ Forward pass test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data_loader():
    """Test data loading"""
    print("\nTesting data loader...")
    try:
        from config import Config
        from data_loader import get_cifar100_loaders
        
        config = Config()
        config.BATCH_SIZE = 4  # Small batch for testing
        config.NUM_WORKERS = 0  # Avoid multiprocessing issues in testing
        
        train_loader, val_loader, test_loader = get_cifar100_loaders(config)
        
        # Test a batch from each loader
        for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            data, target = next(iter(loader))
            print(f"   {name}: data shape {data.shape}, target shape {target.shape}")
            assert data.shape[1:] == (3, 32, 32), f"Expected data shape (*, 3, 32, 32), got {data.shape}"
            assert target.max() < 100, f"Target values should be < 100, got max {target.max()}"
        
        print("✅ Data loader test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_trainer_setup():
    """Test trainer initialization"""
    print("\nTesting trainer setup...")
    try:
        from config import Config
        from resnet import ResNet56
        from data_loader import get_cifar100_loaders
        from trainer import Trainer
        
        config = Config()
        config.BATCH_SIZE = 4
        config.NUM_WORKERS = 0
        config.EPOCHS = 1  # Just for testing
        
        model = ResNet56(num_classes=100)
        train_loader, val_loader, test_loader = get_cifar100_loaders(config)
        
        trainer = Trainer(model, config, train_loader, val_loader, test_loader)
        
        print("✅ Trainer setup successful")
        print(f"   Device: {trainer.device}")
        print(f"   Optimizer: {type(trainer.optimizer).__name__}")
        print(f"   Scheduler: {type(trainer.scheduler).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Trainer setup failed: {e}")
        return False

def test_system_requirements():
    """Test system requirements"""
    print("\nTesting system requirements...")
    
    # Check PyTorch version
    torch_version = torch.__version__
    print(f"   PyTorch version: {torch_version}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   CUDA device: {torch.cuda.get_device_name()}")
    
    # Check memory
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU memory: {gpu_memory:.1f} GB")
        if gpu_memory < 2:
            print("   ⚠️  Warning: Low GPU memory, consider reducing batch size")
    
    print("✅ System requirements check completed")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("ResNet CIFAR-100 Setup Test")
    print("=" * 60)
    
    tests = [
        test_system_requirements,
        test_imports,
        test_model,
        test_data_loader,
        test_trainer_setup,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to start training.")
        print("\nTo start training, run:")
        print("   python train.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())