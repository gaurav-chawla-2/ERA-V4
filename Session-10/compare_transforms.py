#!/usr/bin/env python3
"""
Transform Performance Comparison
===============================

This script compares the performance of Albumentations vs torchvision transforms
with the optimized dataloader to help you choose the best approach.
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from the training script
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_dataloader(dataset_path, use_albumentations=True, batch_size=64, num_workers=4):
    """Create a test dataloader with specified transform type"""
    try:
        from datasets import load_from_disk
        
        # Load dataset
        dataset = load_from_disk(dataset_path)
        
        if use_albumentations:
            # Albumentations transforms
            transform = A.Compose([
                A.Resize(height=256, width=256),
                A.RandomCrop(height=224, width=224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            def preprocess_albumentations(examples):
                if isinstance(examples['image'], list):
                    images = []
                    for img in examples['image']:
                        img_array = np.array(img.convert('RGB'))
                        transformed = transform(image=img_array)
                        images.append(transformed['image'])
                    examples['pixel_values'] = images
                else:
                    img_array = np.array(examples['image'].convert('RGB'))
                    transformed = transform(image=img_array)
                    examples['pixel_values'] = transformed['image']
                return examples
            
            train_data = dataset['train'].with_transform(preprocess_albumentations)
            
        else:
            # Torchvision transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            def preprocess_torchvision(examples):
                if isinstance(examples['image'], list):
                    images = []
                    for img in examples['image']:
                        img = transform(img.convert('RGB'))
                        images.append(img)
                    examples['pixel_values'] = images
                else:
                    img = transform(examples['image'].convert('RGB'))
                    examples['pixel_values'] = img
                return examples
            
            train_data = dataset['train'].with_transform(preprocess_torchvision)
        
        # Create DataLoader
        dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        
        return dataloader
        
    except Exception as e:
        print(f"❌ Error creating dataloader: {e}")
        return None

def benchmark_transforms(dataset_path, num_batches=30):
    """Benchmark both transform approaches"""
    print("🔬 Transform Performance Comparison")
    print("=" * 50)
    
    results = {}
    
    # Test Albumentations
    print("\n1️⃣  Testing Albumentations (superior augmentation quality):")
    albu_loader = create_test_dataloader(dataset_path, use_albumentations=True)
    
    if albu_loader:
        # Warmup
        for i, (images, labels) in enumerate(albu_loader):
            if i >= 2:
                break
        
        # Benchmark
        start_time = time.time()
        for i, (images, labels) in enumerate(albu_loader):
            if i >= num_batches:
                break
        end_time = time.time()
        
        albu_time = end_time - start_time
        albu_time_per_batch = albu_time / num_batches
        batch_size = next(iter(albu_loader))[0].shape[0]
        albu_throughput = batch_size / albu_time_per_batch
        
        results['albumentations'] = {
            'time_per_batch': albu_time_per_batch,
            'throughput': albu_throughput,
            'total_time': albu_time
        }
        
        print(f"   ⏱️  Time per batch: {albu_time_per_batch:.3f}s")
        print(f"   📈 Throughput: {albu_throughput:.1f} samples/sec")
    
    # Test Torchvision
    print("\n2️⃣  Testing Torchvision (optimized performance):")
    torch_loader = create_test_dataloader(dataset_path, use_albumentations=False)
    
    if torch_loader:
        # Warmup
        for i, (images, labels) in enumerate(torch_loader):
            if i >= 2:
                break
        
        # Benchmark
        start_time = time.time()
        for i, (images, labels) in enumerate(torch_loader):
            if i >= num_batches:
                break
        end_time = time.time()
        
        torch_time = end_time - start_time
        torch_time_per_batch = torch_time / num_batches
        batch_size = next(iter(torch_loader))[0].shape[0]
        torch_throughput = batch_size / torch_time_per_batch
        
        results['torchvision'] = {
            'time_per_batch': torch_time_per_batch,
            'throughput': torch_throughput,
            'total_time': torch_time
        }
        
        print(f"   ⏱️  Time per batch: {torch_time_per_batch:.3f}s")
        print(f"   📈 Throughput: {torch_throughput:.1f} samples/sec")
    
    # Comparison
    if 'albumentations' in results and 'torchvision' in results:
        print(f"\n📊 Performance Comparison:")
        print(f"=" * 50)
        
        albu = results['albumentations']
        torch = results['torchvision']
        
        speedup = albu['time_per_batch'] / torch['time_per_batch']
        throughput_improvement = torch['throughput'] / albu['throughput']
        
        print(f"🎨 Albumentations:")
        print(f"   ⏱️  Time per batch: {albu['time_per_batch']:.3f}s")
        print(f"   📈 Throughput: {albu['throughput']:.1f} samples/sec")
        print(f"   ✅ Superior augmentation quality")
        print(f"   ✅ More diverse augmentations (rotation, cutout, etc.)")
        
        print(f"\n🚀 Torchvision:")
        print(f"   ⏱️  Time per batch: {torch['time_per_batch']:.3f}s")
        print(f"   📈 Throughput: {torch['throughput']:.1f} samples/sec")
        print(f"   ✅ {speedup:.1f}x faster than Albumentations")
        print(f"   ✅ {throughput_improvement:.1f}x higher throughput")
        
        print(f"\n💡 Recommendations:")
        if speedup > 1.5:
            print(f"   🏃 Torchvision is significantly faster ({speedup:.1f}x)")
            print(f"   💭 Consider torchvision for faster training iterations")
            print(f"   💭 Use Albumentations for final model training")
        else:
            print(f"   ⚖️  Performance difference is minimal ({speedup:.1f}x)")
            print(f"   💭 Albumentations recommended for better augmentation quality")
        
        print(f"\n🎯 Best Practice:")
        print(f"   • Development/debugging: Use torchvision for faster iterations")
        print(f"   • Final training: Use Albumentations for better accuracy")
        print(f"   • Production: Choose based on your speed vs quality requirements")

def main():
    """Main comparison function"""
    dataset_path = "/lambda/nfs/ERAv4S09/imagenet/full_dataset"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("💡 Please update the dataset_path variable to point to your dataset")
        return
    
    try:
        benchmark_transforms(dataset_path, num_batches=25)
        
        print(f"\n🔧 To switch between approaches in train_resnet50.py:")
        print(f"   • Set USE_ALBUMENTATIONS = True for Albumentations")
        print(f"   • Set USE_ALBUMENTATIONS = False for torchvision")
        print(f"   • Set OPTIMIZE_FOR_SPEED = True for additional optimizations")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")

if __name__ == "__main__":
    main()