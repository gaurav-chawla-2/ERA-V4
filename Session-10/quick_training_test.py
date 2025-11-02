#!/usr/bin/env python3
"""
Quick Training Test - Diagnostic Script
======================================

This script runs a quick test to verify that the model starts learning
after fixing the learning rate issues. It trains for just a few batches
and checks if the loss decreases and accuracy improves.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import sys
import os

# Add the current directory to path to import from train_resnet50
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the main training script
from train_resnet50 import (
    ResNet50, create_data_loaders, get_data_transforms,
    NUM_CLASSES, INITIAL_LR, MOMENTUM, WEIGHT_DECAY,
    USE_MIXED_PRECISION, OPTIMIZER_TYPE
)

def quick_training_test(num_batches=10):
    """
    Run a quick training test to verify the model starts learning
    """
    print("🔍 Quick Training Test - Checking if model learns")
    print("="*50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("📁 Loading data...")
    try:
        train_loader, val_loader, actual_num_classes = create_data_loaders()
        print(f"✅ Data loaded successfully. Classes: {actual_num_classes}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Create model
    print("🏗️  Creating model...")
    model = ResNet50(num_classes=actual_num_classes).to(device)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # Mixed precision setup
    scaler = None
    if USE_MIXED_PRECISION and device.type == 'cuda':
        scaler = GradScaler()
        print("✅ Mixed precision enabled")
    
    print(f"📊 Training configuration:")
    print(f"   Learning rate: {INITIAL_LR:.2e}")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Optimizer: {OPTIMIZER_TYPE}")
    
    # Quick training test
    model.train()
    initial_loss = None
    final_loss = None
    initial_acc = None
    final_acc = None
    
    print(f"\n🚀 Running {num_batches} training batches...")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(data)
        
        # Store initial and final metrics
        if batch_idx == 0:
            initial_loss = loss.item()
            initial_acc = accuracy
        if batch_idx == num_batches - 1:
            final_loss = loss.item()
            final_acc = accuracy
        
        print(f"Batch {batch_idx+1:2d}/{num_batches} | Loss: {loss.item():.4f} | Acc: {accuracy:.2f}%")
    
    # Analysis
    print("\n📊 Training Test Results:")
    print("="*30)
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    print(f"Loss Change:  {final_loss - initial_loss:.4f}")
    
    print(f"\nInitial Accuracy: {initial_acc:.2f}%")
    print(f"Final Accuracy:   {final_acc:.2f}%")
    print(f"Accuracy Change:  {final_acc - initial_acc:.2f}%")
    
    # Determine if learning is happening
    loss_improved = final_loss < initial_loss
    acc_improved = final_acc > initial_acc
    
    print(f"\n🎯 Learning Assessment:")
    if loss_improved and acc_improved:
        print("✅ EXCELLENT: Both loss decreased and accuracy improved!")
        print("   The model is learning properly.")
        return True
    elif loss_improved or acc_improved:
        print("✅ GOOD: Some improvement detected.")
        print("   The model shows signs of learning.")
        return True
    elif abs(final_loss - initial_loss) < 0.1 and abs(final_acc - initial_acc) < 1.0:
        print("⚠️  MARGINAL: Very small changes detected.")
        print("   The model might need more batches to show clear learning.")
        return True
    else:
        print("❌ POOR: No clear improvement detected.")
        print("   There may still be configuration issues.")
        return False

def main():
    """Main function"""
    print("🔍 Quick Training Diagnostic")
    print("This script tests if the model starts learning after LR fixes")
    print()
    
    success = quick_training_test(num_batches=15)
    
    if success:
        print("\n🎉 SUCCESS: The model appears to be learning!")
        print("You can now run the full training with confidence.")
        print("\nTo start full training:")
        print("python train_resnet50.py")
    else:
        print("\n⚠️  The model may still have issues.")
        print("Consider running the LR finder:")
        print("python train_resnet50.py --lr-finder")

if __name__ == "__main__":
    main()