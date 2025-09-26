"""
Training script for Session-6 MNIST models
Implements progressive training strategy across Model_1, Model_2, and Model_3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import all models
from model1 import Model_1, count_parameters
from model2 import Model_2
from model3 import Model_3

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_data_loaders(batch_size=128, validation_split=0.1):
    """Create optimized data loaders for 99.4% accuracy target"""
    
    # Advanced training transforms for Model_3
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-6.0, 6.0), fill=(0,)),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.08, 0.08),
            scale=(0.94, 1.06),
            shear=2
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transforms (no augmentation)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transforms)
    
    # Split training data
    train_size = int((1 - validation_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create validation dataset with test transforms
    val_dataset.dataset = datasets.MNIST('../data', train=True, download=False, transform=test_transforms)
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset.dataset, val_indices)
    
    # Create data loaders
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, device, train_loader, optimizer, epoch, scheduler=None):
    """Train for one epoch"""
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    correct = 0
    processed = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        total_loss += loss.item()
        
        pbar.set_description(f'Epoch {epoch}: Loss={loss.item():.4f} Acc={100*correct/processed:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / processed
    return accuracy, avg_loss

def test_model(model, device, test_loader, dataset_name="Test"):
    """Test model performance"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    
    print(f'{dataset_name} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, test_loss

def train_model(model, model_name, device, train_loader, val_loader, test_loader, epochs=15):
    """Train a specific model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*50}")
    
    # Optimized training setup for each model
    if model_name == "Model_1":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=epochs, 
            steps_per_epoch=len(train_loader), pct_start=0.2
        )
    elif model_name == "Model_2":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.015, epochs=epochs, 
            steps_per_epoch=len(train_loader), pct_start=0.15
        )
    else:  # Model_3
        optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=3e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.012, epochs=epochs, 
            steps_per_epoch=len(train_loader), pct_start=0.1
        )
    
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    best_val_acc = 0
    target_achieved_epoch = None
    
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_epoch(model, device, train_loader, optimizer, epoch, scheduler)
        val_acc, val_loss = test_model(model, device, val_loader, "Validation")
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model_name.lower()}_best.pth')
        
        # Check if target achieved
        if val_acc >= 99.4 and target_achieved_epoch is None:
            target_achieved_epoch = epoch
            print(f"🎯 Target 99.4% achieved at epoch {epoch}!")
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    # Final test
    test_acc, test_loss = test_model(model, device, test_loader, "Test")
    
    # Check consistency in last 3 epochs for Model_3
    if model_name == "Model_3" and len(val_accuracies) >= 3:
        last_3_epochs = val_accuracies[-3:]
        consistent_99_4 = all(acc >= 99.4 for acc in last_3_epochs)
        print(f"Consistent 99.4%+ in last 3 epochs: {consistent_99_4}")
        print(f"Last 3 epochs accuracies: {last_3_epochs}")
    
    return {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc,
        'target_achieved_epoch': target_achieved_epoch,
        'parameters': count_parameters(model)
    }

def plot_results(results_dict):
    """Plot training results for all models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        epochs = range(1, len(results['train_accuracies']) + 1)
        
        # Accuracy plot
        axes[0, idx].plot(epochs, results['train_accuracies'], 'b-', label='Train', linewidth=2)
        axes[0, idx].plot(epochs, results['val_accuracies'], 'r-', label='Validation', linewidth=2)
        axes[0, idx].axhline(y=99.4, color='g', linestyle='--', alpha=0.7, label='Target (99.4%)')
        axes[0, idx].set_title(f'{model_name} - Accuracy\nParams: {results["parameters"]:,}')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Accuracy (%)')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1, idx].plot(epochs, results['train_losses'], 'b-', label='Train', linewidth=2)
        axes[1, idx].plot(epochs, results['val_losses'], 'r-', label='Validation', linewidth=2)
        axes[1, idx].set_title(f'{model_name} - Loss')
        axes[1, idx].set_xlabel('Epoch')
        axes[1, idx].set_ylabel('Loss')
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('session6_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    set_seed(42)
    
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=128)
    
    # Initialize models
    models = {
        'Model_1': Model_1().to(device),
        'Model_2': Model_2().to(device),
        'Model_3': Model_3().to(device)
    }
    
    # Train all models
    results = {}
    for model_name, model in models.items():
        results[model_name] = train_model(
            model, model_name, device, train_loader, val_loader, test_loader, epochs=15
        )
    
    # Plot results
    plot_results(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Best Validation Accuracy: {result['best_val_acc']:.2f}%")
        print(f"  Final Test Accuracy: {result['final_test_acc']:.2f}%")
        if result['target_achieved_epoch']:
            print(f"  Target 99.4% achieved at epoch: {result['target_achieved_epoch']}")
        else:
            print(f"  Target 99.4% not achieved")
        print()

if __name__ == "__main__":
    main()