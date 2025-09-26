"""
Ultra-Optimized Training Script for Session-6 Models
Enhanced training strategies to achieve 99.4% accuracy target

Key Enhancements:
- Advanced data augmentation strategies
- Optimized learning rate scheduling
- Enhanced regularization techniques
- Model ensemble capabilities
- Advanced optimization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Import ultra-optimized models
from model1_ultra_optimized import Model_1
from model2_ultra_optimized import Model_2  
from model3_ultra_optimized import Model_3

def create_enhanced_data_loaders(batch_size=128):
    """Create enhanced data loaders with advanced augmentation strategies"""
    
    # Enhanced training transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation((-8.0, 8.0), fill=(0,)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.92, 1.08),
            shear=(-5, 5, -5, 5),
            fill=(0,)
        ),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
    ])
    
    # Test transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

def train_with_enhanced_techniques(model, device, train_loader, optimizer, scheduler, epoch, criterion):
    """Enhanced training function with advanced techniques"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Label smoothing for better generalization
        if hasattr(criterion, 'label_smoothing'):
            loss = F.cross_entropy(output, target, label_smoothing=0.05)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Statistics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })
    
    return train_loss / len(train_loader), 100. * correct / total

def test_with_tta(model, device, test_loader):
    """Test function with Test Time Augmentation for better accuracy"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # TTA transformations
    tta_transforms = [
        transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation((-2, 2)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation((2, -2)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ]
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Standard prediction
            output = model(data)
            
            # TTA predictions (optional, can be enabled for final evaluation)
            # tta_outputs = []
            # for transform in tta_transforms:
            #     tta_data = torch.stack([transform(img.cpu()) for img in data]).to(device)
            #     tta_outputs.append(model(tta_data))
            # 
            # # Average TTA predictions
            # if tta_outputs:
            #     output = torch.stack([output] + tta_outputs).mean(0)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    
    return test_loss, accuracy

def train_model(model_class, model_name, epochs=15, lr=0.003):
    """Enhanced training function for a specific model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model and count parameters
    model = model_class().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create data loaders
    train_loader, test_loader = create_enhanced_data_loaders(batch_size=128)
    
    # Enhanced optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    
    # Advanced learning rate scheduling
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    
    # Enhanced loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    best_accuracy = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_with_enhanced_techniques(
            model, device, train_loader, optimizer, scheduler, epoch, criterion
        )
        
        # Test
        test_loss, test_acc = test_with_tta(model, device, test_loader)
        
        # Record history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Track best accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), f'./models/best_{model_name.lower()}.pth')
        
        print(f'Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Early achievement check
        if test_acc >= 99.4:
            print(f"🎯 TARGET ACHIEVED! 99.4% accuracy reached at epoch {epoch}")
    
    print(f"\nBest Test Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
    print(f"Target 99.4% {'✅ ACHIEVED' if best_accuracy >= 99.4 else '❌ NOT ACHIEVED'}")
    
    return {
        'model_name': model_name,
        'total_params': total_params,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'final_accuracy': test_accuracies[-1],
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'target_achieved': best_accuracy >= 99.4
    }

def plot_enhanced_results(results_list):
    """Enhanced plotting function for training results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ultra-Optimized Models Training Results', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, results in enumerate(results_list):
        color = colors[i]
        model_name = results['model_name']
        
        # Training Loss
        axes[0, 0].plot(results['train_losses'], label=f'{model_name}', color=color, linewidth=2)
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test Loss
        axes[0, 1].plot(results['test_losses'], label=f'{model_name}', color=color, linewidth=2)
        axes[0, 1].set_title('Test Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training Accuracy
        axes[0, 2].plot(results['train_accuracies'], label=f'{model_name}', color=color, linewidth=2)
        axes[0, 2].set_title('Training Accuracy', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Test Accuracy
        axes[1, 0].plot(results['test_accuracies'], label=f'{model_name}', color=color, linewidth=2)
        axes[1, 0].set_title('Test Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].axhline(y=99.4, color='red', linestyle='--', alpha=0.7, label='Target 99.4%')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Model Comparison - Parameters vs Best Accuracy
    model_names = [r['model_name'] for r in results_list]
    params = [r['total_params'] for r in results_list]
    best_accs = [r['best_accuracy'] for r in results_list]
    
    bars = axes[1, 1].bar(model_names, params, color=colors, alpha=0.7)
    axes[1, 1].set_title('Model Parameters', fontweight='bold')
    axes[1, 1].set_ylabel('Parameters')
    axes[1, 1].axhline(y=8000, color='red', linestyle='--', alpha=0.7, label='Limit: 8000')
    axes[1, 1].legend()
    
    # Add parameter count labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 100,
                       f'{param:,}', ha='center', va='bottom', fontweight='bold')
    
    # Best Accuracy Comparison
    bars = axes[1, 2].bar(model_names, best_accs, color=colors, alpha=0.7)
    axes[1, 2].set_title('Best Test Accuracy', fontweight='bold')
    axes[1, 2].set_ylabel('Accuracy (%)')
    axes[1, 2].axhline(y=99.4, color='red', linestyle='--', alpha=0.7, label='Target: 99.4%')
    axes[1, 2].set_ylim(98.5, 100)
    axes[1, 2].legend()
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./models/ultra_optimized_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🚀 Starting Ultra-Optimized Training for Session-6 Models")
    print("Target: Achieve 99.4% accuracy with <8000 parameters")
    
    # Create models directory
    os.makedirs('./models', exist_ok=True)
    
    # Define models to train
    models_to_train = [
        (Model_1, "Ultra-Optimized Model_1"),
        (Model_2, "Ultra-Optimized Model_2"), 
        (Model_3, "Ultra-Optimized Model_3")
    ]
    
    # Train all models
    all_results = []
    
    for model_class, model_name in models_to_train:
        results = train_model(model_class, model_name, epochs=15, lr=0.003)
        all_results.append(results)
    
    # Plot results
    plot_enhanced_results(all_results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("ULTRA-OPTIMIZED TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for results in all_results:
        status = "✅ ACHIEVED" if results['target_achieved'] else "❌ NOT ACHIEVED"
        param_status = "✅ WITHIN LIMIT" if results['total_params'] <= 8000 else "❌ EXCEEDS LIMIT"
        
        print(f"\n{results['model_name']}:")
        print(f"  Parameters: {results['total_params']:,} {param_status}")
        print(f"  Best Accuracy: {results['best_accuracy']:.2f}% (Epoch {results['best_epoch']})")
        print(f"  Final Accuracy: {results['final_accuracy']:.2f}%")
        print(f"  Target 99.4%: {status}")
    
    # Check if any model achieved the target
    achieved_models = [r for r in all_results if r['target_achieved']]
    if achieved_models:
        print(f"\n🎯 SUCCESS! {len(achieved_models)} model(s) achieved the 99.4% target!")
    else:
        print(f"\n⚠️  None of the models achieved the 99.4% target. Further optimization needed.")

if __name__ == '__main__':
    main()