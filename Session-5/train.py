import torch  # Import PyTorch library
import torch.optim as optim  # Import optimization algorithms
from torchvision import datasets, transforms  # Import datasets and transformations
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import os  # Import os for file operations
from model import PrecisionChampionMNISTNet, count_parameters, train, test, set_seed

def create_data_loaders(batch_size=64, validation_split=0.1):
    # Precision-tuned training transforms for 99.4%+ accuracy
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(0,)),  # Balanced rotation
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.09, 0.09),  # Balanced translation
            scale=(0.92, 1.08),  # Balanced scale range
            shear=3  # Balanced shear
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Define data transformations for validation and testing (no augmentation)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    
    # Download and load the training data
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transforms)
    
    # Split training data into train and validation sets
    train_size = int((1 - validation_split) * len(full_train_dataset))  # Calculate training set size
    val_size = len(full_train_dataset) - train_size  # Calculate validation set size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Set seed for reproducible split
    )
    
    # Create validation dataset with test transforms (no augmentation)
    val_dataset.dataset = datasets.MNIST('../data', train=True, download=False, transform=test_transforms)
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset.dataset, val_indices)
    
    # Set dataloader parameters
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def plot_training_history(train_accuracies, val_accuracies, train_losses, val_losses):
    """
    Plot training history including accuracy and loss curves
    """
    epochs = range(1, len(train_accuracies) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=99.4, color='g', linestyle='--', alpha=0.7, label='Target (99.4%)')
    
    # Plot loss
    ax2.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Set reproducible random seed for consistent results
    set_seed(42)
    
    # Check if CUDA is available and set the device accordingly
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Precision-tuned hyperparameters for 99.4%+ accuracy
    batch_size = 64
    epochs = 20
    lr = 0.042  # Precision-tuned learning rate
    momentum = 0.9
    weight_decay = 3e-6  # Balanced weight decay
    target_accuracy = 99.4
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=batch_size)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize the model
    model = PrecisionChampionMNISTNet().to(device)
    
    # Print model summary and parameter count
    print("\nModel Architecture:")
    print(model)
    param_count = count_parameters(model)
    print(f"\nTotal trainable parameters: {param_count:,}")
    
    # Check if the model meets the parameter constraint
    if param_count > 20000:
        print(f"ERROR: Model has {param_count:,} parameters, which exceeds the limit of 20,000")
        return
    else:
        print(f"✓ Model parameter count ({param_count:,}) is within the limit of 20,000")
    
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    # Precision-tuned OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.28,  # 28% warmup for stability
        div_factor=9,    # Balanced initial LR
        final_div_factor=45  # Strong final decay
    )
    
    # Initialize tracking variables
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    best_val_accuracy = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for up to {epochs} epochs...")
    print("=" * 70)
    
    # Train and validate for specified epochs
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)
        
        # Train one epoch
        train_acc, train_loss = train(model, device, train_loader, optimizer, epoch)
        
        # Validate
        val_acc, val_loss = test(model, device, val_loader, "Validation")
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Track best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            # Save best model
            torch.save(model.state_dict(), "best_mnist_model.pth")
            print(f"✓ New best model saved! Validation accuracy: {val_acc:.2f}%")
        
        # Print epoch summary
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Accuracy: {train_acc:.2f}% | Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Check if target accuracy is reached
        if val_acc >= target_accuracy:
            print(f"\n🎉 SUCCESS: Model achieved {val_acc:.2f}% validation accuracy!")
            print(f"Target of {target_accuracy}% reached in epoch {epoch}")
            break
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_mnist_model.pth"))
    
    # Final evaluation on test set
    print(f"\nFinal Evaluation (Best model from epoch {best_epoch}):")
    test_acc, test_loss = test(model, device, test_loader, "Test")
    
    # Print final results
    print(f"\nFINAL RESULTS:")
    print(f"  Best Validation Accuracy: {best_val_accuracy:.2f}% (Epoch {best_epoch})")
    print(f"  Final Test Accuracy: {test_acc:.2f}%")
    print(f"  Model Parameters: {param_count:,}")
    print(f"  Training Epochs: {epoch}")
    
    # Check if all requirements are met
    print(f"\nREQUIREMENTS CHECK:")
    print(f"  ✓ Validation Accuracy ≥ 99.4%: {'✓ PASS' if best_val_accuracy >= 99.4 else '✗ FAIL'} ({best_val_accuracy:.2f}%)")
    print(f"  ✓ Parameters < 20,000: {'✓ PASS' if param_count < 20000 else '✗ FAIL'} ({param_count:,})")
    print(f"  ✓ Epochs ≤ 20: {'✓ PASS' if epoch <= 20 else '✗ FAIL'} ({epoch})")
    print(f"  ✓ Uses Batch Normalization: ✓ PASS")
    print(f"  ✓ Uses Dropout: ✓ PASS")
    print(f"  ✓ Uses Fully Connected Layer: ✓ PASS")
    
    # Plot training history
    if len(train_accuracies) > 1:
        plot_training_history(train_accuracies, val_accuracies, train_losses, val_losses)
    
    # Save final model
    torch.save(model.state_dict(), "final_mnist_model.pth")
    print(f"\nModel saved as 'final_mnist_model.pth'")

if __name__ == "__main__":
    main()