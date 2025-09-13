import torch  # Import PyTorch library
import torch.optim as optim  # Import optimization algorithms
from torchvision import datasets, transforms  # Import datasets and transformations
from model import Net, count_parameters, train, test  # Import model and utility functions

def main():
    # Check if CUDA is available and set the device accordingly
    use_cuda = torch.cuda.is_available()  # Check if GPU is available
    device = torch.device("cuda" if use_cuda else "cpu")  # Set device to GPU if available, otherwise CPU
    
    # Define training hyperparameters
    batch_size = 32  # Number of samples per batch (smaller for better generalization)
    epochs = 1  # Number of complete passes through the dataset
    lr = 0.1  # Learning rate for optimizer (higher with OneCycleLR)
    momentum = 0.9  # Momentum factor for SGD optimizer
    
    # Define data transformations with augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-15.0, 15.0), fill=(0,)),  # Random rotation within ±15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random translation and scaling
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    
    # Define data transformations for testing (no augmentation)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)  # Load MNIST training data
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transforms)  # Load MNIST test data
    
    # Set dataloader parameters based on device
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {'batch_size': batch_size, 'shuffle': True}
    
    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)  # Create training data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)  # Create test data loader
    
    # Initialize the model and move it to the device
    model = Net().to(device)  # Create model instance and move to GPU/CPU
    
    # Print model summary and parameter count
    print(model)  # Print model architecture
    param_count = count_parameters(model)  # Count trainable parameters
    print(f"Total trainable parameters: {param_count}")  # Print parameter count
    
    # Check if the model meets the parameter constraint
    if param_count > 25000:
        print(f"ERROR: Model has {param_count} parameters, which exceeds the limit of 25,000")  # Check parameter limit
        return  # Exit if limit exceeded
    
    # Initialize optimizer with weight decay for regularization
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=2e-4)  # SGD optimizer with weight decay
    
    # Use OneCycleLR scheduler for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),  # Configure OneCycleLR
        pct_start=0.2, anneal_strategy='cos'  # 20% warmup, cosine annealing
    )
    
    # Train and test for specified epochs
    for epoch in range(1, epochs + 1):  # Loop through epochs
        train_acc = train(model, device, train_loader, optimizer, epoch, scheduler)  # Train one epoch
        test_acc = test(model, device, test_loader)  # Evaluate on test set
        
        # Print epoch results
        print(f"Epoch {epoch}: Train Accuracy = {train_acc:.2f}%, Test Accuracy = {test_acc:.2f}%")  # Print accuracies
        
        # Check if the model meets the accuracy requirement
        if test_acc >= 95.0:
            print(f"SUCCESS: Model achieved {test_acc:.2f}% test accuracy, which meets the requirement of 95%")  # Success message
        else:
            print(f"Model achieved {test_acc:.2f}% test accuracy, which does not meet the requirement of 95%")  # Failure message
    
    # Save the trained model
    torch.save(model.state_dict(), "mnist_model.pth")  # Save model weights
    print("Model saved as mnist_model.pth")  # Confirmation message

if __name__ == "__main__":
    main()  # Execute main function when script is run directly