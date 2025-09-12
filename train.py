import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net, count_parameters, train, test

def main():
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Training hyperparameters
    batch_size = 32  # Smaller batch size for better generalization
    epochs = 1
    lr = 0.1  # Higher learning rate with OneCycleLR
    momentum = 0.9
    
    # Data transformations with more augmentation
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-15.0, 15.0), fill=(0,)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('../data', train=False, transform=test_transforms)
    
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {'batch_size': batch_size, 'shuffle': True}
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    
    # Initialize the model
    model = Net().to(device)
    
    # Print model summary and parameter count
    print(model)
    param_count = count_parameters(model)
    print(f"Total trainable parameters: {param_count}")
    
    # Check if the model meets the parameter constraint
    if param_count > 25000:
        print(f"ERROR: Model has {param_count} parameters, which exceeds the limit of 25,000")
        return
    
    # Initialize optimizer with weight decay for regularization
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=2e-4)
    
    # Use OneCycleLR with higher max_lr
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy='cos'
    )
    
    # Train and test for specified epochs
    for epoch in range(1, epochs + 1):
        train_acc = train(model, device, train_loader, optimizer, epoch, scheduler)
        test_acc = test(model, device, test_loader)
        
        print(f"Epoch {epoch}: Train Accuracy = {train_acc:.2f}%, Test Accuracy = {test_acc:.2f}%")
        
        # Check if the model meets the accuracy requirement
        if test_acc >= 95.0:
            print(f"SUCCESS: Model achieved {test_acc:.2f}% test accuracy, which meets the requirement of 95%")
        else:
            print(f"Model achieved {test_acc:.2f}% test accuracy, which does not meet the requirement of 95%")
    
    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved as mnist_model.pth")

if __name__ == "__main__":
    main()