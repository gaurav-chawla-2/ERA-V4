import torch  # Import PyTorch library for deep learning
import torch.nn as nn  # Import neural network modules from PyTorch
import torch.nn.functional as F  # Import functional interface for operations like activation functions
import torch.optim as optim  # Import optimization algorithms
from torchvision import datasets, transforms  # Import datasets and transformations for computer vision
from tqdm import tqdm  # Import progress bar utility

# Define the model architecture with a class that inherits from nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # Initialize the parent class
        # Input: 28x28x1 (MNIST images are 28x28 with 1 channel)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # First conv layer: 1 input channel, 16 output channels, 3x3 kernel with padding
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization to stabilize training for conv1 output
        self.dropout1 = nn.Dropout(0.03)  # Light dropout to prevent overfitting
        
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)  # Second conv layer: 16 input channels, 24 output channels, 3x3 kernel with padding
        self.bn2 = nn.BatchNorm2d(24)  # Batch normalization for conv2 output
        self.dropout2 = nn.Dropout(0.03)  # Light dropout after conv2
        
        self.pool1 = nn.MaxPool2d(2, 2)  # Max pooling to reduce spatial dimensions by half (28x28 -> 14x14)
        
        # Prepare for residual connection with 1x1 convolution to maintain channel dimensions
        self.conv_res = nn.Conv2d(24, 24, 1)  # 1x1 convolution for residual path (preserves dimensions)
        
        self.conv3 = nn.Conv2d(24, 16, 1)  # 1x1 convolution to reduce channels (dimensionality reduction)
        self.bn3 = nn.BatchNorm2d(16)  # Batch normalization for conv3 output
        
        self.conv4 = nn.Conv2d(16, 24, 3, padding=1)  # Fourth conv layer: 16 input channels, 24 output channels, 3x3 kernel with padding
        self.bn4 = nn.BatchNorm2d(24)  # Batch normalization for conv4 output
        self.dropout3 = nn.Dropout(0.03)  # Light dropout after conv4
        
        self.pool2 = nn.MaxPool2d(2, 2)  # Second max pooling to further reduce dimensions (14x14 -> 7x7)
        
        self.conv5 = nn.Conv2d(24, 10, 1)  # Final 1x1 convolution to get 10 output channels (one per digit class)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to reduce spatial dimensions to 1x1

    def forward(self, x):
        # Forward pass through the network
        x = self.dropout1(self.bn1(F.relu(self.conv1(x))))  # Conv1 -> ReLU -> BatchNorm -> Dropout
        x = self.dropout2(self.bn2(F.relu(self.conv2(x))))  # Conv2 -> ReLU -> BatchNorm -> Dropout
        x = self.pool1(x)  # Apply first max pooling
        
        # Save the output for residual connection
        residual = self.conv_res(x)  # Process the residual path with 1x1 convolution
        
        x = self.bn3(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> BatchNorm
        x = self.dropout3(self.bn4(F.relu(self.conv4(x))))  # Conv4 -> ReLU -> BatchNorm -> Dropout
        
        # Add residual connection to improve gradient flow
        x = x + residual  # Add the residual connection to the main path
        
        x = self.pool2(x)  # Apply second max pooling
        
        x = F.relu(self.conv5(x))  # Final convolution with ReLU activation
        x = self.gap(x)  # Apply global average pooling
        x = x.view(-1, 10)  # Reshape to [batch_size, 10] for classification
        return F.log_softmax(x, dim=1)  # Apply log softmax for numerical stability in loss calculation

# Function to count trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # Sum all parameters that require gradients

# Training function for one epoch
def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    pbar = tqdm(train_loader)  # Create progress bar for training batches
    correct = 0  # Counter for correct predictions
    processed = 0  # Counter for processed samples
    for batch_idx, (data, target) in enumerate(pbar):  # Iterate through batches
        data, target = data.to(device), target.to(device)  # Move data to the specified device (CPU/GPU)
        optimizer.zero_grad()  # Reset gradients to zero before backpropagation
        output = model(data)  # Forward pass: get model predictions
        loss = F.nll_loss(output, target)  # Calculate negative log likelihood loss
        loss.backward()  # Backward pass: calculate gradients
        optimizer.step()  # Update model parameters using optimizer
        if scheduler is not None:
            scheduler.step()  # Update learning rate if scheduler is provided
        
        pred = output.argmax(dim=1, keepdim=True)  # Get predicted class (highest probability)
        correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
        processed += len(data)  # Update processed samples count
        
        # Update progress bar description with current metrics
        pbar.set_description(desc=f'Epoch: {epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:.2f}')
    return 100*correct/processed  # Return training accuracy percentage

# Testing function to evaluate model performance
def test(model, device, test_loader):
    model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)
    test_loss = 0  # Initialize test loss
    correct = 0  # Initialize correct predictions counter
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for data, target in test_loader:  # Iterate through test batches
            data, target = data.to(device), target.to(device)  # Move data to device
            output = model(data)  # Forward pass
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
    
    test_loss /= len(test_loader.dataset)  # Calculate average loss
    # Print test results
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0*correct/len(test_loader.dataset):.2f}%)\n')
    return 100.0*correct/len(test_loader.dataset)