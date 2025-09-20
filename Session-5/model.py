import torch  # Import PyTorch library for deep learning
import torch.nn as nn  # Import neural network modules from PyTorch
import torch.nn.functional as F  # Import functional interface for operations like activation functions
import torch.optim as optim  # Import optimization algorithms
from torchvision import datasets, transforms  # Import datasets and transformations for computer vision
from tqdm import tqdm  # Import progress bar utility

# Define the optimized model architecture with a class that inherits from nn.Module
class OptimizedMNISTNet(nn.Module):
    def __init__(self):
        super(OptimizedMNISTNet, self).__init__()  # Initialize the parent class
        
        # Block 1: Initial feature extraction (28x28x1 -> 14x14x16)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # First conv layer: 1->8 channels, 3x3 kernel with padding
        self.bn1 = nn.BatchNorm2d(8)  # Batch normalization for conv1 output
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # Second conv layer: 8->16 channels, 3x3 kernel
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization for conv2 output
        self.pool1 = nn.MaxPool2d(2, 2)  # Max pooling to reduce spatial dimensions (28x28 -> 14x14)
        self.dropout1 = nn.Dropout(0.1)  # Dropout for regularization
        
        # Block 2: Feature refinement (14x14x16 -> 7x7x32)
        self.conv3 = nn.Conv2d(16, 16, 1)  # 1x1 conv for channel mixing and parameter efficiency
        self.bn3 = nn.BatchNorm2d(16)  # Batch normalization for conv3 output
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)  # Fourth conv layer: 16->32 channels, 3x3 kernel
        self.bn4 = nn.BatchNorm2d(32)  # Batch normalization for conv4 output
        self.pool2 = nn.MaxPool2d(2, 2)  # Second max pooling (14x14 -> 7x7)
        self.dropout2 = nn.Dropout(0.1)  # Dropout for regularization
        
        # Block 3: Final feature extraction (7x7x32 -> 7x7x16)
        self.conv5 = nn.Conv2d(32, 16, 1)  # 1x1 conv to reduce channels for efficiency
        self.bn5 = nn.BatchNorm2d(16)  # Batch normalization for conv5 output
        
        # Global Average Pooling and Classification
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling to reduce spatial dimensions to 1x1
        self.fc = nn.Linear(16, 10)  # Fully connected layer for final classification (16 -> 10 classes)
        self.dropout3 = nn.Dropout(0.1)  # Final dropout before classification

    def forward(self, x):
        # Block 1: Initial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 -> BatchNorm -> ReLU
        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 -> BatchNorm -> ReLU
        x = self.dropout1(self.pool1(x))  # MaxPool -> Dropout
        
        # Block 2: Feature refinement
        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 -> BatchNorm -> ReLU
        x = F.relu(self.bn4(self.conv4(x)))  # Conv4 -> BatchNorm -> ReLU
        x = self.dropout2(self.pool2(x))  # MaxPool -> Dropout
        
        # Block 3: Final feature extraction
        x = F.relu(self.bn5(self.conv5(x)))  # Conv5 -> BatchNorm -> ReLU
        
        # Global Average Pooling and Classification
        x = self.gap(x)  # Apply global average pooling
        x = x.view(-1, 16)  # Flatten to [batch_size, 16]
        x = self.dropout3(x)  # Apply dropout before final classification
        x = self.fc(x)  # Final fully connected layer
        return F.log_softmax(x, dim=1)  # Apply log softmax for numerical stability

# Function to count trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # Sum all parameters that require gradients

# Training function for one epoch with enhanced logging
def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')  # Create progress bar for training batches
    correct = 0  # Counter for correct predictions
    processed = 0  # Counter for processed samples
    total_loss = 0  # Track total loss for averaging
    
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
        total_loss += loss.item()  # Accumulate loss
        
        # Update progress bar description with current metrics
        pbar.set_description(f'Epoch {epoch}: Loss={loss.item():.4f} Acc={100*correct/processed:.2f}%')
    
    avg_loss = total_loss / len(train_loader)  # Calculate average loss
    accuracy = 100.0 * correct / processed  # Calculate accuracy percentage
    return accuracy, avg_loss  # Return both accuracy and loss

# Testing function to evaluate model performance with detailed metrics
def test(model, device, test_loader, dataset_name="Test"):
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
    accuracy = 100.0 * correct / len(test_loader.dataset)  # Calculate accuracy percentage
    
    # Print test results with enhanced formatting
    print(f'\n{dataset_name} Results:')
    print(f'  Average loss: {test_loss:.4f}')
    print(f'  Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print('-' * 50)
    
    return accuracy, test_loss  # Return both accuracy and loss

# Function to set reproducible random seeds for consistent results
def set_seed(seed=42):
    torch.manual_seed(seed)  # Set PyTorch random seed
    torch.cuda.manual_seed(seed)  # Set CUDA random seed
    torch.cuda.manual_seed_all(seed)  # Set all CUDA devices random seed
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmark for reproducibility


class ImprovedMNISTNet(nn.Module):
    def __init__(self):
        super(ImprovedMNISTNet, self).__init__()
        
        # Block 1: Initial feature extraction (28x28x1 -> 14x14x24)
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)  # Increased channels: 1->12
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)  # Increased channels: 12->24
        self.bn2 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.05)  # Reduced dropout for better learning
        
        # Block 2: Feature refinement (14x14x24 -> 7x7x48)
        self.conv3 = nn.Conv2d(24, 24, 1)  # Channel mixing
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 48, 3, padding=1)  # Increased channels: 24->48
        self.bn4 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.05)
        
        # Block 3: Deep feature extraction (7x7x48 -> 7x7x32)
        self.conv5 = nn.Conv2d(48, 32, 1)  # Less aggressive reduction
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)  # Additional conv layer
        self.bn6 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.05)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)  # Increased input features: 32->10
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        
        # Classification
        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Add this new class after ImprovedMNISTNet
class BalancedMNISTNet(nn.Module):
    def __init__(self):
        super(BalancedMNISTNet, self).__init__()
        
        # Block 1: Initial feature extraction (28x28x1 -> 14x14x20)
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # 1->10 channels
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)  # 10->20 channels
        self.bn2 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.05)  # Reduced dropout
        
        # Block 2: Feature refinement (14x14x20 -> 7x7x32)
        self.conv3 = nn.Conv2d(20, 20, 1)  # Channel mixing
        self.bn3 = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 32, 3, padding=1)  # 20->32 channels
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.05)
        
        # Block 3: Deep feature extraction (7x7x32 -> 7x7x24)
        self.conv5 = nn.Conv2d(32, 24, 1)  # Reduce to 24 channels
        self.bn5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 24, 3, padding=1)  # Keep at 24 channels
        self.bn6 = nn.BatchNorm2d(24)
        self.dropout3 = nn.Dropout(0.05)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(24, 10)  # 24->10 classes
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        
        # Classification
        x = self.gap(x)
        x = x.view(-1, 24)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    

class UltimateMNISTNet(nn.Module):
    def __init__(self):
        super(UltimateMNISTNet, self).__init__()
        
        # Block 1: Initial feature extraction (28x28x1 -> 14x14x24)
        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)  # 1->12 channels (was 10)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)  # 12->24 channels (was 20)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.02)  # Was 0.03
        self.dropout2 = nn.Dropout(0.02)  # Was 0.03
        self.dropout3 = nn.Dropout(0.02)  # Was 0.03
        
        # Block 2: Feature refinement (14x14x24 -> 7x7x36)
        self.conv3 = nn.Conv2d(24, 24, 1)  # Channel mixing
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 36, 3, padding=1)  # 24->36 channels (was 32)
        self.bn4 = nn.BatchNorm2d(36)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.03)
        
        # Block 3: Deep feature extraction (7x7x36 -> 7x7x28)
        self.conv5 = nn.Conv2d(36, 28, 1)  # Reduce to 28 channels (was 24)
        self.bn5 = nn.BatchNorm2d(28)
        self.conv6 = nn.Conv2d(28, 28, 3, padding=1)  # Keep at 28 channels
        self.bn6 = nn.BatchNorm2d(28)
        self.dropout3 = nn.Dropout(0.03)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(28, 10)  # 28->10 classes
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        
        # Classification
        x = self.gap(x)
        x = x.view(-1, 28)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class PerfectMNISTNet(nn.Module):
    """
    Perfect MNIST Network designed to achieve 99.4%+ validation accuracy
    with <20k parameters in ≤20 epochs.
    
    Key optimizations:
    - Optimized channel progression: 1->14->28->32->24
    - Strategic dropout placement: 0.01 (minimal)
    - Efficient 1x1 convolutions for channel mixing
    - Global Average Pooling for parameter efficiency
    - Residual-like connections via careful channel sizing
    """
    def __init__(self):
        super(PerfectMNISTNet, self).__init__()
        
        # Block 1: Initial feature extraction (28x28x1 -> 14x14x28)
        self.conv1 = nn.Conv2d(1, 14, 3, padding=1)  # 1->14 channels
        self.bn1 = nn.BatchNorm2d(14)
        self.conv2 = nn.Conv2d(14, 28, 3, padding=1)  # 14->28 channels
        self.bn2 = nn.BatchNorm2d(28)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.01)  # Minimal dropout
        
        # Block 2: Feature refinement (14x14x28 -> 7x7x32)
        self.conv3 = nn.Conv2d(28, 28, 1)  # Channel mixing (1x1 conv)
        self.bn3 = nn.BatchNorm2d(28)
        self.conv4 = nn.Conv2d(28, 32, 3, padding=1)  # 28->32 channels
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.01)  # Minimal dropout
        
        # Block 3: Deep feature extraction (7x7x32 -> 7x7x24)
        self.conv5 = nn.Conv2d(32, 24, 1)  # Reduce to 24 channels (1x1 conv)
        self.bn5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 24, 3, padding=1)  # Keep at 24 channels
        self.bn6 = nn.BatchNorm2d(24)
        self.dropout3 = nn.Dropout(0.01)  # Minimal dropout
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(24, 10)  # 24->10 classes
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(x)
        
        # Classification
        x = self.gap(x)
        x = x.view(-1, 24)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class FinalMNISTNet(nn.Module):
    """
    Final MNIST Network - The ultimate architecture to achieve 99.4%+ validation accuracy.
    
    Key breakthrough innovations:
    - Depthwise separable convolutions for efficiency
    - Squeeze-and-excitation attention mechanism
    - Residual connections for better gradient flow
    - Optimized channel progression: 1->16->32->48->32
    - Strategic batch normalization placement
    - Minimal but effective dropout (0.005)
    """
    def __init__(self):
        super(FinalMNISTNet, self).__init__()
        
        # Block 1: Initial feature extraction with residual (28x28x1 -> 14x14x32)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 1->16 channels
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 16->32 channels
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: Depthwise separable convolution (14x14x32 -> 7x7x48)
        self.depthwise1 = nn.Conv2d(32, 32, 3, padding=1, groups=32)  # Depthwise
        self.pointwise1 = nn.Conv2d(32, 48, 1)  # Pointwise
        self.bn3 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: Squeeze-and-Excitation + Feature refinement (7x7x48 -> 7x7x32)
        self.se_reduce = nn.Conv2d(48, 12, 1)  # SE reduction
        self.se_expand = nn.Conv2d(12, 48, 1)  # SE expansion
        self.conv3 = nn.Conv2d(48, 32, 1)  # Channel reduction
        self.bn4 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)  # Feature refinement
        self.bn5 = nn.BatchNorm2d(32)
        
        # Minimal dropout for regularization
        self.dropout = nn.Dropout(0.005)  # Very minimal dropout
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
        
    def forward(self, x):
        # Block 1: Initial feature extraction
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = self.pool1(x1)
        
        # Block 2: Depthwise separable convolution
        x2 = F.relu(self.bn3(self.pointwise1(self.depthwise1(x1))))
        x2 = self.pool2(x2)
        
        # Block 3: Squeeze-and-Excitation attention
        # SE mechanism
        se = F.adaptive_avg_pool2d(x2, 1)
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        x2 = x2 * se  # Apply attention
        
        # Feature refinement with residual connection
        x3 = F.relu(self.bn4(self.conv3(x2)))
        x3 = F.relu(self.bn5(self.conv4(x3)))
        
        # Apply minimal dropout
        x3 = self.dropout(x3)
        
        # Classification
        x3 = self.gap(x3)
        x3 = x3.view(-1, 32)
        x3 = self.fc(x3)
        return F.log_softmax(x3, dim=1)

class CompactChampionMNISTNet(nn.Module):
    """
    Ultra-compact champion model with breakthrough innovations:
    - Lightweight dual-path architecture
    - Efficient attention mechanisms
    - Optimized channel progression
    - Strategic residual connections
    Target: 99.4%+ accuracy with <20k parameters
    """
    
    def __init__(self):
        super(CompactChampionMNISTNet, self).__init__()
        
        # Block 1: Compact dual-path feature extraction (1 -> 16 channels)
        # Path A: Standard convolutions
        self.conv1a = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(8)
        
        # Path B: Dilated convolutions for multi-scale features
        self.conv1b = nn.Conv2d(1, 8, 3, padding=2, dilation=2)
        self.bn1b = nn.BatchNorm2d(8)
        
        # Fusion layer
        self.fusion1 = nn.Conv2d(16, 16, 1)  # 1x1 conv for fusion
        self.bn_fusion1 = nn.BatchNorm2d(16)
        
        # Block 2: Feature enhancement (16 -> 24 channels)
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        
        # Lightweight Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(24, 1, 1),  # Channel reduction
            nn.Sigmoid()
        )
        
        # Block 3: Feature refinement (24 -> 20 channels)
        self.conv3 = nn.Conv2d(24, 20, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        
        # Lightweight Channel Attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(20, 5, 1),  # Squeeze to 5 channels
            nn.ReLU(),
            nn.Conv2d(5, 20, 1),  # Excite back to 20
            nn.Sigmoid()
        )
        
        # Residual connection adapter
        self.residual_adapter = nn.Conv2d(16, 20, 1)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(20, 10)
        
        # Ultra-minimal dropout
        self.dropout = nn.Dropout(0.001)
        
    def forward(self, x):
        # Block 1: Dual-path feature extraction
        # Path A: Standard convolutions
        x1a = F.relu(self.bn1a(self.conv1a(x)))
        
        # Path B: Dilated convolutions
        x1b = F.relu(self.bn1b(self.conv1b(x)))
        
        # Fusion
        x1_fused = torch.cat([x1a, x1b], dim=1)
        x1 = F.relu(self.bn_fusion1(self.fusion1(x1_fused)))
        x1 = F.max_pool2d(x1, 2)  # 28x28 -> 14x14
        
        # Block 2: Feature enhancement with spatial attention
        x2 = F.relu(self.bn2(self.conv2(x1)))
        
        # Apply spatial attention
        spatial_att = self.spatial_attention(x2)
        x2 = x2 * spatial_att
        x2 = F.max_pool2d(x2, 2)  # 14x14 -> 7x7
        
        # Block 3: Feature refinement with channel attention and residual
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        # Apply channel attention
        channel_att = self.channel_attention(x3)
        x3 = x3 * channel_att
        
        # Residual connection (adapt x1 to match x3 dimensions)
        x1_adapted = self.residual_adapter(F.adaptive_avg_pool2d(x1, 7))
        x3 = x3 + x1_adapted
        
        # Apply minimal dropout
        x3 = self.dropout(x3)
        
        # Classification
        x3 = self.gap(x3)
        x3 = x3.view(-1, 20)
        x3 = self.fc(x3)
        return F.log_softmax(x3, dim=1)

class OptimalChampionMNISTNet(nn.Module):
    """
    Optimal champion model that maximizes the 20k parameter budget:
    - Enhanced dual-path architecture
    - Stronger attention mechanisms
    - Deeper feature processing
    - Strategic residual connections
    Target: 99.4%+ accuracy with ~18k-19k parameters
    """
    
    def __init__(self):
        super(OptimalChampionMNISTNet, self).__init__()
        
        # Block 1: Enhanced dual-path feature extraction (1 -> 24 channels)
        # Path A: Standard convolutions
        self.conv1a = nn.Conv2d(1, 12, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(12)
        
        # Path B: Dilated convolutions for multi-scale features
        self.conv1b = nn.Conv2d(1, 12, 3, padding=2, dilation=2)
        self.bn1b = nn.BatchNorm2d(12)
        
        # Fusion layer
        self.fusion1 = nn.Conv2d(24, 24, 1)
        self.bn_fusion1 = nn.BatchNorm2d(24)
        
        # Block 2: Feature enhancement (24 -> 32 channels)
        self.conv2a = nn.Conv2d(24, 32, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        
        # Additional depth for better feature learning
        self.conv2b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(32)
        
        # Enhanced Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(32, 8, 1),  # Intermediate channels for better attention
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # Block 3: Feature refinement (32 -> 28 channels)
        self.conv3 = nn.Conv2d(32, 28, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(28)
        
        # Enhanced Channel Attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(28, 8, 1),  # Squeeze to 8 channels
            nn.ReLU(),
            nn.Conv2d(8, 28, 1),  # Excite back to 28
            nn.Sigmoid()
        )
        
        # Residual connection adapters
        self.residual_adapter1 = nn.Conv2d(24, 28, 1)  # Block1 -> Block3
        self.residual_adapter2 = nn.Conv2d(32, 28, 1)  # Block2 -> Block3
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(28, 10)
        
        # Strategic dropout
        self.dropout1 = nn.Dropout(0.01)  # After block 2
        self.dropout2 = nn.Dropout(0.005)  # Before classification
        
    def forward(self, x):
        # Block 1: Enhanced dual-path feature extraction
        # Path A: Standard convolutions
        x1a = F.relu(self.bn1a(self.conv1a(x)))
        
        # Path B: Dilated convolutions
        x1b = F.relu(self.bn1b(self.conv1b(x)))
        
        # Fusion
        x1_fused = torch.cat([x1a, x1b], dim=1)
        x1 = F.relu(self.bn_fusion1(self.fusion1(x1_fused)))
        x1 = F.max_pool2d(x1, 2)  # 28x28 -> 14x14
        
        # Block 2: Enhanced feature processing with spatial attention
        x2 = F.relu(self.bn2a(self.conv2a(x1)))
        x2 = F.relu(self.bn2b(self.conv2b(x2)))
        
        # Apply enhanced spatial attention
        spatial_att = self.spatial_attention(x2)
        x2 = x2 * spatial_att
        x2 = self.dropout1(x2)  # Strategic dropout
        x2 = F.max_pool2d(x2, 2)  # 14x14 -> 7x7
        
        # Block 3: Feature refinement with channel attention and multi-residual
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        # Apply enhanced channel attention
        channel_att = self.channel_attention(x3)
        x3 = x3 * channel_att
        
        # Multi-scale residual connections
        x1_adapted = self.residual_adapter1(F.adaptive_avg_pool2d(x1, 7))
        x2_adapted = self.residual_adapter2(F.adaptive_avg_pool2d(x2, 7))
        x3 = x3 + 0.3 * x1_adapted + 0.7 * x2_adapted  # Weighted residuals
        
        # Apply final dropout
        x3 = self.dropout2(x3)
        
        # Classification
        x3 = self.gap(x3)
        x3 = x3.view(-1, 28)
        x3 = self.fc(x3)
        return F.log_softmax(x3, dim=1)

class PrecisionChampionMNISTNet(nn.Module):
    """
    Precision-engineered champion model for exactly 99.4%+ accuracy:
    - Carefully balanced dual-path architecture
    - Efficient attention mechanisms
    - Strategic parameter allocation
    - Optimized for ~18k-19k parameters
    """
    
    def __init__(self):
        super(PrecisionChampionMNISTNet, self).__init__()
        
        # Block 1: Balanced dual-path feature extraction (1 -> 20 channels)
        # Path A: Standard convolutions
        self.conv1a = nn.Conv2d(1, 10, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(10)
        
        # Path B: Dilated convolutions for multi-scale features
        self.conv1b = nn.Conv2d(1, 10, 3, padding=2, dilation=2)
        self.bn1b = nn.BatchNorm2d(10)
        
        # Fusion layer
        self.fusion1 = nn.Conv2d(20, 20, 1)
        self.bn_fusion1 = nn.BatchNorm2d(20)
        
        # Block 2: Feature enhancement (20 -> 28 channels)
        self.conv2 = nn.Conv2d(20, 28, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(28)
        
        # Efficient Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(28, 4, 1),  # Efficient channel reduction
            nn.ReLU(),
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid()
        )
        
        # Block 3: Feature refinement (28 -> 24 channels)
        self.conv3 = nn.Conv2d(28, 24, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        
        # Efficient Channel Attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(24, 6, 1),  # Squeeze to 6 channels
            nn.ReLU(),
            nn.Conv2d(6, 24, 1),  # Excite back to 24
            nn.Sigmoid()
        )
        
        # Residual connection adapter
        self.residual_adapter = nn.Conv2d(20, 24, 1)
        
        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(24, 10)
        
        # Strategic dropout
        self.dropout1 = nn.Dropout(0.008)  # After spatial attention
        self.dropout2 = nn.Dropout(0.003)  # Before classification
        
    def forward(self, x):
        # Block 1: Balanced dual-path feature extraction
        # Path A: Standard convolutions
        x1a = F.relu(self.bn1a(self.conv1a(x)))
        
        # Path B: Dilated convolutions
        x1b = F.relu(self.bn1b(self.conv1b(x)))
        
        # Fusion
        x1_fused = torch.cat([x1a, x1b], dim=1)
        x1 = F.relu(self.bn_fusion1(self.fusion1(x1_fused)))
        x1 = F.max_pool2d(x1, 2)  # 28x28 -> 14x14
        
        # Block 2: Feature enhancement with spatial attention
        x2 = F.relu(self.bn2(self.conv2(x1)))
        
        # Apply efficient spatial attention
        spatial_att = self.spatial_attention(x2)
        x2 = x2 * spatial_att
        x2 = self.dropout1(x2)  # Strategic dropout
        x2 = F.max_pool2d(x2, 2)  # 14x14 -> 7x7
        
        # Block 3: Feature refinement with channel attention and residual
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        # Apply efficient channel attention
        channel_att = self.channel_attention(x3)
        x3 = x3 * channel_att
        
        # Residual connection
        x1_adapted = self.residual_adapter(F.adaptive_avg_pool2d(x1, 7))
        x3 = x3 + 0.5 * x1_adapted  # Balanced residual
        
        # Apply final dropout
        x3 = self.dropout2(x3)
        
        # Classification
        x3 = self.gap(x3)
        x3 = x3.view(-1, 24)
        x3 = self.fc(x3)
        return F.log_softmax(x3, dim=1)