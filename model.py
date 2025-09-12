import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 28x28x1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 28x28x16
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.03)  # Reduced dropout
        
        self.conv2 = nn.Conv2d(16, 24, 3, padding=1)  # 28x28x24
        self.bn2 = nn.BatchNorm2d(24)
        self.dropout2 = nn.Dropout(0.03)  # Reduced dropout
        
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x24
        
        # Residual connection preparation
        self.conv_res = nn.Conv2d(24, 24, 1)  # 14x14x24 (for residual)
        
        self.conv3 = nn.Conv2d(24, 16, 1)  # 14x14x16 (pointwise to reduce channels)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 24, 3, padding=1)  # 14x14x24
        self.bn4 = nn.BatchNorm2d(24)
        self.dropout3 = nn.Dropout(0.03)  # Reduced dropout
        
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x24
        
        self.conv5 = nn.Conv2d(24, 10, 1)  # 7x7x10
        self.gap = nn.AdaptiveAvgPool2d(1) # 1x1x10

    def forward(self, x):
        x = self.dropout1(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(x)
        
        # Save for residual connection
        residual = self.conv_res(x)
        
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.dropout3(self.bn4(F.relu(self.conv4(x))))
        
        # Add residual connection
        x = x + residual
        
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training function
def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
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
        
        pbar.set_description(desc=f'Epoch: {epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:.2f}')
    return 100*correct/processed

# Testing function
def test(model, device, test_loader):
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
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0*correct/len(test_loader.dataset):.2f}%)\n')
    return 100.0*correct/len(test_loader.dataset)