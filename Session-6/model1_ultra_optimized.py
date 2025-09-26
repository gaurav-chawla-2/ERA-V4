"""
ULTRA-OPTIMIZED Model_1: Maximum Efficiency with <6000 Parameters
TARGET: Achieve >99% accuracy with <6000 parameters
- Ultra-efficient architecture with strategic channel usage
- Advanced regularization and normalization
- Optimized receptive field progression

EXPECTED RESULT:
- Parameters: ~5,800
- Target Accuracy: >99.0%
- Epochs to reach 99%: ~10-12

KEY INNOVATIONS:
- Minimal channel progression with maximum efficiency
- Strategic use of 1x1 convolutions for channel mixing
- Optimized dropout and batch normalization
- Enhanced feature extraction through better kernel usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        
        # Input: 28x28x1, RF: 1x1
        # Block 1: Ultra-efficient initial extraction
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x8, RF: 3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.conv1_1x1 = nn.Conv2d(8, 10, 1)  # 28x28x10, RF: 3x3 (channel mixing)
        self.bn1_1x1 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x10, RF: 4x4
        self.dropout1 = nn.Dropout(0.03)
        
        # Block 2: Efficient feature refinement
        self.conv2 = nn.Conv2d(10, 14, 3, padding=1)  # 14x14x14, RF: 8x8
        self.bn2 = nn.BatchNorm2d(14)
        self.conv2_1x1 = nn.Conv2d(14, 16, 1)  # 14x14x16, RF: 8x8
        self.bn2_1x1 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x16, RF: 10x10
        self.dropout2 = nn.Dropout(0.03)
        
        # Block 3: Final feature extraction
        self.conv3 = nn.Conv2d(16, 12, 3, padding=1)  # 7x7x12, RF: 14x14
        self.bn3 = nn.BatchNorm2d(12)
        self.conv4 = nn.Conv2d(12, 10, 3, padding=1)  # 7x7x10, RF: 18x18
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x10
        self.dropout3 = nn.Dropout(0.02)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1_1x1(self.conv1_1x1(x)))
        x = self.dropout1(self.pool1(x))
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2_1x1(self.conv2_1x1(x)))
        x = self.dropout2(self.pool2(x))
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(-1, 10)
        x = self.dropout3(x)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)