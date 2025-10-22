"""
ResNet model implementation for CIFAR-100
Clean, modular implementation following best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type, Union
import logging


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, dropout: float = 0.0):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout2(F.relu(out))
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for deeper ResNets"""
    expansion = 4
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, dropout: float = 0.0):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, 
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.dropout3 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        out = self.dropout2(F.relu(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.dropout3(F.relu(out))
        return out


class ResNet(nn.Module):
    """ResNet architecture for CIFAR-100"""
    
    def __init__(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]], 
        num_blocks: List[int], 
        num_classes: int = 100,
        dropout: float = 0.0
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created ResNet with {self.count_parameters():,} parameters")
    
    def _make_layer(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]], 
        planes: int, 
        num_blocks: int, 
        stride: int
    ) -> nn.Sequential:
        """Create a residual layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Model factory functions
def ResNet18(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-18 for CIFAR-100"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dropout)


def ResNet34(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-34 for CIFAR-100"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, dropout)


def ResNet50(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-50 for CIFAR-100"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, dropout)


def ResNet101(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-101 for CIFAR-100"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, dropout)


def ResNet152(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-152 for CIFAR-100"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, dropout)


# CIFAR-specific ResNets
def ResNet20(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-20 for CIFAR-100"""
    return ResNet(BasicBlock, [3, 3, 3], num_classes, dropout)


def ResNet32(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-32 for CIFAR-100"""
    return ResNet(BasicBlock, [5, 5, 5], num_classes, dropout)


def ResNet44(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-44 for CIFAR-100"""
    return ResNet(BasicBlock, [7, 7, 7], num_classes, dropout)


def ResNet56(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-56 for CIFAR-100"""
    return ResNet(BasicBlock, [9, 9, 9], num_classes, dropout)


def ResNet110(num_classes: int = 100, dropout: float = 0.0) -> ResNet:
    """ResNet-110 for CIFAR-100"""
    return ResNet(BasicBlock, [18, 18, 18], num_classes, dropout)


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model creation and forward pass"""
    model = ResNet56()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    test_model()