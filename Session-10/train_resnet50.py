"""
Single-file ImageNet-mini ResNet50 Training Solution
===================================================

A clean, comprehensive implementation for training ResNet50 on ImageNet-mini dataset
with automatic learning rate optimization using ATOM optimizer.

Features:
- Clear ResNet50 implementation with visible layer operations
- ATOM optimizer for automatic learning rate finding
- Comprehensive training progress tracking and visualization
- Early stopping and progress feedback
- Modular design for easy parameter modification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
from PIL import Image
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS - Modify these for different experiments
# ============================================================================

# Dataset Configuration
DATASET_PATH = "/opt/dlami/nvme/imagenet"  # Path to ImageNet dataset (change to tiny-imagenet-200 for Tiny-ImageNet)
NUM_CLASSES = 1000  # ImageNet has 1000 classes (change to 200 for Tiny-ImageNet)
IMAGE_SIZE = 224    # ImageNet uses 224x224 images (change to 64 for Tiny-ImageNet)
BATCH_SIZE = 96     # Optimized for T4 GPU (16GB VRAM) on g4dn.xlarge
NUM_WORKERS = 4     # Matched to 4 vCPUs on g4dn.xlarge instance

# Training Configuration
MAX_EPOCHS = 100          # Maximum training epochs (increased for full ImageNet)
EARLY_STOP_PATIENCE = 15  # Early stopping patience (increased for full ImageNet)
MIN_EPOCHS_FOR_FEEDBACK = 10  # Minimum epochs before early feedback
TARGET_ACCURACY = 75.0   # Target accuracy percentage (realistic for full ImageNet)
VALIDATION_SPLIT = 0.2    # Validation split ratio

# Model Configuration
DROPOUT_RATE = 0.2        # Increased for full ImageNet to prevent overfitting
LABEL_SMOOTHING = 0.1     # Label smoothing for better generalization

# Optimizer Configuration
OPTIMIZER_TYPE = 'SGD'    # SGD often works better for full ImageNet
INITIAL_LR = 0.1          # Standard learning rate for ImageNet training
MOMENTUM = 0.9            # Momentum for SGD optimizer
WEIGHT_DECAY = 1e-3       # Standard weight decay for ImageNet

# Logging and Visualization Configuration
SAVE_DIR = "./results"    # Directory to save results
LOG_INTERVAL = 10         # Log every N batches
PLOT_STYLE = 'seaborn-v0_8'  # Matplotlib style

# Mixed Precision Training Configuration (optimized for T4 GPU)
USE_MIXED_PRECISION = True        # Enable automatic mixed precision for faster training

# Checkpoint Configuration
CHECKPOINT_DIR = "./checkpoints"  # Directory to save checkpoints
SAVE_CHECKPOINT_EVERY = 5         # Save checkpoint every N epochs
RESUME_FROM_CHECKPOINT = None     # Path to checkpoint to resume from (None to start fresh)

# ============================================================================
# CUSTOM DATASET CLASSES
# ============================================================================

class TinyImageNetValidationDataset(Dataset):
    """Custom dataset for Tiny-ImageNet validation set using val_annotations.txt"""
    
    def __init__(self, val_dir: str, annotations_file: str, class_to_idx: Dict[str, int], transform=None):
        """
        Args:
            val_dir: Path to validation images directory
            annotations_file: Path to val_annotations.txt
            class_to_idx: Dictionary mapping class names to indices (from training set)
            transform: Optional transform to be applied on images
        """
        self.val_dir = val_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        
        # Parse annotations file
        self.samples = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_name = parts[1]
                    if class_name in class_to_idx:
                        img_path = os.path.join(val_dir, img_name)
                        if os.path.exists(img_path):
                            self.samples.append((img_path, class_to_idx[class_name]))
        
        print(f"✅ Loaded {len(self.samples)} validation samples with proper class alignment")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# RESNET50 IMPLEMENTATION - Clear layer-by-layer architecture
# ============================================================================

class ResNet50(nn.Module):
    """
    ResNet50 implementation with clearly visible layer operations.
    Each component is explicitly defined for transparency.
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initial Convolution Block (adapted for 64x64 Tiny-ImageNet)
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)
        # Remove maxpool to preserve spatial dimensions for small images
        # self.initial_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers - Stage 1 (64 channels, 3 blocks)
        self.stage1_block1 = self._make_bottleneck_block(64, 64, 256, stride=1, downsample=True)
        self.stage1_block2 = self._make_bottleneck_block(256, 64, 256, stride=1)
        self.stage1_block3 = self._make_bottleneck_block(256, 64, 256, stride=1)
        
        # ResNet Layers - Stage 2 (128 channels, 4 blocks)
        self.stage2_block1 = self._make_bottleneck_block(256, 128, 512, stride=2, downsample=True)
        self.stage2_block2 = self._make_bottleneck_block(512, 128, 512, stride=1)
        self.stage2_block3 = self._make_bottleneck_block(512, 128, 512, stride=1)
        self.stage2_block4 = self._make_bottleneck_block(512, 128, 512, stride=1)
        
        # ResNet Layers - Stage 3 (256 channels, 6 blocks)
        self.stage3_block1 = self._make_bottleneck_block(512, 256, 1024, stride=2, downsample=True)
        self.stage3_block2 = self._make_bottleneck_block(1024, 256, 1024, stride=1)
        self.stage3_block3 = self._make_bottleneck_block(1024, 256, 1024, stride=1)
        self.stage3_block4 = self._make_bottleneck_block(1024, 256, 1024, stride=1)
        self.stage3_block5 = self._make_bottleneck_block(1024, 256, 1024, stride=1)
        self.stage3_block6 = self._make_bottleneck_block(1024, 256, 1024, stride=1)
        
        # ResNet Layers - Stage 4 (512 channels, 3 blocks)
        self.stage4_block1 = self._make_bottleneck_block(1024, 512, 2048, stride=2, downsample=True)
        self.stage4_block2 = self._make_bottleneck_block(2048, 512, 2048, stride=1)
        self.stage4_block3 = self._make_bottleneck_block(2048, 512, 2048, stride=1)
        
        # Final layers
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.dropout = nn.Dropout(dropout_rate)            # Dropout for regularization
        self.classifier = nn.Linear(2048, num_classes)     # Final classification layer
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_bottleneck_block(self, in_channels: int, mid_channels: int, out_channels: int, 
                              stride: int = 1, downsample: bool = False) -> nn.Module:
        """Create a bottleneck block with 1x1 -> 3x3 -> 1x1 convolutions"""
        
        class BottleneckBlock(nn.Module):
            def __init__(self):
                super().__init__()
                # 1x1 convolution (reduce dimensions)
                self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(mid_channels)
                self.relu1 = nn.ReLU(inplace=True)
                
                # 3x3 convolution (main computation)
                self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                                     stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(mid_channels)
                self.relu2 = nn.ReLU(inplace=True)
                
                # 1x1 convolution (expand dimensions)
                self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(out_channels)
                
                # Shortcut connection (identity or projection)
                if downsample:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                else:
                    self.shortcut = nn.Identity()
                
                self.final_relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                # Main path
                out = self.relu1(self.bn1(self.conv1(x)))
                out = self.relu2(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                
                # Shortcut path
                shortcut = self.shortcut(x)
                
                # Add and apply final ReLU
                out = self.final_relu(out + shortcut)
                return out
        
        return BottleneckBlock()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
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
        """Forward pass through ResNet50"""
        # Initial convolution block (adapted for 64x64 images)
        x = self.initial_conv(x)      # 3 -> 64 channels, same spatial size
        x = self.initial_bn(x)        # Batch normalization
        x = self.initial_relu(x)      # ReLU activation
        # Skip maxpool to preserve spatial dimensions for small images
        
        # Stage 1: 64 -> 256 channels
        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.stage1_block3(x)
        
        # Stage 2: 256 -> 512 channels, /2 spatial
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.stage2_block3(x)
        x = self.stage2_block4(x)
        
        # Stage 3: 512 -> 1024 channels, /2 spatial
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.stage3_block3(x)
        x = self.stage3_block4(x)
        x = self.stage3_block5(x)
        x = self.stage3_block6(x)
        
        # Stage 4: 1024 -> 2048 channels, /2 spatial
        x = self.stage4_block1(x)
        x = self.stage4_block2(x)
        x = self.stage4_block3(x)
        
        # Final layers
        x = self.global_avgpool(x)    # Global average pooling -> (batch, 2048, 1, 1)
        x = torch.flatten(x, 1)       # Flatten -> (batch, 2048)
        x = self.dropout(x)           # Dropout for regularization
        x = self.classifier(x)        # Classification -> (batch, num_classes)
        
        return x

# ============================================================================
# ATOM OPTIMIZER IMPLEMENTATION
# ============================================================================

# ... existing code ...

# ============================================================================
# OPTIMIZER FACTORY FUNCTION
# ============================================================================

def create_optimizer(parameters, optimizer_type: str, lr: float, momentum: float, weight_decay: float):
    """Create optimizer based on type"""
    if optimizer_type == 'SGD':
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation data transforms"""
    
    # Training transforms with light augmentation (optimized for Tiny-ImageNet 64x64)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),   # Further reduced for initial learning
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders() -> Tuple[DataLoader, DataLoader, int]:
    """Create training and validation data loaders and return number of classes"""
    
    train_transform, val_transform = get_data_transforms()
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("Please download the dataset and extract it to the specified path.")
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    # Detect dataset type
    is_tiny_imagenet = 'tiny-imagenet' in DATASET_PATH.lower()
    print(f"🔍 Detected dataset type: {'Tiny-ImageNet' if is_tiny_imagenet else 'Full ImageNet'}")
    
    # Load datasets
    try:
        # Load training dataset
        train_dataset = datasets.ImageFolder(
            root=os.path.join(DATASET_PATH, 'train'),
            transform=train_transform
        )
        
        # Auto-detect number of classes and get class mapping
        num_classes = len(train_dataset.classes)
        class_to_idx = train_dataset.class_to_idx
        
        # Load validation dataset - different handling for Tiny-ImageNet vs full ImageNet
        if is_tiny_imagenet:
            # Tiny-ImageNet specific validation loading
            val_images_dir = os.path.join(DATASET_PATH, 'val', 'images')
            val_annotations_file = os.path.join(DATASET_PATH, 'val', 'val_annotations.txt')
            
            if os.path.exists(val_annotations_file) and os.path.exists(val_images_dir):
                print("🔧 Using val_annotations.txt for Tiny-ImageNet validation")
                val_dataset = TinyImageNetValidationDataset(
                    val_dir=val_images_dir,
                    annotations_file=val_annotations_file,
                    class_to_idx=class_to_idx,
                    transform=val_transform
                )
            else:
                print("⚠️  val_annotations.txt not found, falling back to ImageFolder")
                val_dataset = datasets.ImageFolder(
                    root=os.path.join(DATASET_PATH, 'val'),
                    transform=val_transform
                )
        else:
            # Full ImageNet validation loading
            val_path = os.path.join(DATASET_PATH, 'val')
            if not os.path.exists(val_path):
                # Try alternative validation path names
                alt_paths = ['validation', 'valid', 'test']
                for alt in alt_paths:
                    alt_path = os.path.join(DATASET_PATH, alt)
                    if os.path.exists(alt_path):
                        val_path = alt_path
                        print(f"🔧 Using validation path: {alt}")
                        break
            
            val_dataset = datasets.ImageFolder(
                root=val_path,
                transform=val_transform
            )
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Expected structure for Tiny-ImageNet:")
        print(f"{DATASET_PATH}/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   ├── class2/")
        print("  │   └── ...")
        print("  └── val/")
        print("      ├── images/")
        print("      └── val_annotations.txt")
        print("\nExpected structure for full ImageNet:")
        print(f"{DATASET_PATH}/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   ├── class2/")
        print("  │   └── ...")
        print("  └── val/ (or validation/)")
        print("      ├── class1/")
        print("      ├── class2/")
        print("      └── ...")
        raise
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Number of classes: {num_classes}")
    
    return train_loader, val_loader, num_classes

# ============================================================================
# LEARNING RATE FINDER FOR ATOM OPTIMIZER
# ============================================================================

# LR finder function removed - using stable optimizers instead

# ============================================================================
# TRAINING AND VALIDATION FUNCTIONS
# ============================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, device: torch.device, epoch: int, 
                scaler: GradScaler = None) -> Tuple[float, float, float]:
    """Train for one epoch with mixed precision support"""
    
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    # Initialize scaler if not provided and mixed precision is enabled
    if USE_MIXED_PRECISION and scaler is None:
        scaler = GradScaler()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if USE_MIXED_PRECISION:
            # Mixed precision forward pass
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Calculate Top-1 and Top-5 accuracy
        _, pred_top5 = output.topk(5, 1, True, True)
        pred_top5 = pred_top5.t()
        correct_batch = pred_top5.eq(target.view(1, -1).expand_as(pred_top5))
        
        # Top-1 accuracy
        correct_top1 += correct_batch[:1].reshape(-1).float().sum(0).item()
        
        # Top-5 accuracy
        correct_top5 += correct_batch[:5].reshape(-1).float().sum(0).item()
        
        # Statistics
        running_loss += loss.item()
        total += target.size(0)
        
        # Logging
        if batch_idx % LOG_INTERVAL == 0:
            current_top1 = 100. * correct_top1 / total
            current_top5 = 100. * correct_top5 / total
            precision_str = "FP16" if USE_MIXED_PRECISION else "FP32"
            print(f'   Batch {batch_idx:3d}/{len(train_loader):3d} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Top-1: {current_top1:.2f}% | '
                  f'Top-5: {current_top5:.2f}% | {precision_str}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_top1_acc = 100. * correct_top1 / total
    epoch_top5_acc = 100. * correct_top5 / total
    
    # Log epoch summary
    precision_str = "FP16" if USE_MIXED_PRECISION else "FP32"
    logging.info(f"Train Epoch {epoch}: Loss={epoch_loss:.4f}, Top-1 Acc={epoch_top1_acc:.2f}%, Top-5 Acc={epoch_top5_acc:.2f}% ({precision_str})")
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc
    
    return epoch_loss, epoch_acc

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Tuple[float, float, float]:
    """Validate for one epoch with mixed precision support"""
    
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            if USE_MIXED_PRECISION:
                # Mixed precision inference
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                # Standard precision inference
                output = model(data)
                loss = criterion(output, target)
            
            # Calculate Top-1 and Top-5 accuracy
            _, pred_top5 = output.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_batch = pred_top5.eq(target.view(1, -1).expand_as(pred_top5))
            
            # Top-1 accuracy
            correct_top1 += correct_batch[:1].reshape(-1).float().sum(0).item()
            
            # Top-5 accuracy
            correct_top5 += correct_batch[:5].reshape(-1).float().sum(0).item()
            
            running_loss += loss.item()
            total += target.size(0)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_top1_acc = 100. * correct_top1 / total
    epoch_top5_acc = 100. * correct_top5 / total
    
    # Log validation summary
    precision_str = "FP16" if USE_MIXED_PRECISION else "FP32"
    logging.info(f"Validation: Loss={epoch_loss:.4f}, Top-1 Acc={epoch_top1_acc:.2f}%, Top-5 Acc={epoch_top5_acc:.2f}% ({precision_str})")
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc

# ============================================================================
# VISUALIZATION AND LOGGING FUNCTIONS
# ============================================================================

def plot_training_progress(train_losses: List[float], train_accs: List[float], 
                          val_losses: List[float], val_accs: List[float], 
                          learning_rates: List[float]):
    """Plot comprehensive training progress"""
    
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].axhline(y=TARGET_ACCURACY, color='g', linestyle='--', 
                       label=f'Target: {TARGET_ACCURACY}%')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1, 0].plot(epochs, learning_rates, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting analysis
    overfitting_gap = [train_acc - val_acc for train_acc, val_acc in zip(train_accs, val_accs)]
    axes[1, 1].plot(epochs, overfitting_gap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (%)')
    axes[1, 1].set_title('Overfitting Analysis (Train - Val Accuracy)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_training_results(train_losses: List[float], train_accs: List[float], 
                         val_losses: List[float], val_accs: List[float], 
                         learning_rates: List[float], optimal_lr: float, 
                         best_val_acc: float, total_time: float):
    """Save training results to JSON file"""
    
    results = {
        'configuration': {
            'model': 'ResNet50',
            'dataset': 'Mini-ImageNet',
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'target_accuracy': TARGET_ACCURACY,
            'optimal_lr': optimal_lr,
            'dropout_rate': DROPOUT_RATE,
            'weight_decay': WEIGHT_DECAY,
            'label_smoothing': LABEL_SMOOTHING
        },
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'learning_rates': learning_rates
        },
        'final_results': {
            'best_validation_accuracy': best_val_acc,
            'final_train_accuracy': train_accs[-1] if train_accs else 0,
            'final_val_accuracy': val_accs[-1] if val_accs else 0,
            'total_training_time': total_time,
            'epochs_completed': len(train_losses),
            'target_achieved': best_val_acc >= TARGET_ACCURACY
        }
    }
    
    with open(os.path.join(SAVE_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def print_training_summary(train_losses: List[float], train_accs: List[float], 
                          val_losses: List[float], val_accs: List[float], 
                          optimal_lr: float, best_val_acc: float, total_time: float):
    """Print comprehensive training summary"""
    
    print("\n" + "="*80)
    print("🎯 TRAINING SUMMARY")
    print("="*80)
    
    print(f"📊 Model Architecture: ResNet50")
    print(f"📊 Dataset: Mini-ImageNet ({NUM_CLASSES} classes)")
    print(f"📊 Total Training Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"📊 Epochs Completed: {len(train_losses)}")
    print(f"📊 Optimal Learning Rate: {optimal_lr:.2e}")
    
    print(f"\n🎯 PERFORMANCE METRICS:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Final Training Accuracy: {train_accs[-1]:.2f}%")
    print(f"   Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print(f"   Target Accuracy ({TARGET_ACCURACY}%): {'✅ ACHIEVED' if best_val_acc >= TARGET_ACCURACY else '❌ NOT ACHIEVED'}")
    
    if len(val_accs) >= 2:
        improvement = val_accs[-1] - val_accs[0]
        print(f"   Total Improvement: {improvement:+.2f}%")
    
    print(f"\n📈 TRAINING ANALYSIS:")
    if len(train_accs) > 0 and len(val_accs) > 0:
        overfitting_gap = train_accs[-1] - val_accs[-1]
        print(f"   Overfitting Gap: {overfitting_gap:.2f}%")
        
        if overfitting_gap > 10:
            print("   ⚠️  High overfitting detected - consider more regularization")
        elif overfitting_gap < 2:
            print("   ✅ Good generalization - low overfitting")
        else:
            print("   ✅ Moderate overfitting - acceptable range")
    
    print(f"\n💾 Results saved to: {SAVE_DIR}/")
    print("   - training_progress.png (visualization)")
    print("   - training_results.json (detailed metrics)")
    print("   - lr_finder.png (learning rate analysis)")
    print("="*80)

def check_early_progress(val_accs: List[float], epoch: int) -> bool:
    """Check if training is progressing well in early epochs"""
    
    if epoch < MIN_EPOCHS_FOR_FEEDBACK:
        return True
    
    if len(val_accs) < MIN_EPOCHS_FOR_FEEDBACK:
        return True
    
    # Check if validation accuracy is improving
    recent_accs = val_accs[-MIN_EPOCHS_FOR_FEEDBACK:]
    initial_acc = recent_accs[0]
    current_acc = recent_accs[-1]
    improvement = current_acc - initial_acc
    
    if improvement < 5.0:  # Less than 5% improvement in 6 epochs
        print(f"\n⚠️  WARNING: Low progress detected!")
        print(f"   Validation accuracy improved by only {improvement:.2f}% in {MIN_EPOCHS_FOR_FEEDBACK} epochs")
        print(f"   Current accuracy: {current_acc:.2f}%")
        print(f"   Consider adjusting hyperparameters if this continues...")
        return False
    
    return True

# ============================================================================
# CHECKPOINT MANAGEMENT FUNCTIONS
# ============================================================================

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler, 
                   epoch: int, train_losses: List[float], train_accs: List[float],
                   val_losses: List[float], val_accs: List[float], 
                   learning_rates: List[float], best_val_acc: float,
                   checkpoint_dir: str, is_best: bool = False):
    """
    Save training checkpoint
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The learning rate scheduler
        epoch: Current epoch
        train_losses: Training loss history
        train_accs: Training accuracy history
        val_losses: Validation loss history
        val_accs: Validation accuracy history
        learning_rates: Learning rate history
        best_val_acc: Best validation accuracy so far
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'config': {
            'num_classes': NUM_CLASSES,
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'dropout_rate': DROPOUT_RATE,
            'optimizer_type': OPTIMIZER_TYPE,
            'initial_lr': INITIAL_LR,
            'weight_decay': WEIGHT_DECAY,
            'momentum': MOMENTUM
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save best model checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model_checkpoint.pth')
        torch.save(checkpoint, best_path)
        print(f"🏆 Best model checkpoint saved: {best_path}")
    
    # Save latest checkpoint (for easy resuming)
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None, 
                   scheduler = None) -> Dict:
    """
    Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Dict containing checkpoint information
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model state loaded from epoch {checkpoint['epoch']}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✅ Scheduler state loaded")
    
    return checkpoint

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the checkpoint directory
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Fallback: find highest epoch checkpoint
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('_')[2].split('.')[0])
            epochs.append((epoch, f))
        except:
            continue
    
    if epochs:
        latest_epoch, latest_file = max(epochs)
        return os.path.join(checkpoint_dir, latest_file)
    
    return None

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function"""
    
    # Setup logging first
    logger = setup_logging()
    
    dataset_name = "Tiny-ImageNet" if 'tiny-imagenet' in DATASET_PATH.lower() else "ImageNet"
    print(f"🚀 Starting ResNet50 Training on {dataset_name}")
    print("="*60)
    
    logger.info(f"Starting ResNet50 Training on {dataset_name}")
    logger.info("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Data loading
    print(f"\n📁 Loading dataset from: {DATASET_PATH}")
    train_loader, val_loader, actual_num_classes = create_data_loaders()
    
    # Update global NUM_CLASSES if different
    global NUM_CLASSES
    if NUM_CLASSES != actual_num_classes:
        print(f"⚠️  Updating NUM_CLASSES from {NUM_CLASSES} to {actual_num_classes}")
        NUM_CLASSES = actual_num_classes
    
    # Model setup
    print(f"\n🏗️  Building ResNet50 model...")
    model = ResNet50(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Setup training components with stable optimizer
    print(f"\n🎯 Using stable optimizer configuration")
    print(f"   Optimizer: {OPTIMIZER_TYPE}")
    print(f"   Learning rate: {INITIAL_LR:.2e}")
    print(f"   Weight decay: {WEIGHT_DECAY:.2e}")
    
    # Mixed precision setup
    scaler = None
    if USE_MIXED_PRECISION and device.type == 'cuda':
        scaler = GradScaler()
        print(f"   Mixed Precision: Enabled (FP16) - Optimized for T4 GPU")
    else:
        print(f"   Mixed Precision: Disabled (FP32)")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = create_optimizer(model.parameters(), OPTIMIZER_TYPE, INITIAL_LR, MOMENTUM, WEIGHT_DECAY)
    
    # Warmup + cosine annealing for faster convergence with higher LR
    warmup_epochs = 2
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS-warmup_epochs, eta_min=INITIAL_LR/100)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    # Check for checkpoint to resume from
    start_epoch = 1
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    learning_rates = []
    best_val_acc = 0.0
    patience_counter = 0
    
    # Try to resume from checkpoint
    resume_path = RESUME_FROM_CHECKPOINT
    if resume_path is None:
        resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
    
    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_losses = checkpoint.get('val_losses', [])
            val_accs = checkpoint.get('val_accs', [])
            learning_rates = checkpoint.get('learning_rates', [])
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            
            print(f"🔄 Resuming training from epoch {start_epoch}")
            print(f"🏆 Best validation accuracy so far: {best_val_acc:.2f}%")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("🔄 Starting fresh training...")
            start_epoch = 1
    else:
        print("🆕 Starting fresh training...")
    
    print(f"\n🎯 Training from epoch {start_epoch} to {MAX_EPOCHS}")
    print(f"🎯 Target accuracy: {TARGET_ACCURACY}%")
    start_time = time.time()
    
    # Training loop
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        print(f"\n📈 Epoch {epoch}/{MAX_EPOCHS}")
        print("-" * 50)
        
        # Training
        train_loss, train_top1, train_top5 = train_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        
        # Validation
        val_loss, val_top1, val_top5 = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(current_lr)
        
        # Print epoch results
        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"📊 Learning Rate: {current_lr:.2e}")
        
        # Check for best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"✅ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % SAVE_CHECKPOINT_EVERY == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_losses, train_accs, val_losses, val_accs,
                learning_rates, best_val_acc, CHECKPOINT_DIR, is_best
            )
        
        # Early stopping check
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️  Early stopping triggered (patience: {EARLY_STOP_PATIENCE})")
            # Save final checkpoint before stopping
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_losses, train_accs, val_losses, val_accs,
                learning_rates, best_val_acc, CHECKPOINT_DIR, False
            )
            break
        
        # Target accuracy check
        if val_acc >= TARGET_ACCURACY:
            print(f"\n🎉 Target accuracy {TARGET_ACCURACY}% achieved!")
            # Save final checkpoint
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_losses, train_accs, val_losses, val_accs,
                learning_rates, best_val_acc, CHECKPOINT_DIR, True
            )
            break
        
        # Early progress check
        if not check_early_progress(val_accs, epoch):
            continue  # Just warn, don't stop
    
    # Training completed
    total_time = time.time() - start_time
    
    # Generate final visualizations and summary
    plot_training_progress(train_losses, train_accs, val_losses, val_accs, learning_rates)
    save_training_results(train_losses, train_accs, val_losses, val_accs, 
                         learning_rates, INITIAL_LR, best_val_acc, total_time)
    print_training_summary(train_losses, train_accs, val_losses, val_accs, 
                          INITIAL_LR, best_val_acc, total_time)

if __name__ == "__main__":
    main()