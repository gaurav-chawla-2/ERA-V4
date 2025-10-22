"""
Data loading and augmentation for CIFAR-100
Implements advanced data augmentation techniques for better accuracy
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, random_split
from config import Config

class Cutout:
    """Cutout data augmentation"""
    def __init__(self, length):
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        
        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_cifar100_transforms(config):
    """Get training and test transforms for CIFAR-100"""
    
    # Calculate normalization values for CIFAR-100
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
        std=[0.2675, 0.2565, 0.2761]   # CIFAR-100 std
    )
    
    # Enhanced training transforms with more augmentation
    train_transforms = [
        transforms.RandomCrop(32, padding=config.RANDOM_CROP_PADDING),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    
    # Add rotation if enabled
    if hasattr(config, 'RANDOM_ROTATION') and config.RANDOM_ROTATION:
        train_transforms.append(
            transforms.RandomRotation(degrees=config.ROTATION_DEGREES, fill=0)
        )
    
    # Add color jitter if enabled
    if hasattr(config, 'COLOR_JITTER') and config.COLOR_JITTER:
        train_transforms.append(
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        )
    
    # Add standard transforms
    train_transforms.extend([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Add cutout if enabled
    if config.CUTOUT:
        train_transforms.append(Cutout(config.CUTOUT_LENGTH))
    
    train_transform = transforms.Compose(train_transforms)
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, test_transform

def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_cifar100_loaders(config):
    """Create CIFAR-100 data loaders"""
    
    train_transform, test_transform = get_cifar100_transforms(config)
    
    # Download and load CIFAR-100 dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create validation split from training data
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Create validation dataset with test transforms
    val_indices = val_dataset.indices
    val_dataset_clean = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=False, transform=test_transform
    )
    val_dataset = torch.utils.data.Subset(val_dataset_clean, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader

def get_class_names():
    """Get CIFAR-100 class names"""
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

if __name__ == "__main__":
    # Test data loading
    config = Config()
    train_loader, val_loader, test_loader = get_cifar100_loaders(config)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test a batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        break