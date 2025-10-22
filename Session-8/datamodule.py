"""
Data module for CIFAR-100 dataset
Implements modern data loading patterns with comprehensive augmentation
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional
import logging

from config import ExperimentConfig


class Cutout:
    """Cutout data augmentation technique"""
    
    def __init__(self, length: int):
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout to image"""
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
        img = img * mask
        return img


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation to batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CIFAR100DataModule:
    """CIFAR-100 data module with advanced augmentation"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # CIFAR-100 statistics
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get training and validation transforms"""
        
        # Training transforms with augmentation
        train_transforms = [
            transforms.RandomCrop(32, padding=self.config.augmentation.random_crop_padding),
        ]
        
        if self.config.augmentation.random_horizontal_flip:
            train_transforms.append(transforms.RandomHorizontalFlip())
        
        if self.config.augmentation.random_rotation:
            train_transforms.append(
                transforms.RandomRotation(self.config.augmentation.rotation_degrees)
            )
        
        if self.config.augmentation.color_jitter:
            train_transforms.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        if self.config.augmentation.cutout:
            train_transforms.append(Cutout(self.config.augmentation.cutout_length))
        
        # Validation transforms (no augmentation)
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        return transforms.Compose(train_transforms), val_transforms
    
    def prepare_data(self):
        """Download CIFAR-100 dataset"""
        self.logger.info("Downloading CIFAR-100 dataset...")
        torchvision.datasets.CIFAR100(
            root=self.config.data.data_dir,
            train=True,
            download=True
        )
        torchvision.datasets.CIFAR100(
            root=self.config.data.data_dir,
            train=False,
            download=True
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing"""
        train_transform, val_transform = self.get_transforms()
        
        if stage == 'fit' or stage is None:
            # Full training dataset
            full_train = torchvision.datasets.CIFAR100(
                root=self.config.data.data_dir,
                train=True,
                transform=train_transform
            )
            
            # Split into train and validation
            train_size = int(0.9 * len(full_train))
            val_size = len(full_train) - train_size
            
            self.train_dataset, val_dataset_temp = random_split(
                full_train, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config.system.seed)
            )
            
            # Create validation dataset with validation transforms
            val_dataset_full = torchvision.datasets.CIFAR100(
                root=self.config.data.data_dir,
                train=True,
                transform=val_transform
            )
            
            # Get the same validation indices
            val_indices = val_dataset_temp.indices
            self.val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
        
        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR100(
                root=self.config.data.data_dir,
                train=False,
                transform=val_transform
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True if self.config.data.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True if self.config.data.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            persistent_workers=True if self.config.data.num_workers > 0 else False
        )


def get_cifar100_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Legacy function for backward compatibility"""
    if hasattr(config, 'data'):
        # New config format
        exp_config = config
    else:
        # Legacy config format - convert to new format
        from config import get_config
        exp_config = get_config()
        exp_config.training.batch_size = config.BATCH_SIZE
        exp_config.data.num_workers = config.NUM_WORKERS
        exp_config.data.pin_memory = config.PIN_MEMORY
        exp_config.system.seed = config.SEED
    
    datamodule = CIFAR100DataModule(exp_config)
    datamodule.prepare_data()
    datamodule.setup()
    
    return (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader()
    )


def get_class_names() -> list:
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