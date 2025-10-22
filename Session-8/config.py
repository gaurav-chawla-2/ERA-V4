"""
Configuration management for ResNet CIFAR-100 training
Follows best practices from PyTorch Lightning and modern ML projects
"""

import torch
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Data-related configuration"""
    dataset: str = 'CIFAR100'
    num_classes: int = 100
    input_size: int = 32
    input_channels: int = 3
    data_dir: str = './data'
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_name: str = 'ResNet56'
    depth: int = 56
    num_classes: int = 100
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 128
    epochs: int = 150
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    
    # Learning rate scheduling
    lr_schedule: List[int] = None
    lr_gamma: float = 0.2
    use_cosine_annealing: bool = True
    cosine_t_max: int = 150
    cosine_eta_min: float = 1e-6
    
    def __post_init__(self):
        if self.lr_schedule is None:
            self.lr_schedule = [60, 90, 120]


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    random_rotation: bool = True
    rotation_degrees: int = 15
    color_jitter: bool = True
    cutout: bool = True
    cutout_length: int = 16
    mixup: bool = True
    mixup_alpha: float = 1.0


@dataclass
class SystemConfig:
    """System and environment configuration"""
    device: str = 'auto'
    seed: int = 42
    precision: int = 32
    deterministic: bool = True
    benchmark: bool = False
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_interval: int = 100
    save_model: bool = True
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    tensorboard_dir: str = './runs'
    save_top_k: int = 3
    monitor: str = 'val_acc'
    mode: str = 'max'
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    augmentation: AugmentationConfig = None
    system: SystemConfig = None
    logging: LoggingConfig = None
    
    # Experiment metadata
    experiment_name: str = 'resnet_cifar100'
    target_accuracy: float = 73.0
    description: str = 'ResNet-56 training on CIFAR-100 with advanced augmentation'
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
        if self.system is None:
            self.system = SystemConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


def get_config() -> ExperimentConfig:
    """Get default configuration"""
    return ExperimentConfig()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Legacy support for backward compatibility
class Config:
    """Legacy config class for backward compatibility"""
    def __init__(self):
        config = get_config()
        
        # Dataset settings
        self.DATASET = config.data.dataset
        self.NUM_CLASSES = config.data.num_classes
        self.INPUT_SIZE = config.data.input_size
        self.INPUT_CHANNELS = config.data.input_channels
        
        # Training settings
        self.BATCH_SIZE = config.training.batch_size
        self.EPOCHS = config.training.epochs
        self.LEARNING_RATE = config.training.learning_rate
        self.MOMENTUM = config.training.momentum
        self.WEIGHT_DECAY = config.training.weight_decay
        self.LABEL_SMOOTHING = config.training.label_smoothing
        
        # Model settings
        self.RESNET_DEPTH = config.model.depth
        
        # Augmentation settings
        self.RANDOM_CROP_PADDING = config.augmentation.random_crop_padding
        self.RANDOM_HORIZONTAL_FLIP = config.augmentation.random_horizontal_flip
        self.CUTOUT = config.augmentation.cutout
        self.CUTOUT_LENGTH = config.augmentation.cutout_length
        self.RANDOM_ROTATION = config.augmentation.random_rotation
        self.ROTATION_DEGREES = config.augmentation.rotation_degrees
        self.COLOR_JITTER = config.augmentation.color_jitter
        self.MIXUP = config.augmentation.mixup
        self.MIXUP_ALPHA = config.augmentation.mixup_alpha
        
        # Learning rate schedule
        self.LR_SCHEDULE = config.training.lr_schedule
        self.LR_GAMMA = config.training.lr_gamma
        self.USE_COSINE_ANNEALING = config.training.use_cosine_annealing
        self.COSINE_T_MAX = config.training.cosine_t_max
        self.COSINE_ETA_MIN = config.training.cosine_eta_min
        
        # Device settings
        self.DEVICE = torch.device(config.system.device)
        self.NUM_WORKERS = config.data.num_workers
        self.PIN_MEMORY = config.data.pin_memory
        
        # Logging
        self.LOG_INTERVAL = config.logging.log_interval
        self.SAVE_MODEL = config.logging.save_model
        self.CHECKPOINT_DIR = config.logging.checkpoint_dir
        self.LOG_DIR = config.logging.log_dir
        
        # Target accuracy
        self.TARGET_ACCURACY = config.target_accuracy
        
        # Reproducibility
        self.SEED = config.system.seed