"""
Main training script for ResNet CIFAR-100 using PyTorch Lightning.
Provides clean, modular training with advanced features.
"""

import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from config import Config
from datamodule import CIFAR100DataModule
from model import LightningResNet
from utils import set_seed, print_system_info, setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ResNet CIFAR-100 Training with PyTorch Lightning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training arguments
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, help='Weight decay')
    
    # Model arguments
    parser.add_argument('--model-depth', type=int, choices=[20, 32, 44, 56, 110],
                       help='ResNet depth')
    parser.add_argument('--dropout', type=float, help='Dropout probability')
    
    # System arguments
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--precision', type=int, choices=[16, 32],
                       help='Training precision')
    parser.add_argument('--num-workers', type=int, help='Number of data workers')
    
    # Experiment arguments
    parser.add_argument('--name', type=str, default='resnet_cifar100',
                       help='Experiment name')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing')
    parser.add_argument('--fast-dev-run', action='store_true',
                       help='Fast development run')
    
    # Logging arguments
    parser.add_argument('--log-every-n-steps', type=int, default=50,
                       help='Log every n steps')
    parser.add_argument('--save-top-k', type=int, default=3,
                       help='Save top k checkpoints')
    
    return parser.parse_args()


def create_callbacks(config: Config, args: argparse.Namespace) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.system.checkpoint_dir,
        filename=f'{args.name}-{{epoch:02d}}-{{val_acc:.2f}}',
        monitor='val_acc',
        mode='max',
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=20,
        verbose=True,
        min_delta=0.001
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Rich progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks


def create_trainer(config: Config, args: argparse.Namespace) -> L.Trainer:
    """Create PyTorch Lightning trainer."""
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.system.log_dir,
        name=args.name,
        version=None
    )
    
    # Callbacks
    callbacks = create_callbacks(config, args)
    
    # Strategy for multi-GPU
    strategy = "auto"
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.epochs,
        precision=config.system.precision,
        accelerator="auto",
        devices="auto",
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=args.fast_dev_run,
        deterministic=True
    )
    
    return trainer


def override_config_with_args(config: Config, args: argparse.Namespace) -> Config:
    """Override configuration with command line arguments."""
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.model_depth is not None:
        config.model.depth = args.model_depth
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.device is not None:
        config.system.device = args.device
    if args.precision is not None:
        config.system.precision = args.precision
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        # TODO: Implement config loading from file
        config = Config()
    else:
        config = Config.from_env()
    
    # Override config with command line arguments
    config = override_config_with_args(config, args)
    
    # Set up logging and reproducibility
    setup_logging()
    set_seed(config.system.seed)
    
    # Print system information
    print_system_info()
    config.print_config()
    
    # Create data module
    print("🔄 Setting up data module...")
    datamodule = CIFAR100DataModule(config)
    
    # Create model
    print(f"🏗️  Creating {config.model.name} model...")
    model = LightningResNet(config)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("🚀 Setting up trainer...")
    trainer = create_trainer(config, args)
    
    # Resume from checkpoint if specified
    ckpt_path = None
    if args.resume:
        ckpt_path = args.resume
        print(f"📂 Resuming from checkpoint: {ckpt_path}")
    
    # Test only mode
    if args.test_only:
        print("🧪 Running test evaluation...")
        if not ckpt_path:
            print("❌ No checkpoint specified for testing!")
            return
        
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
        return
    
    # Start training
    print("🎯 Starting training...")
    print(f"📈 Target accuracy: {config.training.target_accuracy}%")
    print("=" * 80)
    
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        
        # Test with best checkpoint
        print("\n🏆 Training completed! Running final test...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
        
        # Print results
        best_val_acc = trainer.callback_metrics.get('val_acc', 0.0) * 100
        test_acc = trainer.callback_metrics.get('test_acc', 0.0) * 100
        
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS")
        print("=" * 80)
        print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
        print(f"🧪 Final test accuracy: {test_acc:.2f}%")
        
        if best_val_acc >= config.training.target_accuracy:
            print(f"✅ Successfully achieved target accuracy of {config.training.target_accuracy}%!")
        else:
            print(f"❌ Did not reach target accuracy of {config.training.target_accuracy}%")
            print("💡 Consider adjusting hyperparameters or training longer")
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Checkpoints have been saved automatically")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()