"""
Comprehensive Configuration System for ImageNet Training
======================================================

This script provides a centralized configuration management system that:
1. Consolidates all training parameters in one location
2. Automatically stores and loads computed statistics and optimal learning rates
3. Maintains backward compatibility while streamlining configuration
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Centralized configuration storage directory
CONFIG_STORAGE_DIR = "./config_storage"
COMPUTED_STATS_FILE = "computed_stats.json"
OPTIMAL_LR_FILE = "optimal_lr.json"

# Comprehensive configuration presets
CONFIGS = {
    'tiny-imagenet': {
        # Dataset Configuration
        'DATASET_PATH': '"/data/tiny-imagenet-200"',
        'NUM_CLASSES': '200',
        'IMAGE_SIZE': '64',
        'VALIDATION_SPLIT': '0.2',
        
        # Training Configuration
        'BATCH_SIZE': '16',
        'NUM_WORKERS': '4',
        'MAX_EPOCHS': '50',
        'EARLY_STOP_PATIENCE': '10',
        'MIN_EPOCHS_FOR_FEEDBACK': '6',
        'TARGET_ACCURACY': '80.0',
        
        # Model Configuration
        'DROPOUT_RATE': '0.1',
        'LABEL_SMOOTHING': '0.0',
        
        # Optimizer Configuration
        'OPTIMIZER_TYPE': "'AdamW'",
        'INITIAL_LR': '1e-3',
        'MOMENTUM': '0.9',
        'WEIGHT_DECAY': '1e-5',
        
        # Mixed Precision
        'USE_MIXED_PRECISION': 'True',
        
        # Logging and Visualization
        'SAVE_DIR': '"./results"',
        'LOG_INTERVAL': '10',
        'PLOT_STYLE': "'seaborn-v0_8'",
        'PLOT_FREQUENCY': '5',
        
        # Checkpointing
        'CHECKPOINT_DIR': '"./checkpoints"',
        'SAVE_CHECKPOINT_EVERY': '5',
        'RESUME_FROM_CHECKPOINT': 'None',
        
        # LR Finder Configuration
        'LR_FINDER_START_LR': '1e-7',
        'LR_FINDER_END_LR': '10',
        'LR_FINDER_NUM_ITER': '100',
        
        # Computed Statistics (auto-populated)
        'DATASET_MEAN': 'None',  # Will be auto-populated
        'DATASET_STD': 'None',   # Will be auto-populated
        'OPTIMAL_LR_COMPUTED': 'None',  # Will be auto-populated
    },
    
    'imagenet': {
        # Dataset Configuration
        'DATASET_PATH': '"/lambda/nfs/ERAv4S09/imagenet/full_dataset"',
        'NUM_CLASSES': '1000',
        'IMAGE_SIZE': '224',
        'VALIDATION_SPLIT': '0.2',
        
        # Training Configuration
        'BATCH_SIZE': '16',   # Optimized for T4 GPU memory constraints
        'NUM_WORKERS': '4',   # Matched to 4 vCPUs on g4dn.xlarge
        'MAX_EPOCHS': '100',
        'EARLY_STOP_PATIENCE': '15',
        'MIN_EPOCHS_FOR_FEEDBACK': '10',
        'TARGET_ACCURACY': '75.0',
        
        # Model Configuration
        'DROPOUT_RATE': '0.2',
        'LABEL_SMOOTHING': '0.1',
        
        # Optimizer Configuration
        'OPTIMIZER_TYPE': "'SGD'",
        'INITIAL_LR': '1.20e-05',  # Optimal LR found by LR range test
        'MOMENTUM': '0.9',
        'WEIGHT_DECAY': '1e-3',
        
        # Mixed Precision
        'USE_MIXED_PRECISION': 'True',  # Enable FP16 for T4 GPU optimization
        
        # Logging and Visualization
        'SAVE_DIR': '"./results"',
        'LOG_INTERVAL': '10',
        'PLOT_STYLE': "'seaborn-v0_8'",
        'PLOT_FREQUENCY': '5',
        
        # Checkpointing
        'CHECKPOINT_DIR': '"./checkpoints"',
        'SAVE_CHECKPOINT_EVERY': '5',
        'RESUME_FROM_CHECKPOINT': 'None',
        
        # LR Finder Configuration
        'LR_FINDER_START_LR': '1e-7',
        'LR_FINDER_END_LR': '10',
        'LR_FINDER_NUM_ITER': '50',  # Reduced for memory efficiency
        
        # Computed Statistics (auto-populated)
        'DATASET_MEAN': 'None',  # Will be auto-populated
        'DATASET_STD': 'None',   # Will be auto-populated
        'OPTIMAL_LR_COMPUTED': 'None',  # Will be auto-populated
    }
}

class ConfigManager:
    """Centralized configuration manager with automated storage/loading capabilities"""
    
    def __init__(self):
        self.storage_dir = Path(CONFIG_STORAGE_DIR)
        self.storage_dir.mkdir(exist_ok=True)
        
    def store_computed_stats(self, config_name: str, mean: list, std: list) -> None:
        """Store computed dataset statistics"""
        stats_file = self.storage_dir / f"{config_name}_{COMPUTED_STATS_FILE}"
        stats_data = {
            'config_name': config_name,
            'computed_at': time.time(),
            'mean': mean,
            'std': std,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"📊 Stored computed statistics for {config_name}")
        print(f"   Mean: {mean}")
        print(f"   Std: {std}")
    
    def load_computed_stats(self, config_name: str) -> Optional[Tuple[list, list]]:
        """Load previously computed dataset statistics"""
        stats_file = self.storage_dir / f"{config_name}_{COMPUTED_STATS_FILE}"
        
        if not stats_file.exists():
            return None
            
        try:
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
            
            print(f"📊 Loaded computed statistics for {config_name}")
            print(f"   Computed at: {stats_data['timestamp']}")
            print(f"   Mean: {stats_data['mean']}")
            print(f"   Std: {stats_data['std']}")
            
            return stats_data['mean'], stats_data['std']
        except Exception as e:
            print(f"⚠️ Error loading computed statistics: {e}")
            return None
    
    def store_optimal_lr(self, config_name: str, optimal_lr: float, lr_finder_results: dict = None) -> None:
        """Store optimal learning rate from LR finder"""
        lr_file = self.storage_dir / f"{config_name}_{OPTIMAL_LR_FILE}"
        lr_data = {
            'config_name': config_name,
            'computed_at': time.time(),
            'optimal_lr': optimal_lr,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'lr_finder_results': lr_finder_results or {}
        }
        
        with open(lr_file, 'w') as f:
            json.dump(lr_data, f, indent=2)
        
        print(f"🎯 Stored optimal learning rate for {config_name}")
        print(f"   Optimal LR: {optimal_lr:.2e}")
    
    def load_optimal_lr(self, config_name: str) -> Optional[float]:
        """Load previously computed optimal learning rate"""
        lr_file = self.storage_dir / f"{config_name}_{OPTIMAL_LR_FILE}"
        
        if not lr_file.exists():
            return None
            
        try:
            with open(lr_file, 'r') as f:
                lr_data = json.load(f)
            
            print(f"🎯 Loaded optimal learning rate for {config_name}")
            print(f"   Computed at: {lr_data['timestamp']}")
            print(f"   Optimal LR: {lr_data['optimal_lr']:.2e}")
            
            return lr_data['optimal_lr']
        except Exception as e:
            print(f"⚠️ Error loading optimal learning rate: {e}")
            return None
    
    def get_enhanced_config(self, config_name: str, force_recompute: bool = False) -> Dict[str, Any]:
        """Get configuration with auto-loaded computed values"""
        if config_name not in CONFIGS:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        config = CONFIGS[config_name].copy()
        
        # Load computed statistics if available and not forcing recompute
        if not force_recompute:
            stats = self.load_computed_stats(config_name)
            if stats:
                mean, std = stats
                config['DATASET_MEAN'] = str(mean)
                config['DATASET_STD'] = str(std)
        
        # Load optimal learning rate if available and not forcing recompute
        if not force_recompute:
            optimal_lr = self.load_optimal_lr(config_name)
            if optimal_lr:
                config['OPTIMAL_LR_COMPUTED'] = str(optimal_lr)
                # Update INITIAL_LR with computed optimal LR
                config['INITIAL_LR'] = str(optimal_lr)
        
        return config

def update_config(config_name: str, force_recompute: bool = False) -> bool:
    """Update the training script with the specified configuration"""
    
    if config_name not in CONFIGS:
        print(f"❌ Unknown configuration: {config_name}")
        print(f"Available configurations: {list(CONFIGS.keys())}")
        return False
    
    config_manager = ConfigManager()
    config = config_manager.get_enhanced_config(config_name, force_recompute)
    script_path = "train_resnet50.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    # Read the current script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update each configuration parameter
    updated_params = []
    for param, value in config.items():
        # Skip auto-populated parameters that don't exist in the script
        if param in ['DATASET_MEAN', 'DATASET_STD', 'OPTIMAL_LR_COMPUTED']:
            continue
            
        # Find the line with this parameter
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{param} ='):
                # Update the line
                comment_part = ''
                if '#' in line:
                    comment_part = ' ' + line.split('#', 1)[1]
                lines[i] = f'{param} = {value}  #{comment_part}' if comment_part else f'{param} = {value}'
                updated_params.append(param)
                break
        content = '\n'.join(lines)
    
    # Write the updated script
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Configuration updated to: {config_name}")
    print(f"📝 Updated parameters: {len(updated_params)}")
    
    # Show which computed values were loaded
    if config.get('DATASET_MEAN') != 'None':
        print("📊 Using pre-computed dataset statistics")
    if config.get('OPTIMAL_LR_COMPUTED') != 'None':
        print("🎯 Using pre-computed optimal learning rate")
    
    return True

def show_current_config():
    """Show the current configuration with enhanced details"""
    script_path = "train_resnet50.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return
    
    print("📋 Current Configuration:")
    print("=" * 50)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Define parameter categories
    categories = {
        'Dataset': ['DATASET_PATH', 'NUM_CLASSES', 'IMAGE_SIZE', 'VALIDATION_SPLIT'],
        'Training': ['BATCH_SIZE', 'NUM_WORKERS', 'MAX_EPOCHS', 'EARLY_STOP_PATIENCE', 'TARGET_ACCURACY'],
        'Model': ['DROPOUT_RATE', 'LABEL_SMOOTHING'],
        'Optimizer': ['OPTIMIZER_TYPE', 'INITIAL_LR', 'MOMENTUM', 'WEIGHT_DECAY'],
        'System': ['USE_MIXED_PRECISION', 'SAVE_DIR', 'CHECKPOINT_DIR']
    }
    
    # Extract and display current values by category
    for category, params in categories.items():
        print(f"\n{category} Configuration:")
        print("-" * 30)
        for param in params:
            for line in content.split('\n'):
                if line.strip().startswith(f'{param} ='):
                    value = line.split('=', 1)[1].split('#')[0].strip()
                    print(f"  {param}: {value}")
                    break
    
    # Show stored computed values
    config_manager = ConfigManager()
    print(f"\n🔍 Stored Computed Values:")
    print("-" * 30)
    
    for config_name in CONFIGS.keys():
        print(f"\n{config_name.upper()}:")
        stats = config_manager.load_computed_stats(config_name)
        optimal_lr = config_manager.load_optimal_lr(config_name)
        
        if stats:
            print(f"  📊 Dataset Stats: Available")
        else:
            print(f"  📊 Dataset Stats: Not computed")
            
        if optimal_lr:
            print(f"  🎯 Optimal LR: {optimal_lr:.2e}")
        else:
            print(f"  🎯 Optimal LR: Not computed")

def main():
    """Main function with enhanced commands"""
    print("🔧 Comprehensive ImageNet Training Configuration System")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python config_switcher.py [command] [options]")
        print()
        print("Commands:")
        print("  tiny-imagenet     - Switch to Tiny-ImageNet configuration")
        print("  imagenet          - Switch to full ImageNet configuration")
        print("  show              - Show current configuration and stored values")
        print("  store-stats       - Store computed dataset statistics")
        print("  store-lr          - Store optimal learning rate")
        print("  clear-cache       - Clear all stored computed values")
        print()
        print("Options:")
        print("  --force-recompute - Force recomputation of cached values")
        return
    
    command = sys.argv[1].lower()
    force_recompute = '--force-recompute' in sys.argv
    
    config_manager = ConfigManager()
    
    if command == 'show':
        show_current_config()
    elif command in CONFIGS:
        if update_config(command, force_recompute):
            print()
            print("📝 Next steps:")
            if command == 'tiny-imagenet':
                print("1. Download Tiny-ImageNet: python download_tiny_imagenet.py")
            else:
                print("1. Download ImageNet: python download_imagenet.py")
            print("2. Compute stats: python train_resnet50.py --compute-stats")
            print("3. Find optimal LR: python train_resnet50.py --lr-finder")
            print("4. Start training: python train_resnet50.py")
    elif command == 'store-stats':
        print("Use this command from within the training script after computing stats")
    elif command == 'store-lr':
        print("Use this command from within the training script after LR finder")
    elif command == 'clear-cache':
        import shutil
        if config_manager.storage_dir.exists():
            shutil.rmtree(config_manager.storage_dir)
            config_manager.storage_dir.mkdir(exist_ok=True)
            print("🗑️ Cleared all stored computed values")
        else:
            print("📁 No cache to clear")
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: tiny-imagenet, imagenet, show, store-stats, store-lr, clear-cache")

if __name__ == "__main__":
    main()