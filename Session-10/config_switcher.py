"""
Configuration Switcher for ImageNet Training
===========================================

This script helps switch between Tiny-ImageNet and full ImageNet configurations
by modifying the train_resnet50.py file.
"""

import os
import sys
from pathlib import Path

# Configuration presets
CONFIGS = {
    'tiny-imagenet': {
        'DATASET_PATH': '"/data/tiny-imagenet-200"',
        'NUM_CLASSES': '200',
        'IMAGE_SIZE': '64',
        'BATCH_SIZE': '16',
        'MAX_EPOCHS': '50',
        'EARLY_STOP_PATIENCE': '10',
        'MIN_EPOCHS_FOR_FEEDBACK': '6',
        'TARGET_ACCURACY': '80.0',
        'DROPOUT_RATE': '0.1',
        'LABEL_SMOOTHING': '0.0',
        'OPTIMIZER_TYPE': "'AdamW'",
        'INITIAL_LR': '1e-3',
        'WEIGHT_DECAY': '1e-5'
    },
    'imagenet': {
        'DATASET_PATH': '"/data/imagenet/full_dataset"',  # Use dedicated /data mount (280GB available)
        'NUM_CLASSES': '1000',
        'IMAGE_SIZE': '224',
        'BATCH_SIZE': '32',   # Conservative for T4 GPU memory constraints on g4dn.xlarge
        'NUM_WORKERS': '4',   # Matched to 4 vCPUs on g4dn.xlarge
        'MAX_EPOCHS': '100',
        'EARLY_STOP_PATIENCE': '15',
        'MIN_EPOCHS_FOR_FEEDBACK': '10',
        'TARGET_ACCURACY': '75.0',
        'DROPOUT_RATE': '0.2',
        'LABEL_SMOOTHING': '0.1',
        'OPTIMIZER_TYPE': "'SGD'",
        'INITIAL_LR': '0.01',
        'WEIGHT_DECAY': '1e-3',
        'USE_MIXED_PRECISION': 'True'  # Enable FP16 for T4 GPU optimization
    }
}

def update_config(config_name):
    """Update the training script with the specified configuration"""
    
    if config_name not in CONFIGS:
        print(f"❌ Unknown configuration: {config_name}")
        print(f"Available configurations: {list(CONFIGS.keys())}")
        return False
    
    config = CONFIGS[config_name]
    script_path = "train_resnet50.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    # Read the current script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update each configuration parameter
    for param, value in config.items():
        # Find the line with this parameter
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(f'{param} ='):
                # Update the line
                comment_part = ''
                if '#' in line:
                    comment_part = ' ' + line.split('#', 1)[1]
                lines[i] = f'{param} = {value}  #{comment_part}' if comment_part else f'{param} = {value}'
                break
        content = '\n'.join(lines)
    
    # Write the updated script
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Configuration updated to: {config_name}")
    return True

def show_current_config():
    """Show the current configuration"""
    script_path = "train_resnet50.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return
    
    print("📋 Current Configuration:")
    print("-" * 30)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Extract current values
    for param in ['DATASET_PATH', 'NUM_CLASSES', 'IMAGE_SIZE', 'BATCH_SIZE', 
                  'MAX_EPOCHS', 'TARGET_ACCURACY', 'OPTIMIZER_TYPE', 'INITIAL_LR']:
        for line in content.split('\n'):
            if line.strip().startswith(f'{param} ='):
                value = line.split('=', 1)[1].split('#')[0].strip()
                print(f"{param}: {value}")
                break

def main():
    """Main function"""
    print("🔧 ImageNet Training Configuration Switcher")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python config_switcher.py [tiny-imagenet|imagenet|show]")
        print()
        print("Commands:")
        print("  tiny-imagenet  - Switch to Tiny-ImageNet configuration")
        print("  imagenet       - Switch to full ImageNet configuration")
        print("  show           - Show current configuration")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'show':
        show_current_config()
    elif command in CONFIGS:
        if update_config(command):
            print()
            print("📝 Next steps:")
            if command == 'tiny-imagenet':
                print("1. Download Tiny-ImageNet: python download_tiny_imagenet.py")
            else:
                print("1. Download ImageNet: python download_imagenet.py")
            print("2. Verify setup: python verify_setup.py")
            print("3. Start training: python train_resnet50.py")
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: tiny-imagenet, imagenet, show")

if __name__ == "__main__":
    main()