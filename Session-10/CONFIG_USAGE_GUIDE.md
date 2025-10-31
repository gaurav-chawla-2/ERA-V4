# ERA-V4 Configuration System Usage Guide

## Overview

The ERA-V4 project now features a centralized configuration system that automatically stores and loads computed statistics and optimal learning rates, streamlining the training process while maintaining backward compatibility.

## Key Features

### 1. Centralized Configuration Management
- All training parameters are managed through `config_switcher.py`
- Predefined configurations for different datasets (tiny-imagenet, imagenet)
- Easy switching between configurations

### 2. Automatic Storage and Loading
- **Dataset Statistics**: Automatically stored when computed with `--compute-stats`
- **Optimal Learning Rates**: Automatically stored when found with `--lr-finder`
- **Automatic Loading**: Stored values are automatically loaded in subsequent training runs
- **Manual Override**: You can still manually override any parameter

### 3. Backward Compatibility
- All existing command-line arguments continue to work
- No breaking changes to existing workflows

## Configuration Commands

### Basic Configuration Management

```bash
# Show current configuration
python config_switcher.py show

# Switch to tiny-imagenet configuration
python config_switcher.py tiny-imagenet

# Switch to full imagenet configuration  
python config_switcher.py imagenet

# Show available configurations
python config_switcher.py list
```

### Advanced Configuration Management

```bash
# Store dataset statistics manually
python config_switcher.py store-stats --mean 0.485,0.456,0.406 --std 0.229,0.224,0.225

# Store optimal learning rate manually
python config_switcher.py store-lr --lr 1.2e-05

# Show stored cache
python config_switcher.py show-cache

# Clear all stored cache
python config_switcher.py clear-cache

# Clear only statistics
python config_switcher.py clear-cache --stats-only

# Clear only learning rate
python config_switcher.py clear-cache --lr-only
```

## Training Workflows

### 1. First-Time Setup (Recommended)

```bash
# Step 1: Set up configuration for your dataset
python config_switcher.py imagenet  # or tiny-imagenet

# Step 2: Compute and store dataset statistics
python train_resnet50.py --compute-stats

# Step 3: Find and store optimal learning rate
python train_resnet50.py --lr-finder

# Step 4: Start training (will automatically use stored values)
python train_resnet50.py
```

### 2. Quick Training (Using Stored Values)

```bash
# If you've already computed stats and found optimal LR:
python train_resnet50.py
# This will automatically load stored statistics and learning rate
```

### 3. Manual Override Training

```bash
# Force recomputation of statistics
python train_resnet50.py --compute-stats

# Force new LR finder run
python train_resnet50.py --lr-finder

# Use both stored values but run LR finder for comparison
python train_resnet50.py --lr-finder
```

### 4. Statistics and LR Finder Only

```bash
# Only compute statistics and exit
python train_resnet50.py --stats-only

# Only run LR finder and exit
python train_resnet50.py --lr-finder-only
```

## Configuration Files

### Available Configurations

#### Tiny-ImageNet Configuration
```python
'tiny-imagenet': {
    'DATASET_PATH': '/data/tiny-imagenet-200',
    'NUM_CLASSES': 200,
    'IMAGE_SIZE': 64,
    'BATCH_SIZE': 32,
    'NUM_WORKERS': 4,
    'INITIAL_LR': 1e-3,
    'OPTIMIZER_TYPE': 'Adam',
    'USE_MIXED_PRECISION': True,
    'MAX_EPOCHS': 50,
    'EARLY_STOP_PATIENCE': 10
}
```

#### Full ImageNet Configuration
```python
'imagenet': {
    'DATASET_PATH': '/data/imagenet/full_dataset',
    'NUM_CLASSES': 1000,
    'IMAGE_SIZE': 224,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 4,
    'INITIAL_LR': 1.20e-05,
    'OPTIMIZER_TYPE': 'SGD',
    'USE_MIXED_PRECISION': True,
    'MAX_EPOCHS': 100,
    'EARLY_STOP_PATIENCE': 15
}
```

## Automatic Behavior

### When Statistics Are Automatically Loaded
- Training starts without `--compute-stats` flag
- Stored statistics exist in cache
- No manual override specified

### When Learning Rate Is Automatically Loaded  
- Training starts without `--lr-finder` flag
- Stored optimal learning rate exists in cache
- No manual override specified

### When Values Are Automatically Stored
- **Statistics**: Automatically stored when `--compute-stats` is used
- **Learning Rate**: Automatically stored when `--lr-finder` is used

## Cache Management

### Cache Location
- Statistics and learning rates are stored in `config_cache.json`
- Located in the same directory as `config_switcher.py`

### Cache Structure
```json
{
    "dataset_stats": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "timestamp": "2024-01-15T10:30:00"
    },
    "optimal_lr": {
        "value": 1.2e-05,
        "timestamp": "2024-01-15T11:00:00"
    }
}
```

## Best Practices

### 1. Initial Setup
1. Choose appropriate configuration for your dataset
2. Compute dataset statistics once
3. Find optimal learning rate once
4. Use stored values for subsequent training runs

### 2. Experimentation
- Use `--lr-finder` to experiment with different learning rates
- Use `--compute-stats` if your dataset changes
- Clear cache when switching between significantly different datasets

### 3. Production Training
- Rely on stored values for consistent training runs
- Only recompute when necessary (dataset changes, hardware changes)

### 4. Debugging
- Use `python config_switcher.py show` to verify current configuration
- Use `python config_switcher.py show-cache` to check stored values
- Clear cache if experiencing unexpected behavior

## Migration from Old System

### If You Were Using Manual Configuration
1. Run `python config_switcher.py show` to see current settings
2. Choose appropriate preset: `python config_switcher.py imagenet`
3. Your existing training commands will continue to work

### If You Have Existing Statistics/LR Values
1. Store them manually using `store-stats` and `store-lr` commands
2. Or recompute them using `--compute-stats` and `--lr-finder`

## Troubleshooting

### Common Issues

#### "Could not load stored statistics"
- Cache file doesn't exist or is corrupted
- Run `python train_resnet50.py --compute-stats` to regenerate

#### "Could not load stored learning rate"  
- No optimal LR stored in cache
- Run `python train_resnet50.py --lr-finder` to find and store

#### Training uses wrong parameters
- Check current configuration: `python config_switcher.py show`
- Verify cache contents: `python config_switcher.py show-cache`
- Clear cache if needed: `python config_switcher.py clear-cache`

#### Configuration not updating
- Ensure you're running config_switcher.py from the correct directory
- Check file permissions on train_resnet50.py and config_cache.json

### Getting Help

```bash
# Show help for config_switcher
python config_switcher.py --help

# Show help for training script
python train_resnet50.py --help
```

## Advanced Usage

### Custom Configurations
You can add custom configurations by editing the `CONFIGS` dictionary in `config_switcher.py`:

```python
CONFIGS = {
    'tiny-imagenet': { ... },
    'imagenet': { ... },
    'custom-dataset': {
        'DATASET_PATH': '/path/to/custom/dataset',
        'NUM_CLASSES': 500,
        'IMAGE_SIZE': 128,
        # ... other parameters
    }
}
```

### Programmatic Access
```python
from config_switcher import ConfigManager

config_manager = ConfigManager()

# Load stored statistics
stats = config_manager.load_stored_stats()
if stats:
    mean, std = stats['mean'], stats['std']

# Store new learning rate
config_manager.store_lr(1.5e-05)
```

This configuration system provides a robust, automated approach to managing training parameters while maintaining full flexibility for experimentation and manual overrides.