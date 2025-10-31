# ResNet50 Training Guide

## Enhanced Features

The training script now includes two powerful features:

### 1. Learning Rate Range Test (LR Finder)
Find the optimal learning rate for your model and dataset.

### 2. Dataset Statistics Computation
Compute mean and standard deviation of your dataset for proper normalization.

### 3. Enhanced Checkpoint Resume
Automatically resume training from the last checkpoint with proper state restoration.

## Usage Examples

### Basic Training
```bash
python3 train_resnet50.py
```

### Compute Dataset Statistics Only
```bash
python3 train_resnet50.py --stats-only
```

### Run LR Finder Only
```bash
python3 train_resnet50.py --lr-finder-only
```

### Full Workflow (Recommended)
```bash
# Step 1: Compute dataset statistics
python3 train_resnet50.py --compute-stats --stats-only

# Step 2: Find optimal learning rate
python3 train_resnet50.py --lr-finder-only

# Step 3: Start training (update LR in config if needed)
python3 train_resnet50.py

# Step 4: Resume training if interrupted
python3 train_resnet50.py --resume
```

### Combined Operations
```bash
# Compute stats and run LR finder before training
python3 train_resnet50.py --compute-stats --lr-finder

# Resume with stats and LR finder
python3 train_resnet50.py --resume --compute-stats --lr-finder
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--lr-finder` | Run LR range test before training |
| `--compute-stats` | Compute dataset statistics before training |
| `--resume` | Resume from checkpoint if available |
| `--lr-finder-only` | Only run LR finder and exit |
| `--stats-only` | Only compute stats and exit |

## Output Files

The script generates several useful files in the save directory:

- `lr_finder.png` - LR finder plot
- `lr_finder_results.txt` - LR finder recommendations
- `dataset_stats.txt` - Dataset statistics
- `training_progress.png` - Training progress plots
- `checkpoint.pth` - Latest checkpoint
- `best_model.pth` - Best model weights

## Configuration

Before training, make sure to:

1. **Set the correct dataset configuration:**
   ```bash
   python3 config_switcher.py imagenet  # or tiny-imagenet
   ```

2. **Verify dataset path in config_switcher.py:**
   - ImageNet: `/data/imagenet`
   - Tiny ImageNet: `/data/tiny-imagenet-200`

3. **Check available storage:**
   ```bash
   df -h /data
   ```

## Recommended Workflow for New Datasets

1. **Download Dataset:**
   ```bash
   # For ImageNet
   python3 download_imagenet.py
   
   # For Tiny ImageNet (faster for testing)
   python3 download_tiny_imagenet.py
   ```

2. **Set Configuration:**
   ```bash
   python3 config_switcher.py imagenet
   ```

3. **Compute Dataset Statistics:**
   ```bash
   python3 train_resnet50.py --stats-only
   ```

4. **Find Optimal Learning Rate:**
   ```bash
   python3 train_resnet50.py --lr-finder-only
   ```

5. **Update Learning Rate (if needed):**
   - Check `lr_finder_results.txt`
   - Update `LEARNING_RATE` in `config_switcher.py` if recommended

6. **Start Training:**
   ```bash
   python3 train_resnet50.py
   ```

7. **Resume if Interrupted:**
   ```bash
   python3 train_resnet50.py --resume
   ```

## Tips

- **LR Finder:** Look for the steepest descent in the loss curve. The script automatically suggests an optimal LR.
- **Dataset Stats:** Use computed statistics to update normalization values if needed.
- **Checkpoints:** Always use `--resume` to continue interrupted training sessions.
- **Memory:** If you encounter OOM errors, reduce `BATCH_SIZE` in the configuration.
- **Storage:** Ensure sufficient space in `/data` before downloading large datasets.

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size in configuration
   - Enable mixed precision (already enabled by default)

2. **Dataset Not Found:**
   - Verify dataset path in configuration
   - Ensure dataset is properly downloaded and extracted

3. **Permission Denied:**
   - Check write permissions for save directory
   - Ensure `/data` mount is accessible

4. **Checkpoint Loading Errors:**
   - Delete corrupted checkpoint files
   - Start fresh training without `--resume`

### Performance Optimization

- **T4 GPU:** Current settings are optimized for T4 (batch size 96, mixed precision)
- **Other GPUs:** Adjust batch size based on GPU memory
- **CPU Training:** Disable mixed precision and reduce batch size significantly