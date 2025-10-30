# ResNet50 ImageNet Training

A comprehensive implementation for training ResNet50 on both Tiny-ImageNet-200 and full ImageNet datasets with checkpoint support and automatic configuration switching.

## 🎯 Features

- **Clear ResNet50 Implementation**: Every layer operation (Conv, BN, MaxPooling, ReLU) is explicitly visible
- **Dual Dataset Support**: Works with both Tiny-ImageNet-200 and full ImageNet
- **Checkpoint System**: Automatic saving and resuming of training progress
- **Configuration Switching**: Easy switching between dataset configurations
- **Comprehensive Logging**: Training progress, validation graphs, and performance metrics
- **Early Feedback**: Progress monitoring and early stopping
- **Modular Design**: Easy parameter modification without code changes

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Your Dataset

**Option A: Tiny-ImageNet-200 (Recommended for testing)**
```bash
# Switch to Tiny-ImageNet configuration
python config_switcher.py tiny-imagenet

# Download dataset
sudo bash setup_tiny_imagenet.sh
```

**Option B: Full ImageNet (Production training)**
```bash
# Switch to ImageNet configuration
python config_switcher.py imagenet

# Get download instructions
python download_imagenet.py
```

### 3. Verify Setup

```bash
python verify_setup.py
```

### 4. Start Training

```bash
python train_resnet50.py
```

## 📊 Dataset Information

| Dataset | Classes | Image Size | Training Samples | Target Accuracy |
|---------|---------|------------|------------------|-----------------|
| Tiny-ImageNet-200 | 200 | 64×64 | ~100,000 | 80% |
| Full ImageNet | 1000 | 224×224 | ~1.2M | 75% |

## 💾 Checkpoint Management

The training script automatically saves checkpoints for resuming interrupted training:

### Automatic Checkpointing
- Saves checkpoint every 5 epochs
- Saves best model checkpoint when validation accuracy improves
- Automatically resumes from latest checkpoint on restart

### Manual Checkpoint Control
```bash
# Resume from specific checkpoint
RESUME_FROM_CHECKPOINT="./checkpoints/checkpoint_epoch_25.pth" python train_resnet50.py

# Start fresh training (ignore existing checkpoints)
rm -rf ./checkpoints && python train_resnet50.py
```

### Checkpoint Contents
- Model state (weights and biases)
- Optimizer state (momentum, learning rate schedule)
- Training history (losses, accuracies)
- Configuration parameters

## ⚙️ Configuration Management

### Quick Configuration Switching
```bash
# Switch to Tiny-ImageNet configuration
python config_switcher.py tiny-imagenet

# Switch to full ImageNet configuration  
python config_switcher.py imagenet

# Show current configuration
python config_switcher.py show
```

### Manual Configuration
Edit the configuration section in `train_resnet50.py`:
```python
# Dataset Configuration
DATASET_PATH = "/opt/dlami/nvme/imagenet"  # or tiny-imagenet-200
NUM_CLASSES = 1000  # or 200 for Tiny-ImageNet
IMAGE_SIZE = 224    # or 64 for Tiny-ImageNet
BATCH_SIZE = 64     # Adjust based on GPU memory

# Training Configuration
MAX_EPOCHS = 100
TARGET_ACCURACY = 75.0
OPTIMIZER_TYPE = 'SGD'  # or 'AdamW'
INITIAL_LR = 0.1
```

## 📊 What You'll Get

### Comprehensive Training Monitoring
- Real-time training progress with loss and accuracy
- Validation accuracy tracking
- Learning rate scheduling visualization
- Early stopping when target reached
- Progress warnings if training stalls

### Automatic Checkpointing
- Resume training from interruptions
- Best model preservation
- Training history preservation
- Easy experiment management

### Rich Visualizations
- Training/validation loss curves
- Accuracy progression graphs
- Learning rate schedule
- Overfitting analysis

### Detailed Results
- Final performance metrics
- Training recommendations
- JSON export of all metrics
- Best model checkpoint

## ⚙️ Configuration

All parameters are configurable at the top of `train_resnet50.py`:

```python
# Dataset Configuration
DATASET_PATH = "./data/mini-imagenet"
NUM_CLASSES = 100
BATCH_SIZE = 64

# Training Configuration
MAX_EPOCHS = 50
TARGET_ACCURACY = 80.0
EARLY_STOP_PATIENCE = 10

# Model Configuration
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# ATOM Optimizer Configuration
ATOM_LR_MIN = 1e-6
ATOM_LR_MAX = 1e-1
ATOM_STEPS = 100
```

## 🏗️ Architecture Details

### ResNet50 Structure
- **Initial Block**: 7×7 Conv → BN → ReLU → MaxPool
- **Stage 1**: 3 bottleneck blocks (64→256 channels)
- **Stage 2**: 4 bottleneck blocks (256→512 channels)
- **Stage 3**: 6 bottleneck blocks (512→1024 channels)
- **Stage 4**: 3 bottleneck blocks (1024→2048 channels)
- **Final**: Global AvgPool → Dropout → Linear

### Bottleneck Block
Each block contains:
1. **1×1 Conv**: Dimension reduction
2. **3×3 Conv**: Main computation
3. **1×1 Conv**: Dimension expansion
4. **Shortcut**: Identity or projection
5. **Residual Addition**: Skip connection

## 📈 Training Process

1. **Dataset Loading**: Automatic data augmentation and normalization
2. **LR Finding**: ATOM optimizer scans for optimal learning rate
3. **Training Loop**: 
   - Forward pass through ResNet50
   - Loss computation with label smoothing
   - ATOM optimizer step
   - Cosine annealing LR schedule
4. **Validation**: Regular accuracy evaluation
5. **Early Stopping**: When target accuracy reached or no improvement
6. **Results Export**: Comprehensive analysis and visualizations

## 📁 Output Files

After training, you'll find in `./results/`:

- `best_model.pth`: Best model checkpoint
- `training_progress.png`: Comprehensive training graphs
- `lr_finder.png`: Learning rate analysis
- `training_results.json`: Detailed metrics and configuration

## 🎯 Expected Performance

- **Target Accuracy**: ~80% on Mini-ImageNet
- **Training Time**: 30-60 minutes (depending on hardware)
- **Early Feedback**: Progress assessment within 6-10 epochs
- **Automatic Optimization**: No manual LR tuning required

## 🔧 Troubleshooting

### Dataset Issues
```bash
# If dataset not found, run setup again
python setup_dataset.py
```

### Memory Issues
```python
# Reduce batch size in configuration
BATCH_SIZE = 32  # or 16 for limited GPU memory
```

### Slow Training
```python
# Reduce image size or use fewer workers
IMAGE_SIZE = 128  # instead of 224
NUM_WORKERS = 2   # instead of 4
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory
- Mini-ImageNet dataset

## 🎉 Success Indicators

- ✅ Optimal LR found automatically
- ✅ Training loss decreasing steadily
- ✅ Validation accuracy improving
- ✅ Target accuracy reached
- ✅ Comprehensive analysis generated

The implementation provides complete transparency into the training process while maintaining simplicity and effectiveness!