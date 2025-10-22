# ResNet50 Mini-ImageNet Training

A clean, single-file implementation for training ResNet50 on the Mini-ImageNet dataset with automatic learning rate optimization using ATOM optimizer.

## 🎯 Features

- **Clear ResNet50 Implementation**: Every layer operation (Conv, BN, MaxPooling, ReLU) is explicitly visible
- **ATOM Optimizer**: Automatically finds optimal learning rates during training
- **Comprehensive Logging**: Training progress, validation graphs, and performance metrics
- **Early Feedback**: Progress monitoring within 6-10 epochs
- **Modular Design**: Easy parameter modification without code changes
- **Target**: ~80% accuracy on Mini-ImageNet

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Dataset

```bash
python setup_dataset.py
```

Choose from:
- **Option 1**: Automatic download using Kaggle API (recommended)
- **Option 2**: Create sample dataset for testing
- **Option 3**: Manual download instructions

### 3. Start Training

```bash
python train_resnet50.py
```

## 📊 What You'll Get

### Automatic Learning Rate Finding
- ATOM optimizer scans learning rates from 1e-6 to 1e-1
- Finds optimal LR automatically
- Saves LR finder plot for analysis

### Comprehensive Training Monitoring
- Real-time training progress
- Validation accuracy tracking
- Early stopping when target reached
- Progress warnings if training stalls

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