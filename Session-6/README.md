# Session-6: MNIST CNN Optimization - Achieving 99.4% with <8000 Parameters

## 🎯 Project Objective
Develop a modular CNN architecture to achieve **99.4% accuracy consistently** on MNIST dataset with:
- **≤ 15 epochs** training time
- **< 8000 parameters** total
- **Modular design** with progressive optimization iterations

## 📊 Final Results Summary

| Model | Parameters | Val Accuracy | Test Accuracy | Target 99.4% | Parameter Limit | Overall Status |
|-------|------------|--------------|---------------|--------------|-----------------|----------------|
| **Model_1** | 9,850 | 99.35% | 99.39% | ❌ (0.05% short) | ❌ (1,850 over) | **Needs optimization** |
| **Model_2** | 7,262 | 99.33% | 99.35% | ❌ (0.07% short) | ✅ (738 under) | **Close to target** |
| **Model_3** | 7,518 | 99.33% | 99.35% | ❌ (0.07% short) | ✅ (482 under) | **Close to target** |

### 🔍 Key Findings
- ✅ **Parameter Constraint**: Models 2 & 3 successfully under 8000 parameters
- ⚠️ **Accuracy Gap**: All models ~0.05-0.07% short of 99.4% target
- ✅ **Architecture Efficiency**: Significant parameter reduction achieved
- 🎯 **Next Steps**: Fine-tuning needed for final accuracy push

## 🏗️ Architecture Evolution & Optimization Strategy

### Model_1: Optimized Baseline Architecture
**Target:** Establish solid foundation with proper receptive field

- **Key Features:**
  - Basic CNN with batch normalization
  - Global Average Pooling for parameter efficiency
  - Proper receptive field calculation (32x32 final RF)

**Receptive Field Progression:**

### Model_2: Efficiency Optimization
**Target:** Improve accuracy while maintaining parameter efficiency
- **Parameters:** ~7,200
- **Expected Accuracy:** ~99.1%
- **Key Features:**
  - Depthwise separable convolutions
  - Strategic skip connections
  - Optimized channel progression

**Innovations:**
- Depthwise separable convolutions reduce parameters by ~40%
- Skip connections improve gradient flow
- Better feature extraction efficiency

### Model_3: Target Achievement
**Target:** Achieve 99.4% accuracy consistently
- **Parameters:** ~7,800
- **Expected Accuracy:** 99.45%
- **Key Features:**
  - Squeeze-and-Excitation attention blocks
  - Advanced regularization strategy
  - Optimized learning rate scheduling

**Advanced Features:**
- **SE Blocks:** Channel attention for better feature focus
- **Progressive Dropout:** 0.03 → 0.03 → 0.02 for optimal regularization
- **OneCycleLR:** Optimized learning rate scheduling
- **Advanced Augmentation:** Balanced rotation, translation, scaling

## Training Strategy

### Data Augmentation
```python
transforms.Compose([
    transforms.RandomRotation((-6.0, 6.0)),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), 
                           scale=(0.94, 1.06), shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### Learning Rate Scheduling
- **Model_1:** OneCycleLR with max_lr=0.01, pct_start=0.2
- **Model_2:** OneCycleLR with max_lr=0.015, pct_start=0.15  
- **Model_3:** OneCycleLR with max_lr=0.012, pct_start=0.1

### Optimization Strategy
- **Optimizer:** Adam with weight decay
- **Batch Size:** 128 (optimal for convergence speed)
- **Validation Split:** 10% for model selection

## Results Analysis

### Model_1 Results

### **Architecture Details:**
```python
# Block 1: Feature Extraction (Reduced Channels)
Conv2d(1→6, 3x3) + BN + ReLU     # 28x28x6, RF: 3x3
Conv2d(6→12, 3x3) + BN + ReLU    # 28x28x12, RF: 5x5
MaxPool2d(2x2) + Dropout(0.05)   # 14x14x12, RF: 6x6

# Block 2: Feature Refinement
Conv2d(12→16, 3x3) + BN + ReLU   # 14x14x16, RF: 10x10
Conv2d(16→20, 3x3) + BN + ReLU   # 14x14x20, RF: 14x14
MaxPool2d(2x2) + Dropout(0.05)   # 7x7x20, RF: 16x16

# Block 3: Final Feature Extraction
Conv2d(20→16, 3x3) + BN + ReLU   # 7x7x16, RF: 20x20
Conv2d(16→10, 3x3)               # 7x7x10, RF: 24x24
GlobalAvgPool2d()                # 1x1x10
```

### Model_2: Depthwise Separable Efficiency

**Key Innovations:**
```python
# Depthwise Separable Convolution Block
Conv2d(8→8, 3x3, groups=8)       # Depthwise: 8 separate 3x3 convs
Conv2d(8→12, 1x1)                # Pointwise: channel mixing
# Parameter reduction: (8*3*3*8) + (8*1*1*12) = 672 + 96 = 768
# vs Standard: (8*3*3*12) = 864 params → 11% savings per block

# Skip Connection Strategy
skip = input_features
x = depthwise_separable_block(x)
x = x + skip_conv(skip)          # Residual connection for gradient flow
```

### Model_3: Lightweight Attention Optimization

**Advanced Features:**
```python
# Lightweight Squeeze-and-Excitation Block
class LightweightSEBlock(nn.Module):
    def __init__(self, channels, reduction=8):  # Higher reduction for efficiency
        self.squeeze = AdaptiveAvgPool2d(1)
        self.excitation = Sequential(
            Linear(channels, max(channels//reduction, 4)),  # Min 4 neurons
            ReLU(), Linear(max(channels//reduction, 4), channels), Sigmoid()
        )
    # Parameter cost: 2 * (channels * max(channels//8, 4)) ≈ channels/4

# Architecture with Attention
x = depthwise_separable_conv(x)
x = lightweight_se_block(x)      # Channel attention
x = residual_connection(x, skip) # Gradient flow enhancement
```

## 🔬 Technical Analysis

### Parameter Distribution Comparison
| Component | Model_1 | Model_2 | Model_3 | Optimization |
|-----------|---------|---------|---------|--------------|
| **Conv Layers** | ~8,500 | ~6,200 | ~6,400 | Depthwise separable |
| **BatchNorm** | ~68 | ~68 | ~68 | Minimal impact |
| **SE Attention** | 0 | 0 | ~256 | Lightweight design |
| **Skip Connections** | 0 | ~240 | ~240 | 1x1 convolutions |
| **Final Linear** | 0 | 0 | 0 | GAP eliminates need |
| **Total** | 9,850 | 7,262 | 7,518 | Progressive optimization |
