# Training Visualization Guide

## Comprehensive Plotting Features

The training script now generates **7 different types of plots** to provide complete visibility into your training process:

### 📊 **Generated Plots**

#### 1. **Main Training Progress** (`training_progress.png`)
- **4-panel overview** with all key metrics
- Training & Validation Loss
- Training & Validation Accuracy (with target line)
- Learning Rate Schedule
- Overfitting Analysis

#### 2. **Detailed Loss Curves** (`loss_curves.png`)
- High-resolution loss visualization
- Training vs Validation loss over time
- Markers for easy epoch identification

#### 3. **Detailed Accuracy Curves** (`accuracy_curves.png`)
- High-resolution accuracy visualization
- Training vs Validation accuracy over time
- Target accuracy line for reference

#### 4. **Learning Rate Schedule** (`learning_rate_schedule.png`)
- Detailed LR progression over epochs
- Log scale for better visualization

#### 5. **Overfitting Analysis** (`overfitting_analysis.png`)
- Gap between training and validation accuracy
- Warning thresholds at 5% and 10%
- Helps identify overfitting issues

#### 6. **Side-by-Side Comparison** (`loss_and_accuracy.png`)
- Loss and accuracy in one view
- Perfect for quick assessment

#### 7. **Training Summary** (`training_summary.png`)
- Performance metrics summary
- Best validation accuracy
- Final training/validation accuracy
- Visual performance bars

#### 8. **LR Finder Plot** (`lr_finder.png`)
- Learning rate range test results
- Optimal LR suggestion
- Generated when using `--lr-finder` flag

### ⚙️ **Plot Generation Timing**

Plots are automatically generated:
- **Every 5 epochs** (configurable via `PLOT_FREQUENCY`)
- **When a new best model is achieved**
- **At epoch 1** (to see initial progress)
- **At the end of training** (final comprehensive plots)

### 🎛️ **Configuration**

You can control plotting behavior by modifying these parameters in the training script:

```python
PLOT_FREQUENCY = 5        # Generate plots every N epochs
PLOT_STYLE = 'seaborn-v0_8'  # Matplotlib style
SAVE_DIR = "./results"    # Where plots are saved
```

### 📁 **Output Location**

All plots are saved to the `./results/` directory: