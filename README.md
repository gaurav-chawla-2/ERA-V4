# MNIST Model with <25,000 Parameters

This project implements a CNN model for the MNIST dataset with the following constraints:
- Fewer than 25,000 parameters
- Achieves at least 95% test accuracy in just 1 epoch

## Setup

### Virtual Environment

It's recommended to use a virtual environment to keep dependencies isolated:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

Alternatively, you can install the dependencies directly:

```bash
pip3 install -r requirements.txt
```

## Model Architecture

The model uses a lightweight CNN architecture with the following key features:
- Multiple small convolutional layers with batch normalization
- Dropout for regularization
- Global Average Pooling to reduce parameters
- Efficient parameter usage with appropriate filter sizes

## Usage

To train and test the model:

```bash
python3 train.py
```

## Model Performance

The model achieves the target accuracy of 95% or higher on the MNIST test set in just 1 epoch, while using fewer than 25,000 parameters.

## Architecture Details

- Input: 28x28x1 MNIST images
- Uses 5 convolutional layers with batch normalization and dropout
- 2 max pooling layers to reduce spatial dimensions
- Global Average Pooling instead of fully connected layers
- Output: 10 classes (digits 0-9)

## MNIST Classification with Lightweight CNN
## Project Overview
This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification. The model is designed to achieve high accuracy (≥95%) in a single epoch while keeping the parameter count under 25,000.

## Model Architecture
The model uses a carefully designed CNN architecture with the following key components:

1. Convolutional Layers :
   
   - 5 convolutional layers with strategic padding and channel sizes
   - Pointwise (1x1) convolutions to reduce parameters
   - Residual connection to improve gradient flow

2. Regularization Techniques :
   
   - Batch Normalization after each convolutional layer
   - Dropout (3%) to prevent overfitting
   - Weight decay (2e-4) in the optimizer

3. Pooling Layers :
   
   - Two MaxPool2d layers to reduce spatial dimensions
   - Global Average Pooling at the end to reduce parameters

```
   Net(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # 28x28x16
  (bn1): BatchNorm2d(16)
  (dropout1): Dropout(p=0.03)
  
  (conv2): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # 28x28x24
  (bn2): BatchNorm2d(24)
  (dropout2): Dropout(p=0.03)
  
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0)  # 14x14x24
  
  (conv_res): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1))  # 14x14x24 (for residual)
  
  (conv3): Conv2d(24, 16, kernel_size=(1, 1), stride=(1, 1))  # 14x14x16
  (bn3): BatchNorm2d(16)
  
  (conv4): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # 14x14x24
  (bn4): BatchNorm2d(24)
  (dropout3): Dropout(p=0.03)
  
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0)  # 7x7x24
  
  (conv5): Conv2d(24, 10, kernel_size=(1, 1), stride=(1, 1))  # 7x7x10
  (gap): AdaptiveAvgPool2d(output_size=1)  # 1x1x10
)
```

Total Parameters : 8,530 (well under the 25,000 limit)

## Training Approach
### Hyperparameters
- Batch Size : 32 (smaller batch size for better generalization)
- Learning Rate : 0.1 (relatively high for faster convergence)
- Optimizer : SGD with momentum (0.9) and weight decay (2e-4)
- Scheduler : OneCycleLR with cosine annealing
- Epochs : 1 (as per requirement)
### Data Augmentation
To improve model generalization with limited training:

- Random rotation (±15°)
- Random affine transformations (translation and scaling)
- Normalization with MNIST dataset statistics
## Key Design Decisions
1. Residual Connection : Improves gradient flow and helps with faster training
2. Low Dropout Rate (3%) : Provides regularization without excessive information loss
3. 1x1 Convolutions : Reduces parameters while maintaining expressiveness
4. OneCycleLR Scheduler : Enables higher learning rates with controlled convergence
5. Batch Normalization : Stabilizes training and allows for higher learning rates
## Performance
The model achieves high accuracy on the MNIST dataset in a single epoch:

- Parameter Count : 8,530 (well under the 25,000 limit)
- Test Accuracy : ~95% (meets the target requirement)

## Project Structure
```
├── model.py       # Model architecture definition
├── train.py       # Training script
├── requirements.txt  # Required packages
├── setup.sh       # Setup script
└── mnist_model.pth   # Saved model weights
```