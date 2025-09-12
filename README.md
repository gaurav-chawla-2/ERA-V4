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