# Flower Classifier

A deep learning model that classifies 102 different types of flowers using MobileNetV2 architecture.

## Overview

This project uses transfer learning with MobileNetV2 to classify flowers from the Oxford Flowers 102 dataset. The model achieves classification by utilizing a pre-trained model and adding custom classification layers.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install tensorflow tensorflow_hub tensorflow_datasets pillow numpy
```

## Project Structure

- `app.py`: Main application for making predictions
- `initiate_model.py`: Script for training the model
- `predict.py`: Contains prediction functionality
- `process_image.py`: Image preprocessing utilities
- `normalize_and_resize.py`: Image normalization functions
- `load_class.py`: Utility to load class names
- `label_map.json`: Mapping of class indices to flower names

## Model Architecture

- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Additional layers:
  - Dense layer (512 units, ReLU activation with L2 regularization)
  - Dropout layer (0.3)
  - Output layer (102 units, softmax activation)

### Training Parameters
- Optimizer: Adam (learning rate: 0.0001)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Epochs: 30
- Regularization: L2 (0.001)

## Usage

### Training the Model

To train the model:
```bash
python initiate_model.py
```

### Making Predictions

To classify a flower image:
```bash
python app.py path/to/image.jpg path/to/model.h5 --top_k 5 --category_names label_map.json
```

Arguments:
- `image_path`: Path to the flower image
- `model_path`: Path to the trained model (.h5 file)
- `--top_k`: Number of top predictions to show (default: 5)
- `--category_names`: Path to JSON file mapping labels to flower names

## Model Architecture

- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Additional layers:
  - Dense layer (512 units, ReLU activation with L2 regularization)
  - Dropout layer (0.3)
  - Output layer (102 units, softmax activation)

### Training Parameters
- Optimizer: Adam (learning rate: 0.0001)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Epochs: 30
- Regularization: L2 (0.001)


## Model Architecture

- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Additional layers:
  - Dense layer (512 units, ReLU activation with L2 regularization)
  - Dropout layer (0.3)
  - Output layer (102 units, softmax activation)

### Training Parameters
- Optimizer: Adam (learning rate: 0.0001)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Epochs: 30
- Regularization: L2 (0.001)

## Model Performance

### Final Metrics
- Training Accuracy: 99.12%
- Validation Accuracy: 78.04%
- Training Loss: 0.8901
- Validation Loss: 1.6045

### Training Progress Highlights
- Early Stage (Epoch 1-5):
  - Started at 1.47% accuracy
  - Reached 42.55% accuracy by epoch 5
  
- Mid Stage (Epoch 10-15):
  - Achieved 77.06% training accuracy at epoch 10
  - Validation accuracy reached 73.04% by epoch 14
  
- Late Stage (Epoch 25-30):
  - Training accuracy exceeded 97% consistently
  - Best validation accuracy: 78.82% (epoch 29)
  - Final validation accuracy: 78.04%

### Learning Curve
Training showed consistent improvement:
- 50% accuracy threshold: Epoch 6
- 70% accuracy threshold: Epoch 11
- 90% accuracy threshold: Epoch 16
- 95% accuracy threshold: Epoch 19

### Model Convergence
- Training loss decreased from 5.4861 to 0.8901
- Validation loss improved from 5.1444 to 1.6045
- Model showed good generalization with acceptable gap between training and validation metrics

## Dataset

The Oxford Flowers 102 dataset contains images of 102 flower categories. Each class consists of between 40 and 258 images.