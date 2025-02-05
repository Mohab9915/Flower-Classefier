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
  - Dense layer (512 units, softmax activation)
  - Dropout layer (0.2)
  - Output layer (102 units, softmax activation)

## Dataset

The Oxford Flowers 102 dataset contains images of 102 flower categories. Each class consists of between 40 and 258 images.