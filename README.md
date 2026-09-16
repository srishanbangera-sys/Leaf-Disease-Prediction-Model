# Leaf Disease Detection Model

A deep learning project for detecting plant leaf diseases using **Convolutional Neural Networks (CNNs)** and **TensorFlow/Keras**.

The model is trained on the **PlantVillage dataset** to classify leaf images into their respective disease categories.

## Project Overview

Plant diseases can significantly affect crop production and quality. This project uses image classification with deep learning to identify diseases from images of plant leaves.

The workflow includes:

* Loading and preprocessing leaf images
* Creating training and validation datasets
* Training a CNN model
* Evaluating model performance
* Predicting the disease category from leaf images

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* CNN (Convolutional Neural Network)
* PlantVillage Dataset

## Dataset

The model uses the **PlantVillage dataset**, which contains images of healthy and diseased plant leaves.

Dataset directory used in the project:

```text
dataset/
└── Plantvillage/
    ├── Class 1/
    ├── Class 2/
    ├── Class 3/
    └── ...
```

Each folder represents a different plant disease class.

## Model Architecture

The CNN consists of multiple layers designed to extract visual features from leaf images.

```text
Input Image
     ↓
Rescaling
     ↓
Conv2D
     ↓
MaxPooling2D
     ↓
Conv2D
     ↓
MaxPooling2D
     ↓
Conv2D
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense
     ↓
Dropout
     ↓
Output Layer
```

## Image Configuration

The images are resized before being provided to the model.

```python
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
```

The dataset is divided into training and validation data using an **80/20 split**.

## How It Works

1. The PlantVillage dataset is loaded using TensorFlow.
2. Images are resized to `128 × 128` pixels.
3. Pixel values are normalized using a rescaling layer.
4. The CNN extracts important visual features from the leaves.
5. The model learns patterns associated with different diseases.
6. The trained model predicts the disease class for a new leaf image.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/leaf-disease-detection.git
cd leaf-disease-detection
```

Install the required dependencies:

```bash
pip install tensorflow numpy
```

## Running the Project

Make sure the dataset is placed inside:

```text
dataset/Plantvillage
```

Then run:

```bash
python train.py
```

The model will train using the PlantVillage dataset and display the training and validation performance.

## Example Prediction

After training, the model can be used to classify a new leaf image.

```text
Input:
Leaf Image

        ↓

CNN Model

        ↓

Predicted Class:
Plant Disease Category
```

## Learning Objectives

This project was built to understand practical concepts in:

* Image classification
* Convolutional Neural Networks
* TensorFlow and Keras
* Dataset preprocessing
* Training and validation
* Deep learning model development
* Plant disease detection using computer vision

## Future Improvements

* Add a prediction web application
* Improve model accuracy using data augmentation
* Experiment with transfer learning models such as MobileNet or EfficientNet
* Add confidence scores for predictions
* Deploy the model as a web or mobile application

## Disclaimer

This project is intended for **educational and experimental purposes**. Model predictions should not be treated as professional agricultural diagnosis.

## Author

**Srishan Bangera**

B.Tech CSE Student | AI/ML Engineer in Training

---

Built as part of a hands-on learning journey in **Machine Learning, Deep Learning, and Computer Vision**.
