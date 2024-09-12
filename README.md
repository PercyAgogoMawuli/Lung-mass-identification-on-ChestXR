# Medical AI - Image Classification for Detecting Masses in ChestXR8
![SAMPLE_XR](https://github.com/user-attachments/assets/5e764eb4-e65a-440d-9d38-314c7814668d)


This repository contains a project focused on detecting a specific finding, such as a mass in medical images, using a pre-trained deep learning model. The project leverages the **InceptionV3** architecture, with transfer learning, to classify medical images as positive or negative for a given medical condition (e.g., a mass).

![ACCUR](https://github.com/user-attachments/assets/70edac26-9330-49e5-8730-b857fb5ddd40) 
![LOSS](https://github.com/user-attachments/assets/1248b9f3-a25d-419b-bc7e-e2382892e968)

![roc](https://github.com/user-attachments/assets/2f99896e-c76a-422d-979e-0b76b80485fb)

![conf_cut_off](https://github.com/user-attachments/assets/7163a7c8-7810-4c90-8efa-eabdd5ea74b0)

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results Visualization](#results-visualization)


## Overview

This project uses medical images to train a binary classification model. The model predicts whether an image contains a particular medical finding (e.g., a **mass**). The key steps in this pipeline include:

- **Preprocessing**: Organizing and splitting image data into training and test sets.
- **Training**: Fine-tuning a pre-trained **InceptionV3** model on the dataset.
- **Evaluation**: Visualizing the model's performance using metrics like accuracy, loss, and AUC (Area Under the Curve) from the Receiver Operating Characteristic (ROC) curve.

## Data Preprocessing

The dataset is assumed to consist of medical images, with associated labels stored in a CSV file. The images are categorized into **positive** (containing the finding) and **negative** (no finding) classes. Key preprocessing steps include:

- **Splitting the data**: The dataset is split into training (80%) and testing (20%) sets for both positive and negative samples.
- **Organizing the dataset**: Image files are moved to appropriate directories (train/test, positive/negative).
- **Resizing images**: Images are resized to 256x256 pixels to match the input size expected by the model.
  
## Model Training

### Pre-trained Model

The **InceptionV3** model, pre-trained on ImageNet, is used for transfer learning. The top layers are removed, and new dense layers are added to adapt the model for binary classification.


pre_trained_model = InceptionV3(input_shape=(256, 256, 3), weights='imagenet', include_top=False)

# Adding new layers
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Data Augmentation
Data augmentation techniques are applied to the training images to improve model generalization. These include:

Rotation
Shifts in width and height
Zooming
Shearing

# Training Process
The model is trained using the augmented data for 20 epochs. Performance is evaluated on the test set.

## Evaluation
The model's performance is assessed using metrics such as accuracy, loss, and the ROC curve. Plots are generated to visualize the following:

Training vs. Validation Accuracy and Loss.
Confidence histograms of predictions for positive and negative classes.
ROC Curve to evaluate model performance in terms of sensitivity and specificity.

## Installation
To get started, clone the repository and install the necessary dependencies:
git clone https://github.com/adleberg/medical-ai
cd medical-ai
pip install -r requirements.txt

# Requirements
tensorflow
keras
scikit-learn
pandas
numpy
matplotlib
Pillow

## Usage
Predicting New Images
You can use the predict_image function to classify new images. The model expects images to be resized to 256x256 pixels.

## Results Visualization
After training, the following visualizations are available:

Example Images: Positive and negative examples from the dataset are displayed.
Accuracy and Loss Plots: Training and validation accuracy/loss over epochs.
ROC Curve: Visualization of the model's true positive rate (sensitivity) against the false positive rate (1-specificity).



**THANK YOU FOR VISITING MY PROJECT**
