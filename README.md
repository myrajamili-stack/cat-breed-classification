# Cat Breed Images Classification with Convolutional Neural Networks (CNN)

## Description

This repository contains a cat breed image classification project using convolutional neural network (CNN) models. It is part of the project for the Principles of Artificial Intelligence subject, focusing on dataset preparation, model training, model evaluation, and performance comparison.

The dataset was prepared from an existing cat breed image dataset uploaded to Google Drive. The images were organized by breed class, cleaned, standardized, and split into training, validation, and testing datasets. A web crawler script is also included in this repository as an optional image collection tool, but the final training dataset used in this project was prepared from the uploaded cat breed dataset.

Detailed analysis of confusion matrices, training accuracy, validation accuracy, training loss, validation loss, accuracy, mean average precision (mAP), parameter count, and training time are included to provide comprehensive insights into model performance.

## Overview

Cat breed classification is a challenging computer vision task because many breeds have similar visual features, fur patterns, face shapes, and colors. This project applies transfer learning using three popular CNN architectures available in Keras:

- ResNet50
- DenseNet121
- MobileNetV3

The goal of this project is to train and compare these models for cat breed classification, evaluate their performance on the testing dataset, and determine which model is most suitable based on accuracy, mAP, training time, and number of parameters.

## Contents

### Attached Files

`catBreed_ImagesPrep.ipynb`  
Python notebook containing the dataset preparation process. This includes mounting Google Drive, checking the raw dataset, cleaning images, preparing the train/validation/test split, displaying dataset summaries, visualizing class distribution, and showing sample cat breed images.

`catBreed_Classification.ipynb`  
Python notebook containing the model training and evaluation process. This notebook trains ResNet50, DenseNet121, and MobileNetV3 for 50 epochs each, evaluates the models using the testing dataset, and generates model comparison results.

`scripts/`  
Supporting Python scripts used for dataset preparation, cleaning, visualization, model training, and evaluation.

`requirements.txt`  
List of Python libraries required to run the project.

## Confusion Matrix Analysis

The confusion matrices provide a visual summary of the classification performance for each model. They show the correct classifications along the diagonal and the misclassifications outside the diagonal.

This helps identify:

- Which cat breeds were classified correctly.
- Which breeds were commonly confused with each other.
- Possible visual similarities between misclassified cat breeds.
- Weaknesses and bias patterns in each model.

## Model Performance Comparison

The three CNN models were compared using:

- Testing accuracy
- Mean average precision (mAP)
- Training time
- Parameter count

This comparison helps determine which model provides the best balance between performance and efficiency.

## Training Accuracy and Loss Trends

The training and validation accuracy graphs show how well each model learned during training. The training and validation loss graphs help identify whether the model was learning effectively, overfitting, or underfitting.

These graphs are used to analyze:

- Accuracy improvement across 50 epochs.
- Gap between training and validation performance.
- Stability of model learning.
- Signs of overfitting or underfitting.

## Performance Metrics

Based on the final testing results:

| Model | Test Accuracy | mAP | Parameter Count | Training Time |
|---|---:|---:|---:|---:|
| ResNet50 | 73.58% | 80.42% | 23,628,692 | 1884.31s |
| DenseNet121 | 75.47% | 81.48% | 7,058,004 | 3928.08s |
| MobileNetV3 | 74.91% | 82.04% | 3,015,572 | 1583.47s |

## Final Conclusion

DenseNet121 achieved the highest testing accuracy at 75.47%, while MobileNetV3 achieved the highest mAP at 82.04%. MobileNetV3 also had the lowest parameter count and the fastest training time compared to the other models.

Although DenseNet121 produced the highest accuracy, MobileNetV3 provides the best overall balance between accuracy, mAP, model size, and training time. Therefore, MobileNetV3 is considered the most suitable model for this cat breed classification task.

## Project By

1. Myra Jasmeen Daniella Binti Bakar Jamili
2. Muhammad Iskandar

