# Chronic Venous Insufficiency Classifier

This project focuses on classifying images of Chronic Venous Insufficiency (CVI) into three categories: **Mild**, **Moderate**, and **Severe**. The goal is to implement an efficient method for automatic categorization of CVI images using deep learning and advanced techniques such as ensemble models.

## Project Description

Inspired by a work published in [Scientific Reports](https://www.nature.com/articles/s41598-018-36284-5), this project aims to create a hybrid approach for classifying Chronic Venous Insufficiency images. Using a combination of transfer learning and advanced augmentation, the model achieves high accuracy in categorizing CVI images into three severity levels.

## Methodology

1. **Dataset Splitting**:
   - The dataset is divided into 5 train-test splits using `StratifiedKFold` with a ratio of 2:1 (train:test).
2. **Data Augmentation**:
   - Advanced augmentation techniques are implemented using `Albumentations` for improved generalization.
3. **Ensemble Model**:
   - The model combines two backbones: EfficientNet-V2-S and ResNet-50.
   - Features extracted from both models are fused and passed through a classifier.
4. **Evaluation**:
   - Each fold's accuracy, confusion matrix, and classification report are recorded.

## Dataset

The dataset contains **211 images** divided into three classes:
- **01**: Mild
- **02**: Moderate
- **03**: Severe

The images are preloaded using PyTorch's `ImageFolder`. Ensure the dataset folder structure is as follows:
```
dataset/
├── 01/
├── 02/
└── 03/
```

## Model Architecture

The ensemble model utilizes the following:
- **EfficientNet-V2-S** and **ResNet-50** as backbone models for feature extraction.
- **Global Average Pooling** for reducing feature dimensions.
- A **fully connected classifier** that combines the extracted features to predict the class.

## Results

The model achieves an average accuracy of **92.41%** across 5 folds, exceeding the baseline accuracy of **80%** provided in the starter code.

### Classification Report:
```
              precision    recall  f1-score   support

          01       0.92      1.00      0.96        11
          02       1.00      0.80      0.89        10
          03       0.95      1.00      0.98        21

    accuracy                           0.95        42
   macro avg       0.96      0.93      0.94        42
weighted avg       0.96      0.95      0.95        42
```

### Performance Summary:
- **Average Accuracy:** 92.41%
- **Fold Accuracies:** [93.02%, 92.86%, 100.00%, 80.95%, 95.24%]

