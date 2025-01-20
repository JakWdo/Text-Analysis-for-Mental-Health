# Text Analysis for Mental Health and Emotions

This repository contains the implementation of two text analysis systems:

1. A classification system for mental health-related posts from Reddit to identify different mental health disorders.
2. A regression model for analyzing emotional dimensions in text using the VAD (Valence-Arousal-Dominance) model based on the EmoBank dataset.

## Project Overview

### Mental Health Classification
The first part of the project focuses on classifying Reddit posts into different mental health categories:
- Stress
- Depression
- Bipolar Disorder
- Personality Disorders
- Anxiety

The model uses TF-IDF vectorization and Logistic Regression with balanced class weights to handle the uneven distribution of categories in mental health-related discussions.

### Emotion Analysis
The second part implements a deep learning model to analyze three emotional dimensions in text:
- Valence (V): positive vs. negative
- Arousal (A): active vs. passive
- Dominance (D): dominant vs. submissive

The model uses a neural network architecture with TF-IDF features as input to predict these three dimensions simultaneously.

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
pandas>=1.2.3
matplotlib>=3.3.4
seaborn>=0.11.1
scikit-learn>=0.24.1
```

## Model Architecture

### Mental Health Classification Model
- TF-IDF Vectorizer with 10,000 features
- N-gram range: (1,2)
- Logistic Regression with balanced class weights
- Validation using stratified splitting

### Emotion Regression Model
- Input layer: TF-IDF vectors (5000 features)
- Hidden layers: 128 -> 64 units with ReLU activation
- Output layer: 3 units (V-A-D predictions)
- Training:
  - Optimizer: Adam
  - Loss: MSE
  - Batch size: 32
  - Epochs: 20

## Results

### Mental Health Classification
The model achieves balanced performance across different mental health categories, with particular effectiveness in identifying:
- Depression
- Anxiety
- Stress-related posts

### Emotion Analysis
The regression model successfully predicts emotional dimensions with MSE scores:
- Valence: ~0.08
- Arousal: ~0.09
- Dominance: ~0.08
