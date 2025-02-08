# Automated Essay Scoring (AES)

## Overview

Automated Essay Scoring (AES) is a project that explores machine learning models for evaluating and scoring student essays. The project leverages Natural Language Processing (NLP) techniques and deep learning models, specifically neural networks implemented with TensorFlow, to assess essay quality based on predefined scoring criteria.

## Features

- **Exploratory Data Analysis (EDA):** Preprocessing and visualization of text data.
- **Text Preprocessing:** Tokenization, lemmatization, stopword removal, and spelling correction.
- **Vectorization Methods:** Word2Vec and GloVe for word embeddings.
- **Modeling Approaches:**
  - Linear Regression (Ridge Regression)
  - Support Vector Regression (SVR)
  - Neural Networks with TensorFlow
- **Comparison with Human Raters:** Evaluates model performance against human-graded essays.
- **Automated Text Generation:** Experimenting with neural networks to generate synthetic essays.

## Dataset

The dataset used in this project originates from the **Automated Essay Scoring Kaggle competition**. It consists of **12,978 essays** across **eight different topics**. Each essay is labeled with a score assigned by human raters.

## Model Implementation

### 1. **Data Preprocessing**

Before training models, the dataset undergoes preprocessing:

- Anonymization: Named entities (e.g., names, locations) are replaced with placeholders.
- Grammatical and spelling corrections.
- Text normalization: Lowercasing, punctuation removal, and tokenization.
- Word embeddings using **Word2Vec** (trained on dataset) and **GloVe** (pre-trained).

### 2. **Neural Network Model with TensorFlow**

A **feedforward neural network** is implemented using TensorFlow and Keras, consisting of:

- Two fully connected layers with varying numbers of neurons.
- **ReLU** activation function.
- **Mean Squared Error (MSE)** loss function.
- **Adam optimizer** for training.

### 3. **Training and Evaluation**

The dataset is split into **70% training** and **30% test**. The best performance is obtained with **Word2Vec embeddings and a linear regression model**, achieving a lower MSE compared to other approaches. Neural networks with TensorFlow provide competitive results but require fine-tuning to avoid overfitting.

## Results

- **Word2Vec + Linear Regression** achieved the best test MSE.
- **GloVe embeddings** showed overfitting on certain topics.
- **Neural networks with TensorFlow** provided competitive performance but required careful tuning.
- **Comparison with human raters** suggests that the model can outperform individual human graders in consistency.
