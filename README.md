# Deep Learning Assignment Solutions  

This repository contains implementations of fundamental deep learning models for both **image** and **text classification** tasks. The project demonstrates how to preprocess data, build and train neural networks, evaluate performance, and visualize results.  

---

## Project Description  

The project is divided into three main parts:  

### 1. Feedforward Neural Network (FFN)  
- **Dataset:** MNIST (handwritten digits).  
- **Goal:** Classify digits (0–9) from 28×28 grayscale images.  
- **Architecture:**  
  - Input layer  
  - Two hidden layers with ReLU activation  
  - Output layer with Softmax activation (10 classes)  
- **Unique Aspects:** Simple yet effective baseline model for classification.  
- **Outputs:** Training/validation loss and accuracy plots.  

---

### 2. Convolutional Neural Network (CNN)  
- **Dataset:** CIFAR-10 (colored images, 10 classes).  
- **Goal:** Recognize objects (airplane, car, bird, cat, etc.).  
- **Architecture:**  
  - Convolutional layers with ReLU activation  
  - MaxPooling for downsampling  
  - Dropout for regularization  
  - Fully connected dense layers + Softmax output  
- **Unique Aspects:**  
  - Captures spatial hierarchies in images.  
  - Includes **feature map visualization** to show how filters detect patterns like edges, textures, and shapes.  
- **Outputs:** Training/validation curves + feature map images.  

---

### 3. Recurrent Neural Network (RNN) with LSTM  
- **Dataset:** IMDB Movie Reviews (binary sentiment classification).  
- **Goal:** Predict whether a review is **positive** or **negative**.  
- **Architecture:**  
  - Embedding layer for word representation  
  - LSTM (Long Short-Term Memory) layer for sequence modeling  
  - Dense output layer with Sigmoid activation  
- **Unique Aspects:**  
  - Handles sequential text data effectively.  
  - Demonstrates how deep learning can capture language context for sentiment analysis.  
- **Outputs:** Training/validation accuracy/loss plots + prediction demo on sample reviews.  

---

## How It Works  

1. **Data Preprocessing**  
   - MNIST/CIFAR-10 images normalized to [0,1].  
   - IMDB reviews tokenized and padded to equal length.  

2. **Model Building**  
   - Each model tailored to its dataset:  
     - FFN for flattened pixel data  
     - CNN for spatial image features  
     - LSTM for sequential text features  

3. **Training & Evaluation**  
   - Optimizers: Adam  
   - Loss functions:  
     - Categorical Crossentropy (FFN, CNN)  
     - Binary Crossentropy (LSTM)  
   - Validation split used for performance monitoring.  

4. **Visualization**  
   - Accuracy and loss plotted across epochs.  
   - CNN feature maps extracted and visualized for interpretability.  

---

## Key Highlights  

- Implements **three core deep learning models** in one repository.  
- Covers **both image and natural language processing (NLP) tasks**.  
- Includes **visual interpretability** via CNN feature maps.  
- Provides **clean, modular, and reproducible code** with model checkpoints.  
- Great as a **learning reference** for students and beginners in deep learning.  

---
