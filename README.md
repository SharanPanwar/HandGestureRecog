# Hand Gesture Recognition Project

This project implements a Convolutional Neural Network (CNN) to recognize hand gestures from the "LeapGestRecog" dataset, available on Kaggle.

## Project Overview

The project follows these main steps:

1.  **Dataset Download:**
    * Utilizes `kagglehub` to download the "gti-upm/leapgestrecog" dataset.
2.  **Data Loading and Preprocessing:**
    * Loads images and their corresponding labels.
    * Verifies image consistency in size (240x640 grayscale).
    * Splits the dataset into training and testing sets (80/20 split).
    * Normalizes pixel values to the range [0, 1].
    * Shuffles the data to ensure randomness.
3.  **Model Building:**
    * Constructs a CNN model using TensorFlow/Keras.
    * The model architecture includes:
        * Convolutional layer (`Conv2D`) for feature extraction.
        * Max pooling layer (`MaxPooling2D`) for dimensionality reduction.
        * Flatten layer.
        * Dense layers with ReLU and softmax activation functions for classification.
    * Model Compiled using the Adam optimizer, and sparse categorical cross entropy loss function.
4.  **Model Training:**
    * Trains the CNN model on the training data.
    * Monitors training and validation loss and accuracy.
5.  **Model Evaluation:**
    * Evaluates the trained model on the test data.
    * Generates a confusion matrix and classification report to assess performance.
    * Plots the training and validation loss and accuracy.
6.  **Model Saving:**
    * Saves the trained model to a file named "hand\_recognition\_model.keras".

## Key Libraries Used

* `kagglehub`
* `numpy`
* `pandas`
* `matplotlib`
* `opencv-python (cv2)`
* `tensorflow`
* `scikit-learn`
* `seaborn`

## Dataset

The "LeapGestRecog" dataset contains images of hand gestures, categorized into 10 classes.

## Model Performance

The model achieves high accuracy on the test data, demonstrating its effectiveness in hand gesture recognition. The confusion matrix and classification report provide detailed insights into the model's performance for each gesture class.
