# Fashion-MNIST Apparel Classification

## Introduction

This project addresses a significant challenge in e-commerce: the accurate categorization of apparel and accessories from images. More than 25% of the entire revenue in e-commerce is attributed to apparels and accessories. A major problem in this domain is the inconsistent categorization provided by various brands, which makes automated classification difficult. This problem presents an interesting computer vision challenge that has attracted the attention of numerous deep learning researchers.

This repository provides a solution for identifying different types of apparel using the Fashion MNIST dataset, a direct replacement for the well-known MNIST digit dataset. Instead of handwritten digits, the images in Fashion MNIST depict various clothing items like T-shirts, trousers, bags, etc. The dataset was originally created by Zalando Research.

## Dataset

The Fashion MNIST dataset comprises a total of 70,000 grayscale images, each sized 28x28 pixels.
* **Training Set**: 60,000 images with corresponding labels.
* **Test Set**: 10,000 unlabelled images.

## Problem Statement

The primary task is to accurately identify and classify the type of apparel for all images in the test set.

## Class Labels

The dataset includes 10 distinct apparel classes, with their corresponding numerical labels:

| Label | Description   |
| :---- | :------------ |
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

## Project Structure

The project is organized to handle data loading, preprocessing, model definition, training, evaluation, and prediction generation.

### Key Libraries Used:
* `numpy`
* `pandas`
* `matplotlib.pyplot`
* `os`
* `struct`
* `sklearn.model_selection` (for `train_test_split`)
* `sklearn.metrics` (for `confusion_matrix`, `classification_report`)
* `tensorflow`
* `tensorflow.keras.models` (for `Sequential`)
* `tensorflow.keras.layers` (for `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `BatchNormalization`)
* `tensorflow.keras.callbacks` (for `EarlyStopping`, `ReduceLROnPlateau`)
* `tensorflow.keras.preprocessing.image` (for `ImageDataGenerator`)

### Data Loading and Preprocessing:
The notebook includes a custom `load_idx_file` function to handle the IDX file format used by Fashion MNIST for images and labels.
Images are normalized to a `[0, 1]` range and reshaped to include a channel dimension for Keras compatibility (e.g., `(num_images, 28, 28, 1)`). Labels are extracted from a CSV file.# fashionMNIST

The notebook includes logic to check for file existence and attempts alternative paths for image files if the primary paths are not found.

## Model Architecture

The model utilizes a Convolutional Neural Network (CNN) built with TensorFlow/Keras. While the full architecture details are within the `fashion_mnist.ipynb` file, it incorporates:
* Convolutional Layers (`Conv2D`)
* Max Pooling Layers (`MaxPooling2D`)
* Batch Normalization (`BatchNormalization`)
* Dropout Layers (`Dropout`) for regularization
* Flatten Layer (`Flatten`)
* Dense Layers (`Dense`) for classification

Callbacks like `EarlyStopping` and `ReduceLROnPlateau` are used during training to optimize performance and prevent overfitting.
