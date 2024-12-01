from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

def load_and_combine_datasets():
    # Load MNIST and Fashion MNIST datasets
    (mnist_train_X, mnist_train_y), (mnist_test_X, mnist_test_y) = mnist.load_data()
    (fashion_train_X, fashion_train_y), (fashion_test_X, fashion_test_y) = fashion_mnist.load_data()

    # Make the Fashion MNIST labels writable
    fashion_train_y = fashion_train_y.copy()
    fashion_test_y = fashion_test_y.copy()

    # Relabel Fashion MNIST classes to avoid overlap with MNIST
    fashion_train_y += 10
    fashion_test_y += 10

    # Combine train and test sets
    train_X = np.concatenate((mnist_train_X, fashion_train_X), axis=0)
    train_y = np.concatenate((mnist_train_y, fashion_train_y), axis=0)
    test_X = np.concatenate((mnist_test_X, fashion_test_X), axis=0)
    test_y = np.concatenate((mnist_test_y, fashion_test_y), axis=0)

    # Normalize the pixel values to the range [0, 1]
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    # Reshape to add channel dimension (28x28x1)
    train_X = train_X.reshape(-1, 28, 28, 1)
    test_X = test_X.reshape(-1, 28, 28, 1)

    # One-hot encode the labels
    train_y = to_categorical(train_y, 20)  # 20 classes: 0-9 for MNIST, 10-19 for Fashion MNIST
    test_y = to_categorical(test_y, 20)

    return train_X, train_y, test_X, test_y

# Load combined dataset
train_X, train_y, test_X, test_y = load_and_combine_datasets()

# Load models
baseline_model_loaded = load_model("baseline_model.h5")
deeper_cnn_loaded = load_model("deeper_cnn_model.h5")
very_deep_cnn_loaded = load_model("very_deep_cnn_model.h5")

# Evaluate the loaded models on the test set
print("\nEvaluating Loaded Models:")
baseline_loss, baseline_accuracy = baseline_model_loaded.evaluate(test_X, test_y, verbose=2)
print(f"Baseline Model Test Accuracy (Loaded): {baseline_accuracy:.4f}")

deeper_loss, deeper_accuracy = deeper_cnn_loaded.evaluate(test_X, test_y, verbose=2)
print(f"Deeper CNN Test Accuracy (Loaded): {deeper_accuracy:.4f}")

very_deep_loss, very_deep_accuracy = very_deep_cnn_loaded.evaluate(test_X, test_y, verbose=2)
print(f"Very Deep CNN Test Accuracy (Loaded): {very_deep_accuracy:.4f}")
