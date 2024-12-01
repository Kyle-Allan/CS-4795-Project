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

# Define class names
class_names = [f"MNIST {i}" for i in range(10)] + [f"Fashion {i}" for i in range(10)]

# Function to compute and display confusion matrix
def plot_confusion_matrix(model, test_X, test_y, model_name):
    # Get predictions
    y_pred = model.predict(test_X).argmax(axis=1)
    y_true = test_y.argmax(axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='viridis', xticks_rotation=90)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()


mnist_class_names = [f"Digit {i}" for i in range(10)]
fashion_mnist_class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Combine MNIST and Fashion MNIST class names
class_names = mnist_class_names + fashion_mnist_class_names

# Function to plot misclassified instances with semantic labels
def plot_balanced_misclassified_images(model, test_X, test_y, model_name, class_names, num_images=10):
    # Get predictions and true labels
    y_pred = model.predict(test_X).argmax(axis=1)
    y_true = test_y.argmax(axis=1)

    # Identify misclassified indices
    misclassified_indices = np.where(y_pred != y_true)[0]

    # Separate misclassifications for MNIST and Fashion MNIST
    mnist_misclassified = [idx for idx in misclassified_indices if y_true[idx] < 10]
    fashion_misclassified = [idx for idx in misclassified_indices if y_true[idx] >= 10]

    # Determine number of images to display from each category
    num_mnist = min(len(mnist_misclassified), num_images // 2)
    num_fashion = min(len(fashion_misclassified), num_images - num_mnist)

    # Combine selected indices
    selected_indices = mnist_misclassified[:num_mnist] + fashion_misclassified[:num_fashion]

    # Plot misclassified images
    plt.figure(figsize=(15, 5))
    for i, index in enumerate(selected_indices):
        plt.subplot(1, len(selected_indices), i + 1)
        plt.imshow(test_X[index].squeeze(), cmap='gray')
        plt.title(f"True: {class_names[y_true[index]]}\nPred: {class_names[y_pred[index]]}")
        plt.axis('off')
    plt.suptitle(f"Misclassified Instances: {model_name}", fontsize=16)
    plt.show()



# Evaluate models, plot confusion matrices, and show balanced misclassified instances
print("\nConfusion Matrices and Misclassified Instances for Loaded Models:")

print("Baseline Model:")
plot_confusion_matrix(baseline_model_loaded, test_X, test_y, "Baseline Model")
plot_balanced_misclassified_images(baseline_model_loaded, test_X, test_y, "Baseline Model", class_names, num_images=10)

print("Deeper CNN:")
plot_confusion_matrix(deeper_cnn_loaded, test_X, test_y, "Deeper CNN")
plot_balanced_misclassified_images(deeper_cnn_loaded, test_X, test_y, "Deeper CNN", class_names, num_images=10)

print("Very Deep CNN:")
plot_confusion_matrix(very_deep_cnn_loaded, test_X, test_y, "Very Deep CNN")
plot_balanced_misclassified_images(very_deep_cnn_loaded, test_X, test_y, "Very Deep CNN", class_names, num_images=10)

