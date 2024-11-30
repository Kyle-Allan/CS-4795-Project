import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

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



def create_unified_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(20, activation='softmax')  # 20 output classes (10 MNIST + 10 Fashion MNIST)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load combined dataset
train_X, train_y, test_X, test_y = load_and_combine_datasets()

# Create and train the model
model = create_unified_cnn()
history = model.fit(train_X, train_y, epochs=15, batch_size=32, validation_data=(test_X, test_y))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_X, test_y, verbose=2)
print(f"Test Accuracy on Combined Dataset: {test_accuracy:.4f}")


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Unified Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Get predictions and true labels
y_pred = model.predict(test_X).argmax(axis=1)
y_true = test_y.argmax(axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
class_names = [f"MNIST {i}" for i in range(10)] + [f"Fashion {i}" for i in range(10)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='viridis', xticks_rotation=90)
plt.show()

