import tensorflow as tf

# Load MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the data to add a channel dimension (for CNN input)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN model
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes (digits 0-9)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

import numpy as np

# Load a new digit image (e.g., from MNIST test set)
digit_image = X_test[0]  # Example: the first test image
digit_label = np.argmax(y_test[0])  # True label

# Reshape and normalize if necessary
digit_image = digit_image.reshape(1, 28, 28, 1)  # Add batch dimension

# Predict the digit
predicted_label = np.argmax(model.predict(digit_image))
print(f"True Label: {digit_label}, Predicted Label: {predicted_label}")

import cv2

def preprocess_image(image_path):
    """Preprocess a handwritten digit image for prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img = cv2.resize(img, (28, 28))  # Resize to 28x28
    img = img / 255.0  # Normalize to range [0, 1]
    img = img.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return img


# Load and preprocess your handwritten digit image
image_path = r"C:\Users\kylea\OneDrive\2024 Fall Sem\Tech Report\Seven.jpg"
digit_image = preprocess_image(image_path)

# Predict the digit
predicted_label = np.argmax(model.predict(digit_image))
print(f"Predicted Label: {predicted_label}")
