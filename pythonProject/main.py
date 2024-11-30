import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


# Load both datasets
def load_and_preprocess_data():
    # Load MNIST
    (mnist_train_X, mnist_train_y), (mnist_test_X, mnist_test_y) = mnist.load_data()
    # Load Fashion MNIST
    (fashion_train_X, fashion_train_y), (fashion_test_X, fashion_test_y) = fashion_mnist.load_data()

    # Normalize the pixel values to the range [0, 1]
    mnist_train_X, mnist_test_X = mnist_train_X / 255.0, mnist_test_X / 255.0
    fashion_train_X, fashion_test_X = fashion_train_X / 255.0, fashion_test_X / 255.0

    # Reshape data to include a channel dimension
    mnist_train_X = mnist_train_X.reshape(-1, 28, 28, 1)
    mnist_test_X = mnist_test_X.reshape(-1, 28, 28, 1)
    fashion_train_X = fashion_train_X.reshape(-1, 28, 28, 1)
    fashion_test_X = fashion_test_X.reshape(-1, 28, 28, 1)

    # One-hot encode the labels
    mnist_train_y = to_categorical(mnist_train_y, 10)
    mnist_test_y = to_categorical(mnist_test_y, 10)
    fashion_train_y = to_categorical(fashion_train_y, 10)
    fashion_test_y = to_categorical(fashion_test_y, 10)

    return (mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y), \
           (fashion_train_X, fashion_train_y, fashion_test_X, fashion_test_y)


def create_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),  # Slight dropout for deeper models
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, train_X, train_y, test_X, test_y, dataset_name):
    history = model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(test_X, test_y), verbose=2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_X, test_y, verbose=0)
    print(f"{dataset_name} - Test Accuracy: {test_accuracy:.4f}")

    # Plot learning curves
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{dataset_name} Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load datasets
    (mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y), \
    (fashion_train_X, fashion_train_y, fashion_test_X, fashion_test_y) = load_and_preprocess_data()
    '''
    # Create and train the model for MNIST
    print("Training on MNIST dataset...")
    mnist_model = create_cnn()
    train_and_evaluate_model(mnist_model, mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y, "MNIST")
    '''
    # Create and train the model for Fashion MNIST
    print("Training on Fashion MNIST dataset...")
    fashion_model = create_cnn()
    train_and_evaluate_model(fashion_model, fashion_train_X, fashion_train_y, fashion_test_X, fashion_test_y, "Fashion MNIST")

    # Use the trained model to predict labels for the test set
    predictions = fashion_model.predict(fashion_test_X)

    # Convert predicted probabilities to class labels
    predicted_labels = predictions.argmax(axis=1)

    # Print the first 10 predicted labels and their corresponding true labels
    print("Predicted labels for the first 10 test images:", predicted_labels[:10])
    print("True labels for the first 10 test images:", fashion_test_y[:10].argmax(axis=1))

    # Define Fashion MNIST class names
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Plot the first 9 test images along with predictions
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(fashion_test_X[i].reshape(28, 28), cmap='gray')
        predicted_label = class_names[predicted_labels[i]]
        true_label = class_names[fashion_test_y[i].argmax()]
        plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
