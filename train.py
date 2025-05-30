from model import create_model
from keras import datasets
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # Preprocess the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    print(x_train.shape)

    # Plot 5 random samples
    plt.figure(figsize=(10, 2))
    for i in range(5):
        idx = np.random.randint(0, len(x_train))
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_train[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {y_train[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=300, bbox_inches='tight')

    # Create the model
    model = create_model()
    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)
    
    # Save the trained model
    model.save('mnist_model.keras')
    print("Model saved as 'mnist_model.keras'")

    # Plot 5 random samples and their predictions
    plt.figure(figsize=(10, 2))
    for i in range(5):
        idx = np.random.randint(0, len(x_test))
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {y_test[idx]}\n Prediction: {model.predict(x_test[idx:idx+1]).argmax()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples_predictions.png', dpi=300, bbox_inches='tight')
