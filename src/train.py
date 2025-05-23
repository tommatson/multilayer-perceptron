import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize to [0, 1] and binarize (threshold = 0.5)
x_train = (x_train / 255.0 > 0.5).astype(int)

# Flatten images to 1D arrays (28x28 = 784)
x_train_flat = x_train.reshape((-1, 784))

# Save the first 1000 samples for demo
np.savetxt("mnist_binary_input.csv", x_train_flat[:1000], fmt='%d', delimiter=',')
np.savetxt("mnist_labels.csv", y_train[:1000], fmt='%d', delimiter=',')
