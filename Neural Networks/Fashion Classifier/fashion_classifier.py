import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
keras = tf.keras

def show_images(data):
    """Shows 9 random images."""
    fig = plt.figure()
    for i in range(9):
        fig.add_subplot(3, 3, i+1)
        plt.imshow(data[np.random.randint(0, len(train_images))], cmap="Greys")
        plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)

    plt.show()

def normalize(data):
    return data / 255

# Setup Data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = normalize(train_images) # data now needs to be normalized before entering
test_images = normalize(test_images)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Preliminary Testing
print(train_images[0])
show_images(train_images)

# Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28))
])

