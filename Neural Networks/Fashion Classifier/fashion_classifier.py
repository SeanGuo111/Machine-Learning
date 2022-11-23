import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
keras = tf.keras

def display_images(data):
    """Shows 9 random images."""
    fig = plt.figure()
    for i in range(9):
        fig.add_subplot(3, 3, i+1)
        plt.imshow(data[np.random.randint(0, len(data))], cmap="Greys")
        plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)

    plt.show()

def predict_image(data, labels, predictions, class_names, index):
    plt.figure()
    plt.imshow(data[index], cmap="Greys")
    plt.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False)
    
    expected_label = class_names[np.argmax(predictions[index])]
    predicted_label = class_names[np.argmax(predictions[index])]
    plt.title(f"Expected: {expected_label}\nPredicted: {predicted_label}")

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
#print(train_images.shape[0])
#show_images(train_images)

# Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

#model.fit(train_images, train_labels, epochs=10)
#model.save_weights('C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\Fashion Classifier\\fashion_classifier_weights', save_format="tf")
print("Loading weights:")
model.load_weights('C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Neural Networks\\Fashion Classifier\\fashion_classifier_weights') # Warning because optimizer is unused.

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Test Accuracy: ", test_acc)

test_predictions = model.predict(test_images)
predict_image(test_images, test_labels, test_predictions, class_names, 0)