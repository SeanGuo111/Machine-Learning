import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
keras = tf.keras


model = tf.keras.Sequential([
  keras.layers.Dense(25, input_shape=(400,), activation="sigmoid"),
  keras.layers.Dense(10, activation = "softmax")
])

#model.compile(tf.keras.optimizers.RMSprop(0.001), loss='mse')
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

data = np.zeros((3500, 400))
labels = np.ones((3500, 10))
model.fit(data, labels, epochs=10)
