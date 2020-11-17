#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Converts characters 0-9 and a-z to integer
def index(c):
    n = ord(c)
    # a number
    if n < 58:
        return n - 48
    else:
        return n - 87

images = os.listdir("./data/")

x_data = np.array([tf.image.decode_png(tf.io.read_file("data/" + x))[:,20:70,0:1] for x in images])
x_data = x_data.astype("float32")
x_data /= 255.0

y_data = np.array([index(x[0]) for x in images])

x_train = x_data[:900]
y_train = y_data[:900]

x_test = x_data[900:]
y_test = y_data[900:]

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (4, 4), activation='relu', input_shape=(50, 50, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (4, 4), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(36),
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train,
        epochs=10,
        verbose=2,
        validation_data=(x_test, y_test))
