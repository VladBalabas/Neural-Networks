import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential, Input
from keras.src.layers import InputLayer, Conv2D, AveragePooling2D
from keras.src.utils import to_categorical

from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageEnhance, ImageOps


def prepare_dataset(image_path, cell_size=(28, 28), rows=10, cols=20):
    image = Image.open(image_path).convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = ImageOps.invert(image)

    dataset = []
    labels = []
    cell_width, cell_height = cell_size

    for row in range(rows):
        for col in range(cols):
            left, upper = col * cell_width, row * cell_height
            right, lower = left + cell_width, upper + cell_height
            digit_image = image.crop((left, upper, right, lower))
            digit_array = np.array(digit_image, dtype=np.float32) / 255.0
            digit_array = np.expand_dims(digit_array, axis=-1)
            dataset.append(digit_array)
            labels.append(row)

    return np.array(dataset), np.array(labels)


def preprocess_data(x_train, x_test, y_train, y_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, x_test, y_train, y_test


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train, x_test, y_train, y_test = preprocess_data(x_train, x_test, y_train, y_test)

model = Sequential([
    Input(shape=(28, 28, 1)),

    Conv2D(filters=6, kernel_size=(5, 5), padding="same", activation="relu"),
    AveragePooling2D(pool_size=(2, 2), strides=2),

    Conv2D(filters=16, kernel_size=(5, 5), activation="relu"),
    AveragePooling2D(pool_size=(2, 2), strides=2),

    Flatten(),

    Dense(120, activation="relu"),
    Dense(84, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

image_path = "лаб4.8.jpeg"
custom_data, custom_labels = prepare_dataset(image_path)

predictions = model.predict(custom_data)
predicted_labels = np.argmax(predictions, axis=1)

conf_matrix = confusion_matrix(custom_labels, predicted_labels)
print(conf_matrix)

rows, cols = 10, 20
fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        axes[i, j].imshow(custom_data[idx].squeeze(), cmap="gray")
        axes[i, j].axis("off")
        axes[i, j].set_title(f"{predicted_labels[idx]}", fontsize=8)

plt.tight_layout()
plt.show()
