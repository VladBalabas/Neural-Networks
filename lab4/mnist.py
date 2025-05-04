import numpy as np
import matplotlib.pyplot as plt
from keras.src.datasets import mnist
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

rows, cols = 10, 20
fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

output_image = np.zeros((28 * rows, 28 * cols), dtype=np.uint8)

for i in range(rows):
    digit = i
    digit_indices = np.where(y_train == digit)[0]
    selected_indices = np.random.choice(digit_indices, cols, replace=False)

    for j in range(cols):
        img = x_train[selected_indices[j]]
        output_image[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img

plt.figure(figsize=(10, 5))
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.show()