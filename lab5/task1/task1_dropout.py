import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Параметри
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "D:/lab5_neuro/hotdog/train"
TEST_DIR = "D:/lab5_neuro/hotdog/test"
VALIDATION_SPLIT = 0.2

def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

# Завантаження даних
full_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb',
    shuffle=True,
    seed=123
).map(preprocess_image)

train_batches = tf.data.experimental.cardinality(full_train_ds).numpy()
val_size = int(train_batches * VALIDATION_SPLIT)

val_ds = full_train_ds.take(val_size)
train_ds = full_train_ds.skip(val_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb',
    shuffle=False
).map(preprocess_image)

# Функція для побудови моделі з параметром dropout_rate
def create_model(dropout_rate=0.5):
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        Conv2D(32, (7, 7), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(4, 4)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (1, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Функція для оцінки моделі
def evaluate_model(model, test_ds):
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    return accuracy

# Список dropout-коефіцієнтів для експериментів
dropout_rates = [0.0, 0.3, 0.5, 0.7]
results = {}

# Запуск експериментів
for rate in dropout_rates:
    print(f"\nТренування моделі з Dropout = {rate}")
    model = create_model(dropout_rate=rate)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)

    test_accuracy = evaluate_model(model, test_ds)
    print(f"Test accuracy with dropout {rate}: {test_accuracy:.4f}")

    results[rate] = {
        "history": history,
        "test_accuracy": test_accuracy
    }

# Побудова графіка залежності точності від коефіцієнта Dropout
plt.figure(figsize=(8, 5))
dropout_values = list(results.keys())
accuracies = [results[rate]['test_accuracy'] for rate in dropout_values]
plt.plot(dropout_values, accuracies, marker='o')
plt.title('Test Accuracy vs Dropout Rate')
plt.xlabel('Dropout Rate')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()
