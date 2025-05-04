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
VALIDATION_SPLIT = 0.2  # 20% на валідацію

# Функція для обробки зображень
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

# Завантаження датасетів
def load_datasets():
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

    return train_ds, val_ds, test_ds

# Функція для побудови моделі з заданою кількістю згорткових шарів
def build_model(num_conv_layers):
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))

    # Перший Conv шар
    model.add(Conv2D(32, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Додаємо додаткові згорткові шари
    for i in range(1, num_conv_layers):
        model.add(Conv2D(32 * (2 ** min(i, 3)), (3, 3), activation='relu'))  # Збільшуємо кількість фільтрів
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Функція для побудови графіків
def plot_training_history(history, num_layers):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss vs Epochs (Layers: {num_layers})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Accuracy vs Epochs (Layers: {num_layers})')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Функція для оцінки моделі
def evaluate_model(model, test_ds):
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Головна функція для запуску експериментів
def experiment_with_conv_layers(layer_counts):
    train_ds, val_ds, test_ds = load_datasets()
    results = {}

    for num_layers in layer_counts:
        print(f"\n==== Training model with {num_layers} convolutional layers ====")
        model = build_model(num_layers)
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

        plot_training_history(history, num_layers)
        test_acc = evaluate_model(model, test_ds)

        results[num_layers] = test_acc

    # Побудова фінальної діаграми
    plt.figure(figsize=(8, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.title('Test Accuracy vs Number of Conv Layers')
    plt.xlabel('Number of Conv Layers')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.show()

    return results

# Запустити експерименти
layer_counts_to_test = [2, 3, 4, 5]  # Наприклад, від 2 до 5 згорткових шарів
results = experiment_with_conv_layers(layer_counts_to_test)

# Показати результати
for num_layers, acc in results.items():
    print(f"{num_layers} Conv Layers: Test Accuracy = {acc:.4f}")
