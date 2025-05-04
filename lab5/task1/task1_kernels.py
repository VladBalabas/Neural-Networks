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

# Завантаження датасету
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

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

# Комбінації ядер згортки
kernel_combinations = [
    ((3,3), (3,3), (3,3)),
    ((3,3), (5,5), (3,3)),
    ((5,5), (5,5), (3,3)),
    ((5,5), (7,7), (3,3)),
    ((3,3), (7,7), (3,3)),
    ((7,7), (5,5), (3,3)),
    ((7,7), (7,7), (3,3)),
    ((3,3), (3,3), (1,1)),
    ((5,5), (5,5), (1,1)),
    ((7,7), (5,5), (1,1)),
    ((7,7), (7,7), (1,1)),
    ((5,5), (3,3), (1,1)),
]

results = []

# Функція для створення моделі
def create_model(kernels):
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        Conv2D(32, kernels[0], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernels[1], activation='relu'),
        MaxPooling2D(pool_size=(4, 4)),

        Conv2D(128, kernels[2], activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Тренування моделей
for idx, kernels in enumerate(kernel_combinations, start=1):
    print(f"Тренування моделі {idx}: {kernels}")
    model = create_model(kernels)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    results.append({
        "model_number": idx,
        "kernels": kernels,
        "accuracy": accuracy,
        "model": model,
        "history": history
    })

# Вивід таблиці результатів
print("\nТаблиця результатів:")
print(f"{'№':<5}{'Ядро 1':<10}{'Ядро 2':<10}{'Ядро 3':<10}{'Точність':<10}")
for res in results:
    k1, k2, k3 = res['kernels']
    print(f"{res['model_number']:<5}{str(k1):<10}{str(k2):<10}{str(k3):<10}{res['accuracy']:.4f}")

# Вибір 3 найкращих моделей
top3 = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:3]

# Функції для візуалізації
def plot_training_history(history, model_number):
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Модель {model_number}: Навчання')

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def show_confusion_matrix(dataset, model, model_number):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images) > 0.5
        y_true.extend(labels.numpy())
        y_pred.extend(preds.astype(int).flatten())

    cm = confusion_matrix(np.array(y_true), np.array(y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not-hotdog", "hotdog"])
    disp.plot(cmap='Blues')
    plt.title(f"Модель {model_number}: Матриця помилок")
    plt.show()

# Візуалізація для топ-3 моделей
for res in top3:
    model_number = res['model_number']
    model = res['model']
    history = res['history']

    plot_training_history(history, model_number)
    show_confusion_matrix(test_ds, model, model_number)
