import os
from collections import Counter

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, jaccard_score, accuracy_score, \
    multilabel_confusion_matrix, confusion_matrix
import pandas as pd
import seaborn as sns

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = r"D:\lab5_neuro\task3_dataset\train"
TEST_DIR = r"D:\lab5_neuro\task3_dataset\test"

CLASSES = ['nothing', 'pen', 'pencil', 'marker',
           'pen_pencil', 'pen_marker', 'pencil_marker', 'pen_pencil_marker']

CLASS_TO_VEC = {
    'pen': [1, 0, 0],
    'pencil': [0, 1, 0],
    'marker': [0, 0, 1],
    'nothing': [0, 0, 0],
    'pen_pencil': [1, 1, 0],
    'pen_marker': [1, 0, 1],
    'pencil_marker': [0, 1, 1],
    'pen_pencil_marker': [1, 1, 1]
}

VEC_TO_CLASS = {tuple(v): idx for idx, (k, v) in enumerate(CLASS_TO_VEC.items())}

def load_dataset(directory):
    images = []
    labels = []

    for fname in os.listdir(directory):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            class_name = '_'.join(fname.split('_')[:-1])
            img_path = os.path.join(directory, fname)

            label = CLASS_TO_VEC[class_name]
            images.append(img_path)
            labels.append(label)

    return images, np.array(labels, dtype=np.float32)


def create_dataset(img_paths, labels, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(4, 4)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(256, (1, 1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='sigmoid')
    ])
    return model

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='sigmoid')
    ])
    return model

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(4, 4)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(512, (1, 1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='sigmoid')
    ])
    return model

train_paths, train_labels = load_dataset(TRAIN_DIR)
test_paths, test_labels = load_dataset(TEST_DIR)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42
)

train_dataset = create_dataset(train_paths, train_labels, is_training=True)
val_dataset = create_dataset(val_paths, val_labels, is_training=False)
test_dataset = create_dataset(test_paths, test_labels, is_training=False)

model = build_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# ==========================
# РОЗРАХУНОК МЕТРИК
# ==========================

def vector_to_class(vecs):
    return np.array([VEC_TO_CLASS[tuple(v)] for v in vecs])

y_true = []
y_pred = []

for batch in test_dataset:
    images, labels = batch
    preds = model.predict(images)
    if images.shape[0] == 0:
        break
    y_true.append(labels.numpy())
    y_pred.append((preds > 0.5).astype(int))

y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

hamming = hamming_loss(y_true, y_pred)
mean_jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=1)

y_true_classes = vector_to_class(y_true)
y_pred_classes = vector_to_class(y_pred)

precision_per_class = precision_score(y_true_classes, y_pred_classes, average=None, labels=range(len(CLASSES)), zero_division=1)
recall_per_class = recall_score(y_true_classes, y_pred_classes, average=None, labels=range(len(CLASSES)), zero_division=1)
f1_per_class = f1_score(y_true_classes, y_pred_classes, average=None, labels=range(len(CLASSES)), zero_division=1)

macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=1)
micro_f1 = f1_score(y_true_classes, y_pred_classes, average='micro', zero_division=1)
weighted_f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=1)

exact_match = accuracy_score(y_true_classes, y_pred_classes)

metrics_table = pd.DataFrame({
    'Class': CLASSES,
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-score': f1_per_class
})

print("\n=== Метрики по кожному класу ===")
print(metrics_table)

print("\n=== Загальні метрики ===")
print(f"Exact Match Ratio (Accuracy): {exact_match:.4f}")
print(f"Hamming Loss: {hamming:.4f}")
print(f"Mean Jaccard Index: {mean_jaccard:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

cm = confusion_matrix(y_true_classes, y_pred_classes, labels=range(len(CLASSES)))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

