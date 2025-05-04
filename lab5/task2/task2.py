import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.layers import AveragePooling2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from tabulate import tabulate
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "D:/lab5_neuro/task2_dataset/train_aug"
TEST_DIR = "D:/lab5_neuro/task2_dataset/test"
VALIDATION_SPLIT = 0.2


def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label


full_train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb',
    shuffle=True
).map(preprocess_image)

train_batches = tf.data.experimental.cardinality(full_train_ds).numpy()
val_size = int(train_batches * VALIDATION_SPLIT)

val_ds = full_train_ds.take(val_size)
train_ds = full_train_ds.skip(val_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb',
    shuffle=False
).map(preprocess_image)


# model = Sequential([
#     Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
#
#     Conv2D(32, (7, 7), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Conv2D(64, (5, 5), activation='relu'),
#     MaxPooling2D(pool_size=(4, 4)),
#
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(3, activation='softmax')
# ])

# model = Sequential([
#     Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
#
#     Conv2D(32, (7, 7), activation='relu'),
#     AveragePooling2D(pool_size=(2, 2)),
#
#     Conv2D(64, (5, 5), activation='relu'),
#     AveragePooling2D(pool_size=(4, 4)),
#
#     Conv2D(128, (3, 3), activation='relu'),
#     AveragePooling2D(pool_size=(2, 2)),
#
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(3, activation='softmax')
# ])

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
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

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


plot_training_history(history)


def show_confusion_and_examples(dataset, model):
    y_true = []
    y_pred = []
    images_list = []

    for images, labels in dataset:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        images_list.extend(images)

    calculate_metrics(y_true, y_pred, num_classes=3)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["pen", "pencil", "marker"])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


def calculate_metrics(y_true, y_pred, num_classes):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # MacroF1
    macro_f1 = np.mean(f1)

    # MicroF1
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    # WeightedF1
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)

    metrics = [
        ["pen", precision[0], recall[0], f1[0]],
        ["pencil", precision[1], recall[1], f1[1]],
        ["marker", precision[2], recall[2], f1[2]],
        ["MacroF1", "", "", macro_f1],
        ["MicroF1", "", "", micro_f1],
        ["WeightedF1", "", "", weighted_f1]
    ]

    headers = ["Class", "Precision", "Recall", "F1"]

    table = tabulate(metrics, headers=headers, tablefmt="grid")
    print(table)

    print(f"\nOverall Accuracy: {accuracy}")

    return table


show_confusion_and_examples(test_ds, model)
