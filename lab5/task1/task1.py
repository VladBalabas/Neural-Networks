import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "D:/lab5_neuro/hotdog/train"
TEST_DIR = "D:/lab5_neuro/hotdog/test"
PRODUCTION_DIR = "D:/lab5_neuro/hotdog/production"
VALIDATION_SPLIT = 0.2

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

production_ds = tf.keras.utils.image_dataset_from_directory(
    PRODUCTION_DIR,
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb',
    shuffle=False
).map(preprocess_image)

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
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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


def evaluate_and_show_confusion_and_examples(dataset, model, dataset_name):
    y_true = []
    y_pred = []
    images_list = []

    for images, labels in dataset:
        preds = model.predict(images) > 0.5
        y_true.extend(labels.numpy())
        y_pred.extend(preds.astype(int).flatten())
        images_list.extend(images)

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not-hotdog", "hotdog"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {dataset_name}")
    plt.show()

    categories = {
        "TP": (1, 1),
        "TN": (0, 0),
        "FP": (0, 1),
        "FN": (1, 0)
    }

    shown = {key: False for key in categories}
    plt.figure(figsize=(12, 3))
    count = 1

    for i in range(len(y_true)):
        pair = (y_true[i], y_pred[i])
        for key, val in categories.items():
            if pair == val and not shown[key]:
                plt.subplot(1, 4, count)
                plt.imshow(tf.squeeze(images_list[i]))
                plt.title(f"{key}\nTrue: {val[0]}, Pred: {val[1]}")
                plt.axis('off')
                shown[key] = True
                count += 1
                break
        if all(shown.values()):
            break

    plt.suptitle(f"Examples for {dataset_name}")
    plt.tight_layout()
    plt.show()

    return acc


train_acc = model.evaluate(train_ds, verbose=0)[1]
test_acc = evaluate_and_show_confusion_and_examples(test_ds, model, "Test Dataset")
production_acc = evaluate_and_show_confusion_and_examples(production_ds, model, "Production Dataset")

print("\n=== Final Accuracy Table ===")
print(f"Train Accuracy     : {train_acc * 100:.2f}%")
print(f"Test Accuracy      : {test_acc * 100:.2f}%")
print(f"Production Accuracy: {production_acc * 100:.2f}%")
