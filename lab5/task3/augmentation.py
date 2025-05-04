import os
import random

from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array, array_to_img

source_dir = r'D:\pen_pencil_marker'
target_base = r'D:\lab5_neuro\task3_dataset'
train_dir = os.path.join(target_base, 'train')
test_dir = os.path.join(target_base, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

categories = os.listdir(source_dir)

counters = {category: 1 for category in categories}

for category in categories:
    category_path = os.path.join(source_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    test_images = images[:30]
    train_images = images[30:]

    for img_name in test_images:
        src = os.path.join(category_path, img_name)
        img = Image.open(src).convert('RGB')
        dst = os.path.join(test_dir, f"{category}_{counters[category]}.jpg")
        img.save(dst, format='JPG')
        counters[category] += 1

    for img_name in train_images:
        src = os.path.join(category_path, img_name)
        img = load_img(src)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        aug_iter = datagen.flow(x, batch_size=1)

        for _ in range(5):
            aug_image = next(aug_iter)[0].astype('uint8')
            aug_img = array_to_img(aug_image)
            aug_img = aug_img.convert('RGB')
            aug_img_path = os.path.join(train_dir, f"{category}_{counters[category]}.jpg")
            aug_img.save(aug_img_path, format='JPEG')
            counters[category] += 1

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from keras import Sequential
# from keras.src.callbacks import ReduceLROnPlateau
# from keras.src.layers import BatchNormalization
# from keras.src.optimizers import Adam
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
# from tabulate import tabulate

# from tensorflow.keras import Input
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import os
#
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 64
# EPOCHS = 10
# TRAIN_DIR = r"D:\lab5_neuro\task3_dataset\train"
# TEST_DIR = r"D:\lab5_neuro\task3_dataset\test"
#
# # Все возможные классы
# CLASSES = ['nothing', 'pen', 'pencil', 'marker',
#            'pen_pencil', 'pen_marker', 'pencil_marker', 'pen_pencil_marker']
#
# def parse_image(filename):
#     image = tf.io.read_file(filename)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, IMG_SIZE)
#     image = image / 255.0
#     return image
#
#
# def parse_label(filename):
#     name = tf.strings.split(tf.strings.split(filename, os.sep)[-1], '.')[0]
#     parts = tf.strings.split(name, '_')[:-1]
#
#     label = tf.zeros(len(CLASSES), dtype=tf.float32)
#     full_label = tf.strings.reduce_join(parts, separator='_')
#
#     match_idx = tf.argmax(tf.cast(tf.equal(full_label, CLASSES), tf.float32))
#     label = tf.tensor_scatter_nd_update(label, [[match_idx]], [1.0])
#
#     return label
#
#
# def load_dataset(directory):
#     file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.jpg')]
#     path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
#
#     image_ds = path_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
#     label_ds = path_ds.map(parse_label, num_parallel_calls=tf.data.AUTOTUNE)
#
#     ds = tf.data.Dataset.zip((image_ds, label_ds))
#     return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
#
# train_ds = load_dataset(TRAIN_DIR).shuffle(1000)
# test_ds = load_dataset(TEST_DIR)
#
# # Модель
# model = Sequential([
#     Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
#
#     Conv2D(32, (5, 5), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(len(CLASSES), activation='sigmoid')  # Мульти-метки => sigmoid + binary_crossentropy
# ])
#
# optimizer = Adam(learning_rate=0.001)
#
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#
# # Уменьшение скорости обучения при плато
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
#
# history = model.fit(
#     train_ds,
#     validation_data=test_ds,
#     epochs=EPOCHS,
#     callbacks=[lr_scheduler]
# )
#
# # Графики
# def plot_training_history(history):
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title('Loss vs Epochs')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.title('Accuracy vs Epochs')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
# plot_training_history(history)
#
# # Матрица ошибок
# def show_confusion_and_examples(dataset, model):
#     y_true = []
#     y_pred = []
#
#     for images, labels in dataset:
#         preds = model.predict(images)
#         y_true.extend(labels.numpy())
#         y_pred.extend((preds > 0.5).astype(int))
#
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#
#     for i, cls in enumerate(CLASSES):
#         cm = confusion_matrix(y_true[:, i], y_pred[:, i])
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'not {cls}', cls])
#         disp.plot(cmap='Blues')
#         plt.title(f"Confusion Matrix for class: {cls}")
#         plt.show()
#
# show_confusion_and_examples(test_ds, model)