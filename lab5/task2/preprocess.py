import os
import shutil
import random

source_directory = 'D:/pen_pencil_marker'
target_directory = 'D:/task2_dataset'

train_directory = os.path.join(target_directory, 'train')
test_directory = os.path.join(target_directory, 'test')

categories = ['pen', 'pencil', 'marker']

os.makedirs(train_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

for category in categories:
    os.makedirs(os.path.join(train_directory, category), exist_ok=True)
    os.makedirs(os.path.join(test_directory, category), exist_ok=True)

    category_path = os.path.join(source_directory, category)
    images = os.listdir(category_path)

    random.shuffle(images)

    train_images = images[:80]
    test_images = images[80:]

    for img in train_images:
        src_path = os.path.join(category_path, img)
        dst_path = os.path.join(train_directory, category, img)
        shutil.copy(src_path, dst_path)

    for img in test_images:
        src_path = os.path.join(category_path, img)
        dst_path = os.path.join(test_directory, category, img)
        shutil.copy(src_path, dst_path)