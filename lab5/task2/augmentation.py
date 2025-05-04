import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
import shutil

dataset_path = "D:/task2_dataset"
train_path = os.path.join(dataset_path, "train")
classes = ["pen", "pencil", "marker"]

augmented_path = os.path.join(dataset_path, "train_augmented")
if os.path.exists(augmented_path):
    shutil.rmtree(augmented_path)

for class_name in classes:
    os.makedirs(os.path.join(augmented_path, class_name), exist_ok=True)


def rotate_image(img, angle):
    return img.rotate(angle)


def flip_image(img, direction="horizontal"):
    if direction == "horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img.transpose(Image.FLIP_TOP_BOTTOM)


def adjust_brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def add_noise(img, level=0.05):
    img_array = np.array(img)
    noise = np.random.normal(0, level * 255, img_array.shape)
    noisy_img_array = img_array + noise
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img_array)


def blur_image(img, radius=2):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def crop_and_resize(img, margin=0.1):
    width, height = img.size
    left = int(width * random.uniform(0, margin))
    top = int(height * random.uniform(0, margin))
    right = int(width * (1 - random.uniform(0, margin)))
    bottom = int(height * (1 - random.uniform(0, margin)))
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((width, height))


augmentation_functions = [
    (rotate_image, {"angle": lambda: random.uniform(-30, 30)}),
    (flip_image, {"direction": lambda: random.choice(["horizontal", "vertical"])}),
    (adjust_brightness, {"factor": lambda: random.uniform(0.7, 1.3)}),
    (adjust_contrast, {"factor": lambda: random.uniform(0.7, 1.3)}),
    (add_noise, {"level": lambda: random.uniform(0.01, 0.05)}),
    (blur_image, {"radius": lambda: random.uniform(0.5, 1.5)}),
    (crop_and_resize, {"margin": lambda: random.uniform(0.05, 0.15)})
]

target_per_image = 5

for class_name in classes:
    class_dir = os.path.join(train_path, class_name)
    aug_class_dir = os.path.join(augmented_path, class_name)

    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(class_dir, img_file)
        img = Image.open(img_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_name = os.path.splitext(img_file)[0]
        img_ext = os.path.splitext(img_file)[1]
        orig_save_path = os.path.join(aug_class_dir, f"{img_name}_orig{img_ext}")
        img.save(orig_save_path)

        for i in range(1, target_per_image):
            aug_img = img.copy()

            num_augs = random.randint(1, 3)
            selected_augs = random.sample(augmentation_functions, num_augs)

            for aug_func, aug_params in selected_augs:
                params = {k: v() if callable(v) else v for k, v in aug_params.items()}

                aug_img = aug_func(aug_img, **params)

            aug_save_path = os.path.join(aug_class_dir, f"{img_name}_aug{i}{img_ext}")
            aug_img.save(aug_save_path)