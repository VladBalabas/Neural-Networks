from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt


def prepare_dataset(image_path, output_dir, cell_size=(28, 28), rows=10, cols=20):
    image = Image.open(image_path).convert("L")

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = ImageOps.invert(image)

    cell_width, cell_height = cell_size
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    dataset = []
    labels = []

    for row in range(rows):
        for col in range(cols):
            left, upper = col * cell_width, row * cell_height
            right, lower = left + cell_width, upper + cell_height
            digit_image = image.crop((left, upper, right, lower))

            digit_array = np.array(digit_image, dtype=np.float32) / 255.0
            digit_array = np.expand_dims(digit_array, axis=-1)

            dataset.append(digit_array)
            labels.append(row)

            save_path = os.path.join(output_dir, str(row), f"{col}.png")
            digit_image.save(save_path)

    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels


if __name__ == "__main__":
    image_path = "лаб4.8.jpeg"
    output_dir = "numbers"
    dataset, labels = prepare_dataset(image_path, output_dir)

    rows, cols = 10, 20
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            axes[i, j].imshow(dataset[idx].squeeze(), cmap="gray")
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{labels[idx]}", fontsize=8)

    plt.tight_layout()
    plt.show()



