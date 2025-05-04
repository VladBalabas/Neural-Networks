import os
import random
from PIL import Image

base_path = r'D:\pen_pencil_marker\single'
output_paths = {
    'pen_pencil': r'D:\pen_pencil_marker\pen_pencil',
    'pen_marker': r'D:\pen_pencil_marker\pen_marker',
    'pencil_marker': r'D:\pen_pencil_marker\pencil_marker',
    'pen_pencil_marker': r'D:\pen_pencil_marker\pen_pencil_marker'
}

pen_images = [os.path.join(base_path, 'pen', img) for img in os.listdir(os.path.join(base_path, 'pen')) if
              img.endswith(('.jpg', '.png', '.jpeg'))]
pencil_images = [os.path.join(base_path, 'pencil', img) for img in os.listdir(os.path.join(base_path, 'pencil')) if
                 img.endswith(('.jpg', '.png', '.jpeg'))]
marker_images = [os.path.join(base_path, 'marker', img) for img in os.listdir(os.path.join(base_path, 'marker')) if
                 img.endswith(('.jpg', '.png', '.jpeg'))]


def concat_images_horizontally(image_paths):
    images = [Image.open(p) for p in image_paths]
    heights = [img.height for img in images]
    widths = [img.width for img in images]

    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    for img in images:
        img.close()

    return new_image


def generate_images(pair, count, output_folder, name_pattern):
    for i in range(1, count + 1):
        selected_images = [random.choice(images) for images in pair]
        combined_image = concat_images_horizontally(selected_images)
        filename = f"{name_pattern}_{i}.jpg"
        save_path = os.path.join(output_folder, filename)
        combined_image.save(save_path)
        combined_image.close()


# generate_images([pen_images, pencil_images], 100, output_paths['pen_pencil'], 'pen_pencil')
generate_images([pen_images, marker_images], 128, output_paths['pen_marker'], 'pen_marker')
generate_images([pencil_images, marker_images], 128, output_paths['pencil_marker'], 'pencil_marker')

for i in range(1, 131):
    selected_images = [
        random.choice(pen_images),
        random.choice(pencil_images),
        random.choice(marker_images)
    ]
    combined_image = concat_images_horizontally(selected_images)
    filename = f"pen_pencil_marker_{i}.jpg"
    save_path = os.path.join(output_paths['pen_pencil_marker'], filename)
    combined_image.save(save_path)
    combined_image.close()

print("Готово! Все изображения сохранены.")
