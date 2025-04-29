import os
import sys
import shutil
import random

root_data_path = "../../data/"

# Verify presence of images directory.

if not os.path.isdir(os.path.join(root_data_path, "images_original")):
    print("Failed to locate data.", file=sys.stderr)
else:
    print("Data located.")

# Create directories.

new_dirs = {
    "train": os.path.join(root_data_path, "train"),
    "val": os.path.join(root_data_path, "val"),
    "test": os.path.join(root_data_path, "test")
}

for new_dir_name, new_dir_path in new_dirs.items():
    print("Building " + new_dir_name + ".")
    if os.path.exists(new_dir_path):
        shutil.rmtree(new_dir_path)
    os.makedirs(new_dir_path, exist_ok=True)

# Split dataset.

image_data_path = os.path.join(root_data_path, "images_original")

for genre in os.listdir(image_data_path):

    for dir_dir_name, new_dir_path in new_dirs.items():
        os.makedirs(os.path.join(new_dir_path, genre), exist_ok=True)

    images = [img for img in os.listdir(os.path.join(image_data_path, genre))
        if img.lower().endswith(".png")
    ]

    random.shuffle(images)
    n = len(images)

    train_images = images[:int(0.8 * n)]
    val_images = images[int(0.8 * n):int(0.9 * n)]
    test_images = images[int(0.9 * n):]

    for image in train_images:
        shutil.copy(os.path.join(image_data_path, genre, image),
            os.path.join(new_dirs["train"], genre))
    for image in val_images:
        shutil.copy(os.path.join(image_data_path, genre, image),
            os.path.join(new_dirs["val"], genre))
    for image in test_images:
        shutil.copy(os.path.join(image_data_path, genre, image),
            os.path.join(new_dirs["test"], genre))