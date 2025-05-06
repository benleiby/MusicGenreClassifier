import os
import sys
import shutil
import random

GTZAN_data_path = os.path.join("..", "..", "data", "GTZAN")
root_data_path = os.path.join("..", "..", "data")

# Verify presence of images directory.

image_data_path = os.path.join(GTZAN_data_path, "images_original")
if not os.path.isdir(os.path.join(GTZAN_data_path, "images_original")):
    print("Failed to locate data.", file=sys.stderr)
else:
    print("Images located.")

# Create directories.

dataset_path = os.path.join(root_data_path, "dataset")

if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)
os.makedirs(dataset_path)

new_dirs = {
    "train": os.path.join(dataset_path, "train"),
    "val": os.path.join(dataset_path, "val"),
    "test": os.path.join(dataset_path, "test")
}

for new_dir_name, new_dir_path in new_dirs.items():
    if os.path.exists(new_dir_path):
        shutil.rmtree(new_dir_path)
    os.makedirs(new_dir_path, exist_ok=True)

# Split dataset.

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

print("Successfully split data.")