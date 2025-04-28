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

# Create files.

new_dirs = {
    "train": os.path.join(root_data_path, "train"),
    "val": os.path.join(root_data_path, "val"),
    "test": os.path.join(root_data_path, "test")
}

for new_dir in new_dirs:
    print("Building " + new_dir + ".")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)

# Split dataset.

image_data_path = os.path.join(root_data_path, "images_original")

for genre in os.listdir(image_data_path):

    images = os.listdir(os.path.join(image_data_path, genre))
    random.shuffle(images)
    n = len(images)

    train_images = images[:int(0.8 * n)]
    val_images = images[int(0.8 * n):int(0.9 * n)]
    test_images = images[int(0.9 * n):]

    for image in train_images:
        shutil.copy(os.path.join(image_data_path, genre, image), new_dirs["train"])
    for image in val_images:
        shutil.copy(os.path.join(image_data_path, genre, image), new_dirs["val"])
    for image in test_images:
        shutil.copy(os.path.join(image_data_path, genre, image), new_dirs["test"])