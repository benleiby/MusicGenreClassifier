import shutil
import numpy as np
import torchvision.transforms as transforms
import os
import torch
from PIL import Image

def crop_white_border(img, threshold=245):
    img_np = np.array(img)
    mask = np.any(img_np < threshold, axis=-1)
    cords = np.argwhere(mask)
    if cords.size == 0:
        return img
    y0, x0 = cords.min(axis=0)
    y1, x1 = cords.max(axis=0) + 1
    return img.crop((x0, y0, x1, y1))

class CropWhiteBorder:
    def __init__(self, threshold=245):
        self.threshold = threshold
    def __call__(self, img):
        return crop_white_border(img, self.threshold)

def first_transformation(root_dataset_path, root_input_path):

    transform = transforms.Compose([
        CropWhiteBorder(threshold=245),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    stats = {  # stats[split] = (mean, std_dev)
        "test": ([], []),
        "train": ([], []),
        "val": ([], [])
    }

    splits = os.listdir(root_dataset_path)

    for split in splits:
        split_directory = os.path.join(root_dataset_path, split)

        # Create input split directory.

        if os.path.exists(os.path.join(root_input_path, split)):
            shutil.rmtree(os.path.join(root_input_path, split))
        os.makedirs(os.path.join(root_input_path, split))

        split_tensors = []

        for genre in os.listdir(split_directory):
            genre_directory = os.path.join(split_directory, genre)

            genre_tensors = []

            # Apply transformation.

            for image_file in os.listdir(genre_directory):
                image_path = os.path.join(genre_directory, image_file)
                tensor = transform(Image.open(image_path).convert("RGB"))
                genre_tensors.append(tensor)
                split_tensors.append(tensor)

            torch.save(genre_tensors, os.path.join(root_input_path, split, genre + "_tensors.pt"))

        # Stack the input and compute mean/std for each split.

        stacked = torch.stack(split_tensors)
        mean = stacked.mean(dim=(0, 2, 3))  # mean per channel
        std = stacked.std(dim=(0, 2, 3))  # std per channel
        stats[split] = (mean, std)


    torch.save(stats, os.path.join(root_input_path, "stats.pt"))

def second_transformation(root_dataset_path, root_input_path):

    splits = os.listdir(root_dataset_path)
    stats = torch.load(os.path.join(root_input_path, "stats.pt"))

    for split in splits:
        split_directory = os.path.join(root_dataset_path, split)

        # Create input split directory.

        if os.path.exists(os.path.join(root_input_path, split)):
            shutil.rmtree(os.path.join(root_input_path, split))
        os.makedirs(os.path.join(root_input_path, split))

        transform = transforms.Compose([
            CropWhiteBorder(threshold=245),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(stats["train"][0], stats["train"][1])
        ])

        for genre in os.listdir(split_directory):
            genre_directory = os.path.join(split_directory, genre)

            genre_tensors = []

            # Apply transformation.

            for image_file in os.listdir(genre_directory):
                image_path = os.path.join(genre_directory, image_file)
                tensor = transform(Image.open(image_path).convert("RGB"))
                genre_tensors.append(tensor)

            torch.save(genre_tensors, os.path.join(root_input_path, split, genre + "_tensors.pt"))

# Main procedure.

root_dataset_path = os.path.join("..", "..", "data", "dataset")
root_input_path = os.path.join("..", "..", "data", "input")

# Create input directory.

if os.path.exists(root_input_path):
    shutil.rmtree(root_input_path)
os.makedirs(root_input_path)

# First pass: make the tensors and calculate the mean and std dev across each split.

first_transformation(root_dataset_path, root_input_path)
print("First transformation successful")

# Second pass: re-make the tensors and normalize each split.

second_transformation(root_dataset_path, root_input_path)
print("Second transformation successful")