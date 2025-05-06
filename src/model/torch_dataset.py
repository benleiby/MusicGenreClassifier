import torch
from torch.utils.data import Dataset
import os

class TorchDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.label_map = {}
        label_idx = 0

        for file in os.listdir(data_dir):
            if file.endswith("_tensors.pt"):
                genre = file.replace("_tensors.pt", "")
                if genre not in self.label_map:
                    self.label_map[genre] = label_idx
                    label_idx += 1

                file_path = os.path.join(data_dir, file)
                tensors = torch.load(file_path)
                label = self.label_map[genre]

                for tensor in tensors:
                    self.samples.append((tensor, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_label_map(self):
        return self.label_map