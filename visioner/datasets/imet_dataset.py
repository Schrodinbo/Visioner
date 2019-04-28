import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class IMetDataset(Dataset):
    def __init__(self, img_paths, labels, img_size=(128, 128), transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.transform = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size).transpose((2, 0, 1)) / 255.0
        label = self.labels[idx, :]
        if self.transform is not None:
            img = self.transform(img)
        # return img.float(), torch.from_numpy(np.array(label)).float()
        return torch.from_numpy(img).float(), torch.from_numpy(np.array(label)).float()
