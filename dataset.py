import torch
import numpy as np
from torch.utils.data import Dataset
import random
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from glob import glob
import os

class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = glob(os.path.join(path, "**/*.npz"), recursive=True)
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        data = np.load(self.files[index])
        latent_dist_param = data['latent_dist']
        cls_ = data['cls'].tolist()
        # r = random.random()
        # if r > 0.5: latent_dist_param = latent_dist_param[1]
        # else:   
        latent_dist_param = latent_dist_param[0]
        return torch.from_numpy(latent_dist_param).unsqueeze(0), torch.LongTensor([cls_])
    @staticmethod
    def collate_fn(data):
        param, cls_ = map(list, zip(*data))
        param = torch.cat(param)
        cls_ = torch.cat(cls_)
        return param, cls_


class MemmapLatentDataset(Dataset):
    """Fast drop-in for CustomDataset backed by a single uncompressed memmap
    (see repack_latents.py). Per-sample load is a sub-ms mmap slice instead of a
    ~52ms zip decompress, so training becomes compute-bound. mmap is opened lazily
    per worker (never in the parent) so it survives DataLoader fork cleanly."""
    def __init__(self, path):
        self.lat_path = os.path.join(path, "latents.npy")
        self.lab_path = os.path.join(path, "labels.npy")
        self._len = int(np.load(self.lab_path, mmap_mode="r").shape[0])
        self.lat = None
        self.lab = None

    def _ensure(self):
        if self.lat is None:
            self.lat = np.load(self.lat_path, mmap_mode="r")
            self.lab = np.load(self.lab_path, mmap_mode="r")

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        self._ensure()
        x = torch.from_numpy(np.ascontiguousarray(self.lat[index]))   # (8,32,32) fp32
        return x, torch.LongTensor([int(self.lab[index])])

    @staticmethod
    def collate_fn(data):
        param, cls_ = map(list, zip(*data))
        return torch.stack(param), torch.cat(cls_)