import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        data = self.raw_data[idx]

        board = data[0].to_tensor()
        pi = data[1]
        z = data[2]

        return board, torch.tensor(pi), torch.tensor(z)


def save_idxs(SAVE_DIR, idxs, title="generic.npy"):
    fp = os.path.join(SAVE_DIR, title)
    np.save(fp, np.array(idxs))
