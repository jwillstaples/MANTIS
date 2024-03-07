import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json

BENCHMARK_FILE = "BENCH.txt"
BENCHMARK_DATA = "BENCH.json"


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


def max_key(data):
    ls = list([int(k) for k in data.keys()])
    return str(max(ls))


def create_benchmark_file():
    data = {}

    with open(BENCHMARK_DATA, "w") as file:
        json.dump(data, file, indent=4)


def create_new_benchmark_run():
    with open(BENCHMARK_DATA, "r") as file:
        data = json.load(file)

    if len(data) == 0:
        data[0] = {}
    else:
        data[int(max_key(data)) + 1] = {}

    with open(BENCHMARK_DATA, "w") as file:
        json.dump(data, file, indent=4)


def save_recent_key(key, value):
    with open(BENCHMARK_DATA, "r") as file:
        data = json.load(file)

    data[max_key(data)][key] = value

    with open(BENCHMARK_DATA, "w") as file:
        json.dump(data, file, indent=4)


def append_to_recent_key(key, value):
    with open(BENCHMARK_DATA, "r") as file:
        data = json.load(file)
    if key not in data[max_key(data)]:
        data[max_key(data)][key] = []

    data[max_key(data)][key].append(value)

    with open(BENCHMARK_DATA, "w") as file:
        json.dump(data, file, indent=4)
