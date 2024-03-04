from common.training_player import TrainingPlayer
from common.training_util import save_idxs

# TRAINING LOOP FILE

# FOR GOOGLE COLAB RUNNING
import sys

from connect4.c4net import C4Net

# sys.path.append("/content/drive/My Drive/bot")
# sys.path.append("/content/drive/My Drive/bot/connect4")
# sys.path.append("/content/drive/My Drive/bot/common")
sys.path.append("/home/jovyan/work/MANTIS")
sys.path.append("C:\\Users\\xiayi\\Desktop\\1. Duke University Classes\\MANTIS")

from common.mcts import mcts
from common.pmcts import Parallel_MCTS

import torch
import torch.nn as nn
import torch.optim as optim

from connect4.board_c4 import BoardC4, print_board
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import numpy as np

import os

import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process, Value

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    player: TrainingPlayer,
    MAX_ITERATIONS,
    EPOCHS_PER_ITERATION,
    NUM_GENERATED,
    BATCH_SIZE,
    GAMES_TO_EVAL,
    START_ITERATION,
    old_exists,
    SAVE_DIR,
):
    # MAX_ITERATIONS = 1000
    # EPOCHS_PER_ITERATION = 50
    # NUM_GENERATED = 200
    # BATCH_SIZE = 15
    # GAMES_TO_EVAL = 30
    # START_ITERATION = 69
    # old_exists = True
    # SAVE_DIR = "data7"

    # MAX_ITERATIONS = 1
    # EPOCHS_PER_ITERATION = 1
    # NUM_GENERATED = 6
    # BATCH_SIZE = 1
    # GAMES_TO_EVAL = 6
    # START_ITERATION = 0
    # old_exists = False

    for i in range(START_ITERATION, MAX_ITERATIONS):
        net, dataset, idxs = player.generate_self_games(NUM_GENERATED)

        save_idxs(SAVE_DIR, idxs, f"sp{i}")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        o_criterion = nn.MSELoss().to(device)
        p_criterion = nn.CrossEntropyLoss().to(device)

        optimizer = optim.Adam(
            net.parameters(), lr=0.002, betas=(0.5, 0.999), weight_decay=1e-5
        )
        for _ in tqdm(range(EPOCHS_PER_ITERATION), desc="training..."):
            net.train()
            for _, (bt, pi, z) in enumerate(dataloader):
                bt = bt.to(device)
                pi = pi.to(device)
                z = z.to(device)
                p, v = net(bt)
                loss = p_criterion(p, pi) + o_criterion(z, v)
                net.zero_grad()
                loss.backward()
                optimizer.step()

        old_net = C4Net().to(device)
        if old_exists:
            old_net.load_state_dict(torch.load("old.pt"))

        fp = os.path.join(SAVE_DIR, f"net{i}.pt")
        torch.save(net.state_dict(), fp)

        old_net.eval()
        net.eval()

        score, res = player.generate_eval_games(GAMES_TO_EVAL, i, net, old_net, fp)

        print(f"Iteration {i} has score {score}: " + "-" * 50)
        print(f"with result w: {res[0]}, d: {res[1]}, l: {res[2]}")
        if score > 0:
            print("...Saving...")
            torch.save(net.state_dict(), "old.pt")
            old_exists = True
