# TRAINING LOOP FILE

# FOR GOOGLE COLAB RUNNING
import sys

# sys.path.append("/content/drive/My Drive/bot")
# sys.path.append("/content/drive/My Drive/bot/connect4")
# sys.path.append("/content/drive/My Drive/bot/common")

sys.path.append("C:\\Users\\xiayi\\Desktop\\1. Duke University Classes\\MANTIS")

from c4net import C4Net
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

device = "cuda" if torch.cuda.is_available() else "cpu"


class C4Dataset(Dataset):
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


def play_a_game(net0, net1, mcts_iter, track=True, self_play=False):
    board = BoardC4.from_start()
    turn = 0
    boards = []
    pis = []
    idxs = []
    current_tree = None
    while board.terminal_eval() == 2:
        if turn == 0:
            move, pi, idx, current_tree = mcts(
                board, net0, runs=mcts_iter, head_node=current_tree if self_play else None
            )
        else:
            move, pi, idx, current_tree = mcts(
                board, net1, runs=mcts_iter, head_node=current_tree if self_play else None
            )
        if track:
            boards.append(board)
            pis.append(pi)
        board = move
        idxs.append(idx)
        turn = 1 if turn == 0 else 0
    result = board.terminal_eval()
    if track:
        training_data = []
        for i in range(len(boards)):
            reward = result if i % 2 == 0 else -result
            training_data.append((boards[i], pis[i], [float(reward)]))
        return training_data, idxs
    return result, idxs


def self_play(generate: int, first: bool, mcts_iter: int):
    """
    Returns net, dataset, idxs of a sample game
    """
    net = C4Net().to(device)
    if not first:
        net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
    net.eval()
    all_data = []
    for _ in tqdm(range(generate), desc="self play..."):
        data, idxs = play_a_game(net, net, mcts_iter, self_play=True)
        all_data.extend(data)
    return net, C4Dataset(all_data), idxs

def self_play_parallel(generate: int, first: bool, mcts_iter: int):
    """
    Returns net, dataset, idxs of a sample game
    """
    net = C4Net().to(device)
    if not first:
        net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
    net.eval()
    all_data, idxs = play_games_in_parallel(generate, net, net, mcts_iter, self_play=True)
    return net, C4Dataset(all_data), idxs

def play_games_in_parallel(num, net0, net1, mcts_iter, self_play=False):
    games = [BoardC4.from_start() for _ in range(num)] 
    results = [2 for _ in range(num)] 
    turn = 0

    boards = [[] for _ in range(num)] # boards[i] is the list of moves from game i
    pis = [[] for _ in range(num)]
    idxs = [[] for _ in range(num)]
    current_trees = [None for _ in range(num)]
    pbar = tqdm(desc="SP - Moves Played", total=100)
    while 2 in results:

        for i in range(num):
            if results[i] != 2:
                games[i] = BoardC4.from_start()
                current_trees[i] = None
        
        mcts = Parallel_MCTS(num, net0 if turn == 0 else net1, mcts_iter, games, current_trees)
        moves, mpis, midxs, mcurrent_trees = mcts.play()
        turn = 1 if turn == 0 else 0
        games = moves
        for i, (move, pi, idx, tree) in enumerate(zip(moves, mpis, midxs, mcurrent_trees)):
            if results[i] == 2:
                boards[i].append(move)
                pis[i].append(pi)
                idxs[i].append(idx)
                current_trees[i] = tree
                results[i] = games[i].terminal_eval()
        pbar.update(1)
    
    t_datas = [[] for _ in range(num)]
    for i, (result, seq_boards, seq_pis) in enumerate(zip(results, boards, pis)):
        for j, (board, pi) in enumerate(zip(seq_boards, seq_pis)):
            reward = result if j % 2 == 0 else -result
            t_datas[i].append((board, pi, [float(reward)]))
    
    training_data = []
    [training_data.extend(t_data) for t_data in t_datas]
    
    ret_idxs = idxs[0]

    return training_data, ret_idxs

def save_idxs(idxs, title="generic.npy"):
    fp = os.path.join(SAVE_DIR, title)
    np.save(fp, np.array(idxs))


SAVE_DIR = "data5"


def training_loop():
    MAX_ITERATIONS = 500
    EPOCHS_PER_ITERATION = 50
    NUM_GENERATED = 300
    BATCH_SIZE = 15
    GAMES_TO_EVAL = 9
    MCTS_ITER = 500
    old_exists = False

    # MAX_ITERATIONS = 1
    # EPOCHS_PER_ITERATION = 1
    # NUM_GENERATED = 1
    # BATCH_SIZE = 1
    # GAMES_TO_EVAL = 1
    # MCTS_ITER = 50
    # old_exists = False

    for i in range(MAX_ITERATIONS):
        net, dataset, idxs = self_play_parallel(NUM_GENERATED, not old_exists, MCTS_ITER)
        save_idxs(idxs, f"sp{i}")
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

        old_net.eval()
        net.eval()

        score = 0
        res = [0] * 3
        for j in tqdm(range(GAMES_TO_EVAL), desc="evaluating..."):
            net0 = net
            net1 = old_net
            win = 1
            if j % 2 == 1:
                net0 = old_net
                net1 = net
                win = -1

            result, idxs = play_a_game(net0, net1, MCTS_ITER, track=False)
            score += result * win

            if win == 1:
                if result == 1:
                    res[0] += 1
                if result == 0:
                    res[1] += 1
                if result == -1:
                    res[2] += 1
            if win == -1:
                if result == 1:
                    res[2] += 1
                if result == 0:
                    res[1] += 1
                if result == -1:
                    res[0] += 1

        save_idxs(idxs, f"e{i}")

        print(f"Iteration {i} has score {score}: " + "-" * 50)
        print(f"with result w: {res[0]}, d: {res[1]}, l: {res[2]}")
        if score > 0:
            print("...Saving...")
            torch.save(net.state_dict(), "old.pt")
            old_exists = True

        fp = os.path.join(SAVE_DIR, f"net{i}.pt")
        torch.save(net.state_dict(), fp)


if __name__ == "__main__":
    training_loop()

    # net = C4Net()
    # net.load_state_dict(torch.load("old.pt", map_location='cpu'))

    # # play_games_in_parallel(10, net, net, 50)
    # # play_a_game(net, net, 50)
    # net, dataset, idx = self_play_parallel(10, False, 50)
    # print(dataset)
    # print(idx)

    # print(dataset.raw_data)