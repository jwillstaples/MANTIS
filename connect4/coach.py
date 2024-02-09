# TRAINING LOOP FILE

# FOR GOOGLE COLAB RUNNING
import sys

sys.path.append("/content/drive/My Drive/bot")
sys.path.append("/content/drive/My Drive/bot/connect4")
sys.path.append("/content/drive/My Drive/bot/common")

from c4net import C4Net
from mcts import mcts

import torch
import torch.nn as nn
import torch.optim as optim

from connect4.board_c4 import BoardC4, print_board
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

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


def play_a_game(net0, net1, track = True):
    board = BoardC4.from_start()
    turn = 0
    boards = []
    pis = []
    while board.terminal_eval() == 2:
        if turn == 0:
            move, pi = mcts(board, net0)
        else:
            move, pi = mcts(board, net1)
        if track:
            boards.append(board)
            pis.append(pi)
        board = move
    result = board.terminal_eval()
    if track:
        training_data = []
        for i in range(len(boards)):
            reward = result if i % 2 == 0 else -result
            training_data.append((boards[i], pis[i], [float(reward)]))
        return training_data
    return result

def self_play(generate, first: bool) -> Dataset:
    net = C4Net().to(device)
    if not first:
        net.load_state_dict(torch.load("old.pt"))
    all_data = []
    for _ in tqdm(range(generate), desc='self play...'):
        data = play_a_game(net, net)
        all_data.extend(data)
    return net, C4Dataset(all_data)

def training_loop():
    MAX_ITERATIONS = 20
    EPOCHS_PER_ITERATION = 10
    NUM_GENERATED = 10
    BATCH_SIZE = 10
    GAMES_TO_EVAL = 10

    for i in range(MAX_ITERATIONS):
        net, dataset = self_play(NUM_GENERATED, i == 0)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        o_criterion = nn.MSELoss().to(device)
        p_criterion = nn.CrossEntropyLoss().to(device)

        optimizer = optim.Adam(net.parameters(), lr=0.002, betas=(0.5, 0.999), weight_decay=1e-5)
        for _ in tqdm(range(EPOCHS_PER_ITERATION), desc='training...'):
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
        old_net.load_state_dict(torch.load("old.pt"))

        old_net.eval()
        net.eval()

        score = 0
        for j in tqdm(range(GAMES_TO_EVAL), desc='evaluating...'):
            net0 = net
            net1 = old_net
            win = 1
            if j % 2 == 1:
                net0 = old_net
                net1 = net
                win = -1

            result = play_a_game(net0, net1, track = False)
            score += result * win

        print(f"Iteration {i} has score {score}: " + "-"*50)
        if score > 0:
            print("...Saving...")
            torch.save(net.state_dict(), "old.pt")

if __name__ == "__main__":
    training_loop()