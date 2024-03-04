# TRAINING LOOP FILE

# FOR GOOGLE COLAB RUNNING
import sys

# sys.path.append("/content/drive/My Drive/bot")
# sys.path.append("/content/drive/My Drive/bot/connect4")
# sys.path.append("/content/drive/My Drive/bot/common")
sys.path.append("/home/jovyan/work/MANTIS")
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

import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process, Value

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
                board,
                net0,
                runs=mcts_iter,
                head_node=current_tree if self_play else None,
            )
        else:
            move, pi, idx, current_tree = mcts(
                board,
                net1,
                runs=mcts_iter,
                head_node=current_tree if self_play else None,
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


def parallel_parallel(generate: int, first: bool, mcts_iter: int):
    net = C4Net().to(device)
    if not first:
        net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
    net.eval()

    queue = Queue()
    stop_sig = Value("i", 0)

    # process = Process(
    #     target=pp_wrapper, args=(stop_sig, queue, generate, net, net, mcts_iter, True)
    # )
    process = Process(
        target=pp_wrapper2, args=(stop_sig, queue, generate, first, mcts_iter, True)
    )

    process.start()

    all_data, idxs = play_games_in_parallel(
        generate, net, net, mcts_iter, self_play=True, telemetry=True
    )

    while (
        stop_sig.value == 0
    ):  # Unknown where the dead lock is so using value to auto-reliquish
        continue

    process.join(timeout=1)
    # process.join()
    all_data2 = queue.get()
    all_data.extend(all_data2)
    return net, C4Dataset(all_data), idxs


def pp_wrapper2(stop_sig, queue, num, first, mcts_iter, self_play):
    # may not be able to pass the network into the new function because it can't be pickled

    net = C4Net().to(device)
    if not first:
        net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
    net.eval()

    print("Parallel Start")
    all_data, _ = play_games_in_parallel(num, net, net, mcts_iter, self_play)
    queue.put(all_data)
    stop_sig.value = 1


def pp_wrapper(stop_sig, queue, num, net0, net1, mcts_iter, self_play):
    print("Parallel Start")
    all_data, _ = play_games_in_parallel(num, net0, net1, mcts_iter, self_play)
    queue.put(all_data)
    stop_sig.value = 1


def self_play_parallel(generate: int, first: bool, mcts_iter: int):
    """
    Returns net, dataset, idxs of a sample game
    """
    net = C4Net().to(device)
    if not first:
        net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
    net.eval()
    all_data, idxs = play_games_in_parallel(
        generate, net, net, mcts_iter, self_play=True
    )
    return net, C4Dataset(all_data), idxs


def play_games_in_parallel(
    num, net0, net1, mcts_iter, self_play=False, telemetry=False
):
    games = [BoardC4.from_start() for _ in range(num)]
    results = [2 for _ in range(num)]
    finished = [False for _ in range(num)]
    turn = 0

    boards = [[] for _ in range(num)]  # boards[i] is the list of moves from game i
    pis = [[] for _ in range(num)]
    idxs = [[] for _ in range(num)]
    current_trees = [None for _ in range(num)]
    if telemetry:
        pbar = tqdm(desc="SP - Moves Played", total=100)
    while 2 in results:
        for i in range(num):
            if results[i] != 2:
                games[i] = BoardC4.from_start()
                current_trees[i] = None
                finished[i] = True

        mcts = Parallel_MCTS(
            games=num,
            net=net0 if turn == 0 else net1,
            runs=mcts_iter,
            first_boards=games,
            passed_trees=current_trees,
            finished_games=finished,
        )
        first_boards, moves, mpis, midxs, mcurrent_trees = mcts.play()
        turn = 1 if turn == 0 else 0
        games = moves
        for i, (fb, pi, idx, tree) in enumerate(
            zip(first_boards, mpis, midxs, mcurrent_trees)
        ):
            if results[i] == 2:
                boards[i].append(fb)
                pis[i].append(pi)
                idxs[i].append(idx)
                if self_play:
                    current_trees[i] = tree
                results[i] = games[i].terminal_eval()
        if telemetry:
            pbar.update(1)

    if not self_play:
        return results, idxs[0]

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


SAVE_DIR = "data6"


def training_loop():
    MAX_ITERATIONS = 1000
    EPOCHS_PER_ITERATION = 50
    NUM_GENERATED = 200
    BATCH_SIZE = 15
    GAMES_TO_EVAL = 30
    MCTS_ITER = 500
    START_ITERATION = 69
    old_exists = True

    # MAX_ITERATIONS = 1
    # EPOCHS_PER_ITERATION = 1
    # NUM_GENERATED = 6
    # BATCH_SIZE = 1
    # GAMES_TO_EVAL = 6
    # MCTS_ITER = 50
    # START_ITERATION = 0
    # old_exists = False

    for i in range(69, MAX_ITERATIONS):
        net, dataset, idxs = parallel_parallel(NUM_GENERATED, not old_exists, MCTS_ITER)
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

        fp = os.path.join(SAVE_DIR, f"net{i}.pt")
        torch.save(net.state_dict(), fp)

        old_net.eval()
        net.eval()

        # score, res = serial_evaluation(GAMES_TO_EVAL, MCTS_ITER, i, net, old_net)

        score, res = parallel_parallel_evaluation(
            GAMES_TO_EVAL, MCTS_ITER, i, net, old_net, fp, old_exists
        )

        print(f"Iteration {i} has score {score}: " + "-" * 50)
        print(f"with result w: {res[0]}, d: {res[1]}, l: {res[2]}")
        if score > 0:
            print("...Saving...")
            torch.save(net.state_dict(), "old.pt")
            old_exists = True


def parallel_parallel_evaluation(
    GAMES_TO_EVAL, MCTS_ITER, i, net, old_net, fp, old_exists
):
    score = 0
    res = [0] * 3

    queue = Queue()
    stop_signal = Value("i", 0)
    # process = Process(
    #     target=parallel_parallel_eval_wrapper, args=(queue, stop_signal, GAMES_TO_EVAL, MCTS_ITER, net, old_net)
    # )
    process = Process(
        target=parallel_parallel_eval_wrapper2,
        args=(queue, stop_signal, GAMES_TO_EVAL, MCTS_ITER, fp, old_exists),
    )
    process.start()

    w_results, idxs = play_games_in_parallel(
        GAMES_TO_EVAL // 2, net, old_net, MCTS_ITER, False, telemetry=True
    )
    while stop_signal.value == 0:
        continue
    process.join(timeout=1)
    b_results = queue.get()

    for result in w_results:
        if result == 1:
            res[0] += 1
        if result == 0:
            res[1] += 1
        if result == -1:
            res[2] += 1
        score += result

    for result in b_results:
        if result == -1:
            res[0] += 1
        if result == 0:
            res[1] += 1
        if result == 1:
            res[2] += 1
        score -= result
    save_idxs(idxs, f"e{i}")

    return score, res


def parallel_parallel_eval_wrapper(
    queue, stop_sig, GAMES_TO_EVAL, MCTS_ITER, net, old_net
):
    b_results, _ = play_games_in_parallel(
        GAMES_TO_EVAL // 2, old_net, net, MCTS_ITER, False
    )
    queue.put(b_results)
    stop_sig.value = 1


def parallel_parallel_eval_wrapper2(
    queue, stop_sig, GAMES_TO_EVAL, MCTS_ITER, fp, old_exists
):
    net = C4Net().to(device)
    net.load_state_dict(torch.load(fp, map_location=torch.device(device)))

    old_net = C4Net().to(device)
    if old_exists:
        old_net.load_state_dict(torch.load("old.pt"))

    b_results, _ = play_games_in_parallel(
        GAMES_TO_EVAL // 2, old_net, net, MCTS_ITER, False
    )
    queue.put(b_results)
    stop_sig.value = 1


def serial_evaluation(GAMES_TO_EVAL, MCTS_ITER, i, net, old_net):
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
    return score, res


if __name__ == "__main__":
    mp.set_start_method("spawn")
    training_loop()

    # net = C4Net()
    # net.load_state_dict(torch.load("old.pt", map_location='cpu'))
    # training_data, idxs = play_a_game(net, net, 100, self_play=True)
    # training_data, idxs = play_games_in_parallel(2, net, net, 50, self_play=True, telemetry=True)
#     for d in training_data:
#         board = d[0]
#         p = d[1]
#         reward = d[2]

#         print_board(board)
#         print(p)
#         print(reward)
#         print("-" * 50)
# # play_a_game(net, net, 50)
# net, dataset, idx = self_play_parallel(10, False, 50)
# print(dataset)
# print(idx)

# print(dataset.raw_data)

# parallel_parallel(10, False, 50)
