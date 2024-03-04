import torch
from tqdm import tqdm
from common.board import BlankBoard
from common.mcts import mcts
from common.training_player import TrainingPlayer
from common.training_util import GameDataset, save_idxs
import torch.nn as nn
from typing import Type

device = "cuda" if torch.cuda.is_available() else "cpu"


class SerialPlayer(TrainingPlayer):
    def __init__(
        self,
        mcts_iter: int,
        old_exists: bool,
        SAVE_DIR: str,
        multicore: bool,
        Net: Type[nn.Module],
        Board: Type[BlankBoard],
    ):
        """
        Note: Multicore not used
        """
        super().__init__(mcts_iter, old_exists, SAVE_DIR, multicore, Net, Board)

    def generate_self_games(self, num):
        net = self.Net().to(device)
        if not self.old_exists:
            net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
        net.eval()
        all_data = []
        for _ in tqdm(range(num), desc="self play..."):
            data, idxs = self.play_a_game(net, net, self.mcts_iter, self_play=True)
            all_data.extend(data)
        return net, GameDataset(all_data), idxs

    def generate_eval_games(
        self, num, iteration, candidate_net, old_net, candidate_net_file
    ):
        score = 0
        res = [0] * 3
        for j in tqdm(range(num), desc="evaluating..."):
            net0 = candidate_net
            net1 = old_net
            win = 1
            if j % 2 == 1:
                net0 = old_net
                net1 = candidate_net
                win = -1

            result, idxs = self.play_a_game(net0, net1, track=False)
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

        save_idxs(self.save_dir, idxs, f"e{iteration}")
        return score, res

    def play_a_game(self, net0, net1, track=True, self_play=False):
        board = self.Board.from_start()
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
                    runs=self.mcts_iter,
                    head_node=current_tree if self_play else None,
                )
            else:
                move, pi, idx, current_tree = mcts(
                    board,
                    net1,
                    runs=self.mcts_iter,
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
