from abc import abstractmethod
import torch.nn as nn
from typing import Type
from common.board import BlankBoard


class TrainingPlayer:
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
        Inputs:
            Monte Carlo Tree Search Iterations,
            If a file named old.py exists
            Current Saving Directory
            If multiple processors are allowed
            Net Class
            Board Class
        """
        self.mcts_iter = mcts_iter
        self.old_exists = old_exists
        self.save_dir = SAVE_DIR
        self.multicore = multicore
        self.Net = Net
        self.Board = Board

    @abstractmethod
    def generate_self_games(self, num):
        """
        Returns net, GameDataset, idxs of a sample game
        """
        pass

    @abstractmethod
    def generate_eval_games(
        self, num, iteration, candidate_net, old_net, candidate_net_file
    ):
        """
        Saves to directory a sample game

        Returns Score, Result Array ([W, D, L])
        """
        pass
