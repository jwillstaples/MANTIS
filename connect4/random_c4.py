from common.player import BlankPlayer
from connect4.board_c4 import BoardC4
from connect4.c4net import C4Net
import torch
import numpy as np
from common.mcts import mcts


class RandomC4(BlankPlayer):

    def move_and_get_index(self, board: BoardC4):
        legals = board.legal_moves()
        true_indices = np.where(legals)[0]
        index = np.random.choice(true_indices)
        return index
