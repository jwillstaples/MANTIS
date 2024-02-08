import numpy as np

from common.player import BlankPlayer
from connect4.board_c4 import BoardC4

from connect4.mcts import mcts
from connect4.c4net import C4Net

import random


class PlayerC4:
    def __init__(self, nnet: C4Net):
        self.nnet = nnet

    def move(self, board: BoardC4) -> BoardC4:
        new_board = mcts(board, self.nnet)

        return new_board
