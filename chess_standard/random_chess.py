import torch
from chess_standard.board_chess_pypi import BoardPypiChess
from chess_standard.chessnet import ChessNet
from common.mcts import mcts
from common.player import BlankPlayer
import numpy as np

class RandomChess(BlankPlayer):
    
    def move_and_get_index(self, board: BoardPypiChess):
        legals = board.legal_moves()
        true_indices = np.where(legals)[0]
        index = np.random.choice(true_indices)
        return index