import numpy as np

from common.player import BlankPlayer
from connect4.board_c4 import BoardC4

import random


class PlayerC4:

    def __init__(self):
        pass

    def move(self, board: BoardC4) -> BoardC4:

        possible_moves = board.get_legal_moves()

        return random.choice(possible_moves)
    
    def mcts(board: BoardC4, runs: int): 
        pass 

