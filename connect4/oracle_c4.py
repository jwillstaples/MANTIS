import numpy as np
from common.player import BlankPlayer
from connect4.board_c4 import BoardC4
from typing import Tuple


class OracleC4(BlankPlayer):

    def __init__(self, depth: int):

        self.depth = depth

    def move(self, board: BoardC4) -> BoardC4:

        def evaluate(board: BoardC4, depth: int, which_move: int) -> int:

            term_eval = board.terminal_eval()
            if term_eval != 2:
                return term_eval
            elif depth == 0:
                return 2
            else:
                possible_moves = board.get_legal_moves()
                possible_evals = [
                    evaluate(child, depth - 1, which_move=which_move * -1)
                    for child in possible_moves
                ]

                if which_move in possible_evals:
                    return which_move
                elif 2 in possible_evals:
                    return 2
                elif 0 in possible_evals:
                    return 2
                else:
                    return which_move * -1

        possible_moves = board.get_legal_moves()
        which_move = 1 if board.red_move else -1
        child_evals = [
            (evaluate(child, self.depth, which_move * -1), child)
            for child in possible_moves
        ]

        def sorter(x):
            if x[0] == which_move:
                return 3
            elif x[0] == 2:
                return 2
            elif x[0] == 0:
                return 1
            else:
                return 0

        child_evals = sorted(child_evals, key=sorter)
        eval = print(child_evals[-1][0])
        return child_evals[-1][1]
