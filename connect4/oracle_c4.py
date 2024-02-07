import numpy as np
from common.player import BlankPlayer
from connect4.board_c4 import BoardC4
from typing import Tuple

from scipy.signal import convolve2d


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


class TestNet:

    def forward(self, board: BoardC4):

        p_vec_test = np.array([1, 2, 3, 4, 3, 2, 1]) / 16

        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [
            horizontal_kernel,
            vertical_kernel,
            diag1_kernel,
            diag2_kernel,
        ]

        eval = 0.0

        for kernel in detection_kernels:
            convolution = convolve2d(board.board_matrix, kernel, mode="valid")
            if (convolution == 4).any():
                return (p_vec_test, 1)
            if (convolution == -4).any():
                return (p_vec_test, -1)
            friendly_almost = np.sum(convolution == 3)
            enemy_almost = np.sum(convolution == -3)
            eval += 0.01 * (
                5 * np.sum(convolution == 3)
                + np.sum(convolution == 2)
                - 5 * np.sum(convolution == -3)
                - np.sum(convolution == -2)
            )

        eval += np.sum(board.board_matrix[3]) * 0.1

        return (p_vec_test, eval)
