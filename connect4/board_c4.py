from typing import List, Tuple 
import numpy as np 
# from board import BlankBoard
from scipy.signal import convolve2d

class BoardC4(): 

    def __init__(self, board_matrix: np.ndarray, red_move: bool): 
        """
        board_matrix: 6x7 numpy array holding integer values. 1: red, 0: empty, -1: yellow 
        red_move: boolean holding whether it's red's move. We assume red moves first.
        """

        self.board_matrix = board_matrix
        self.red_move = red_move

    def get_legal_moves(self) -> List:
        """ return List of boards representing legal moves"""
        
        legal_placements = []
        for (col, row), val in np.ndenumerate(self.board_matrix): 
            if val == 0: 
                if self.board_matrix[col][row-1] != 0 or row == 0: 
                    legal_placements.append((col, row))

        return [self.make_move(move) for move in legal_placements]
    
    def open_cols(self) -> List: 
        """ Returns columns with an open space"""

        open_cols = []
        for i, col in enumerate(self.board_matrix): 
            if np.sum(np.abs(col)) < 7: 
                open_cols.append(i)

        return open_cols
    
    def bottom_available(self, col: int): 
        """ returns bottom available slot in a given column """

        for i, val in enumerate(self.board_matrix[col]): 
            if val == 0: 
                return i
            
        raise ValueError(f"No open spaces in column {col}")

    def terminal_eval(self) -> int:
        
        horizontal_kernel = np.array([[ 1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

        for kernel in detection_kernels: 
            convolution = convolve2d(self.board_matrix, kernel, mode="valid")
            if (convolution == 4).any(): 
                return 1 
            if (convolution == -4).any(): 
                return -1
        
        if not (self.board_matrix == 0).any(): 
            return 0 
        
        return 2

    def make_move(self, move): 

        new_board = self.board_matrix.copy()
        new_board[move[0]][move[1]] = 1 if self.red_move else -1

        return BoardC4(new_board, not self.red_move)


    @classmethod
    def from_start(cls): 
        return cls(np.zeros((6, 7), dtype=int), True)
