import torch
from chess_standard.board_chess_pypi import BoardPypiChess
from chess_standard.chessnet import ChessNet
from common.mcts import mcts
from common.player import BlankPlayer


class MantisChess(BlankPlayer):
    def __init__(self, fp, random=False, runs=500):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.runs = runs
        self.net = ChessNet().to(device)
        if not random:
            self.net.load_state_dict(torch.load(fp, map_location=device))

        self.net.eval()

    def move(self, board: BoardPypiChess):
        board, _, _, _ = mcts(board, self.net, runs=self.runs)
        return board
    
    def move_and_get_index(self, board: BoardPypiChess):
        _, _, index, _ = mcts(board, self.net, runs=self.runs)
        return index