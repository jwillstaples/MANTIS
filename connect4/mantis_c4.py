from common.player import BlankPlayer
from connect4.board_c4 import BoardC4
from connect4.c4net import C4Net
import torch

from common.mcts import mcts


class MantisC4(BlankPlayer):

    def __init__(self, fp):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = C4Net()
        self.net.load_state_dict(torch.load(fp, map_location=device))

        self.net.eval()

    def move(self, board: BoardC4):
        board, _, _, _ = mcts(board, self.net, runs=1000)
        return board
