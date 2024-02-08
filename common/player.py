import numpy as np
import abc
from common.board import BlankBoard


class BlankPlayer:
    @abc.abstractmethod
    def move(self, board: BlankBoard) -> BlankBoard:
        """
        Takes current board and returns suggested move as another instance of board
        """

        pass
