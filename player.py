import numpy as np
import abc
from board import BlankBoard


class BlankPlayer:

    def __init__(self):
        pass

    @abc.abstractmethod
    def move(self, board: BlankBoard) -> BlankBoard:
        """
        Takes current board and returns suggested move as another instance of board
        """

        pass
