import numpy as np
from typing import List
import abc


class BlankBoard:
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_legal_moves(self) -> List["BlankBoard"]:
        """Returns board instances representing each legal move"""
        pass

    @abc.abstractmethod
    def terminal_eval(self) -> int:
        """
        Returns dumb evaluation (win, loss, etc)
        -1 : player two wins
        0 : draw
        1 : player one wins
        2 : unterminated
        """
        pass

    @abc.abstractmethod
    def move_from_int(self, num: int) -> "BlankBoard": 
        """
        Returns a board instance based on the integer position 
        of the move in the p-vector
        """
        pass
