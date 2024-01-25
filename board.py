import numpy as np
from typing import List

class BlankBoard: 

    def __init__(self): 

        pass

    def get_legal_moves(self) -> List[BlankBoard]:
        """Returns board instances representing each legal move"""
        return [] 
    
    def terminal_eval(self) -> int: 
        """
        Returns dumb evaluation (win, loss, etc)
        -1 : player two wins
        0 : draw 
        1 : player one wins
        2 : unterminated 
        """
        return 0
    
