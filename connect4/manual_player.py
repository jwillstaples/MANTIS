import numpy as np 
from board_c4 import BoardC4
import random

def print_board(board: BoardC4) -> None: 

    for row in board.board_matrix.T[::-1]: 
        print("|", end="")
        for val in row: 
            if val == 1: 
                print("x", end="")
            elif val == -1: 
                print("o", end="")
            else: 
                print(" ", end="")
            print("|", end="")
        print("")
    print("\n")



if __name__ == "__main__": 
    
    game_board = BoardC4.from_start()
    result = 2

    while result == 2: 

        print_board(game_board)
        result = game_board.terminal_eval()

        possible_moves = game_board.get_legal_moves()
        game_board = random.choice(possible_moves)

    if result == 1: 
        print("x wins")
    elif result == -1: 
        print("o wins")
    else: 
        print("draw")

