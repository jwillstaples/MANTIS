import numpy as np
from connect4.board_c4 import BoardC4
import random
from connect4.player_c4 import PlayerC4
from connect4.oracle_c4 import OracleC4
import time


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
    for i in range(7):
        print(f" {i}", end="")
    print("\n")


def take_human(board: BoardC4) -> BoardC4:
    legal_placements = board.open_cols()
    print("You may place in the following columns: ")
    for i in legal_placements:
        print(f"{i}, ", end="")
    print("")
    choice = int(input("Column to play  >>> "))

    move = (choice, board.bottom_available(choice))

    new_board = board.make_move(move)

    return new_board

