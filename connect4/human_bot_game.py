import numpy as np
from board_c4 import BoardC4
import random
from player_c4 import PlayerC4
from oracle_c4 import OracleC4
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


if __name__ == "__main__":
    game_board = BoardC4.from_start()
    result = 2

    
    bot1 = OracleC4(depth=3)
    bot2 = OracleC4(depth=4)

    print_board(game_board)

    while result == 2:
        if game_board.red_move:
            game_board = bot1.move(game_board)

        else:
            # game_board = take_human(game_board)
            game_board = bot2.move(game_board)

        print_board(game_board)
        result = game_board.terminal_eval()

    if result == 1:
        print("x wins")
    elif result == -1:
        print("o wins")
    else:
        print("draw")
