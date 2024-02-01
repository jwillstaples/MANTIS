import numpy as np
from connect4.board_c4 import BoardC4
import random
from connect4.player_c4 import PlayerC4
from connect4.oracle_c4 import OracleC4
from connect4.human_bot_game import print_board, take_human
import time


if __name__ == "__main__":
    game_board = BoardC4.from_start()
    result = 2

    # bot = PlayerC4()
    bot = OracleC4(depth=4)

    print_board(game_board)

    while result == 2:
        if game_board.red_move:
            game_board = bot.move(game_board)

        else:
            game_board = take_human(game_board)

        print_board(game_board)
        result = game_board.terminal_eval()

    if result == 1:
        print("x wins")
    elif result == -1:
        print("o wins")
    else:
        print("draw")