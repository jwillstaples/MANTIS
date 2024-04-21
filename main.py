import numpy as np
from connect4.board_c4 import BoardC4
import random
from connect4.mantis_c4 import MantisC4
from connect4.minimax_c4 import MinimaxC4
from connect4.c4net import C4Net
from connect4.minimax_c4 import TestNet
import time
import arcade

from connect4.view import C4Game


def bot_v_bot():
    game_board = BoardC4.from_start()
    result = 2

    # bot = MinimaxC4(depth=4)
    # bot2 = MinimaxC4(depth=4)

    bot = MantisC4("old.pt", runs=1000)
    bot2 = MantisC4("data6/net4.pt", True)

    print(game_board)

    while result == 2:
        if game_board.red_move:
            game_board = bot.move(game_board)

        else:
            # game_board = take_human(game_board)
            game_board = bot2.move(game_board)

        print(game_board)
        result = game_board.terminal_eval()

    if result == 1:
        print("x wins")
    elif result == -1:
        print("o wins")
    else:
        print("draw")


def bot_v_human_C4():
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 700
    game = C4Game(SCREEN_WIDTH, SCREEN_HEIGHT, "c4_500mcts_80iter.pt")
    game.setup()
    arcade.run()


if __name__ == "__main__":
    bot_v_human_C4()
    # bot_v_bot()
