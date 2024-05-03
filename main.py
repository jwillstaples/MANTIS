import numpy as np
from chess_standard.board_chess_pypi import BoardPypiChess
from chess_standard.mantis_chess import MantisChess
from connect4.board_c4 import BoardC4
import random
from connect4.mantis_c4 import MantisC4
from connect4.minimax_c4 import MinimaxC4
from connect4.c4net import C4Net
from connect4.minimax_c4 import TestNet
import time
import arcade

from connect4.random_c4 import RandomC4
from connect4.view import C4Game

def initializec4board():
    moves = np.load("minimaxgamesample.npy")
    moves = moves[:-2]
    print(moves)
    g = BoardC4.from_start()
    for m in moves:
        g = g.move_from_int(m)
    g = g.move_from_int(3)
    print(g)
    bot = MantisC4("c4_58iter_1000mcts.pt", runs=1500)
    print(bot.move_and_get_index(g))
    print(moves)
def bot_v_bot():
    game_board = BoardC4.from_start()
    result = 2

    # bot = MinimaxC4(depth=4)
    bot2 = MinimaxC4(depth=4)

    bot = MantisC4("c4_58iter_1000mcts.pt", runs=1500)
    # bot2 = MantisC4("data6/net4.pt", True)
    # bot2 = RandomC4()

    print(game_board)
    indices = []
    while result == 2:
        if game_board.red_move:
            game_board = bot.move(game_board)
            # index = bot.move_and_get_index(game_board)
        else:
            game_board = bot2.move(game_board)
            # index = bot2.move_and_get_index(game_board)
        # indices.append(index)
        # game_board = game_board.move_from_int(index)
        print(game_board)
        result = game_board.terminal_eval()

    if result == 1:
        print("x wins")
    elif result == -1:
        print("o wins")
    else:
        print("draw")
    print(indices)

def bot_v_human_C4():
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 700
    game = C4Game(SCREEN_WIDTH, SCREEN_HEIGHT, "c4_500mcts_80iter.pt")
    game.setup()
    arcade.run()

def bot_v_human_Chess():
    # IMPLEMENT
    print("Begun")
    bot = MantisChess("chessbot.pt", runs=10)

    starting_board = BoardPypiChess()
    move1 = bot.move(starting_board)
    print(move1.board)


if __name__ == "__main__":
    # bot_v_human_C4()
    # bot_v_bot()
    # bot_v_human_Chess()
    initializec4board()