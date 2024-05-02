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

from connect4.view import C4Game
# from chess_standard.view import ChessGame
from chess_standard.view_game import ChessGame


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

def bot_v_human_Chess():
    # IMPLEMENT
    print("Begun")
    game = ChessGame(
        chess_bot_fp="chessbot.pt",
        chess_bot_runs=10,
        board_init_fen="",
        player_color=True
    )
    arcade.run()
 

if __name__ == "__main__":
    # bot_v_human_C4()
    # bot_v_bot()
    # bot_v_human_Chess()
    HUMAN_V_HUMAN = 0
    HUMAN_V_BOT = 1
    BOT_V_BOT = 2

    white_bot = MantisChess(fp="chessbot.pt", random=False, runs=10)
    black_bot = MantisChess(fp="chessbot.pt", random=True, runs=10)

    # colors: White = True, Black = False
    game = ChessGame(
        mode=BOT_V_BOT,
        board_init_fen="",
        
        # load bots, not necessarily used unless mode specifies
        white_bot=white_bot,
        black_bot=black_bot,

        # only for human v bot mode
        player_clr=True,

        # only for bot v bot mode
        bot_clr=True, # sets OUR bot's color
        random_eval_iters=10 # sets # of iters to play bot v bot
    )
    arcade.run()