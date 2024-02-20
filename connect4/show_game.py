import numpy as np 
from connect4.board_c4 import BoardC4
import time 


if __name__ == "__main__": 
    
    file = "data2/e41.npy"
    move_arr = np.load(file)

    game_board = BoardC4.from_start()
    print(game_board)

    for move in move_arr: 

        time.sleep(0.5)

        game_board = game_board.move_from_int(move)

        print(game_board)

    terminal_eval = game_board.terminal_eval() 
    if terminal_eval == 0: 
        print("DRAW")
    elif terminal_eval == -1: 
        print("BLACK WINS")
    else: 
        print("WHITE WINS")