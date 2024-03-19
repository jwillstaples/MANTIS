from common.parallel_player import ParallelPlayer
from common.serial_player import SerialPlayer
from common.training_loop import train
from connect4.board_c4 import BoardC4
from connect4.c4net import C4Net
import torch.multiprocessing as mp

import sys

sys.path.append("/home/jovyan/work/MANTIS")
sys.path.append("C:\\Users\\xiayi\\Desktop\\1. Duke University Classes\\MANTIS")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    MAX_ITERATIONS = 1000
    EPOCHS_PER_ITERATION = 50
    NUM_GENERATED = 200
    BATCH_SIZE = 15
    GAMES_TO_EVAL = 30
    MCTS_ITER = 500
    START_ITERATION = 69
    old_exists = False

    # MAX_ITERATIONS = 1
    # EPOCHS_PER_ITERATION = 1
    # NUM_GENERATED = 6
    # BATCH_SIZE = 1
    # GAMES_TO_EVAL = 6
    # MCTS_ITER = 50
    # START_ITERATION = 0
    # old_exists = False

    SAVE_DIR = "data7"
    TEMP_NAME = "old.pt"
    multicore = 2
    Net = C4Net
    Board = BoardC4

    # player = SerialPlayer(MCTS_ITER, old_exists, SAVE_DIR, TEMP_NAME, multicore, Net, Board)
    player = ParallelPlayer(
        MCTS_ITER, old_exists, SAVE_DIR, TEMP_NAME, multicore, Net, Board
    )

    train(
        player,
        Net,
        MAX_ITERATIONS,
        EPOCHS_PER_ITERATION,
        NUM_GENERATED,
        BATCH_SIZE,
        GAMES_TO_EVAL,
        START_ITERATION,
        old_exists,
        SAVE_DIR,
        TEMP_NAME,
    )
