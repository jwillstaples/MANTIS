from common.parallel_player import ParallelPlayer
from common.training_loop import train
from common.training_util import (
    BENCHMARK_FILE,
    create_benchmark_file,
    create_new_benchmark_run,
    save_recent_key,
)
from connect4.board_c4 import BoardC4
from connect4.c4net import C4Net
import torch.multiprocessing as mp

import sys

import time

sys.path.append("/home/jovyan/work/MANTIS")
sys.path.append("C:\\Users\\xiayi\\Desktop\\1. Duke University Classes\\MANTIS")


if __name__ == "__main__":
    print("Start")
    mp.set_start_method("spawn")

    MAX_ITERATIONS = 1
    EPOCHS_PER_ITERATION = 0
    generateds = [10, 50, 100, 500, 1000]
    BATCH_SIZE = 15
    GAMES_TO_EVAL = 0
    MCTS_ITER = 10
    START_ITERATION = 0
    old_exists = False

    SAVE_DIR = "data_bench"
    TEMP_NAME = "bench.pt"
    cores = [1, 2, 4, 8]
    Net = C4Net
    Board = BoardC4
    
    with open(BENCHMARK_FILE, "w") as f:
        f.write(f"Start of Benchmarking\n")

    create_benchmark_file()

    for g in generateds:
        for core in cores:
            adj_g = g // core

            NUM_GENERATED = adj_g
            multicore = core

            player = ParallelPlayer(
                MCTS_ITER,
                old_exists,
                SAVE_DIR,
                TEMP_NAME,
                multicore,
                Net,
                Board,
                benchmark=True,
            )

            create_new_benchmark_run()

            with open(BENCHMARK_FILE, "a") as f:
                f.write(
                    f"Generated Total: {adj_g * core}, Generated per Core: {NUM_GENERATED}, Cores: {core}\n"
                )
                save_recent_key("gen_total", adj_g * core)
                save_recent_key("gen_per_core", NUM_GENERATED)
                save_recent_key("cores", core)

            st = time.time()
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

            et = time.time() - st

            with open(BENCHMARK_FILE, "a") as f:
                s = "-" * 100
                f.write(
                    f"Total Time: {et}, Seconds per game: {et / (adj_g * core)}, Games per second: {(adj_g * core) / et} {s}\n"
                )
                save_recent_key("total_time", et)
                save_recent_key("sec_per_game", et / (adj_g * core))
                save_recent_key("game_per_sec", (adj_g * core) / et)


# 500 2
