import numpy as np
from common.board import BlankBoard
import torch
from typing import List
from tqdm import tqdm
from common.mcts import select, add_dirichlet, get_board, Node, get_output
import threading
import time

from common.training_util import BENCHMARK_FILE, append_to_recent_key

device = "cuda" if torch.cuda.is_available() else "cpu"


class Parallel_MCTS:
    def __init__(
        self,
        games: int,
        net: torch.nn.Module,
        runs: int,
        first_boards: List[BlankBoard],
        passed_trees: List[Node],
        finished_games: List[bool],
        benchmark: bool,
        telemetry: bool,
    ):
        self.benchmark = benchmark
        self.telemetry = telemetry
        self.first_boards = first_boards
        self.net = net
        self.runs = np.array([runs] * games)
        self.board_shape = first_boards[0].to_tensor().shape
        p_vecs, _ = self.parallel_pass(first_boards)
        self.trees = [None] * games
        for i in range(games):
            if passed_trees[i] is None:
                node = Node()
                node.board = first_boards[i]
                node.make_children(add_dirichlet(np.array(p_vecs[i])))
                node.n = 0
                self.trees[i] = node
            else:
                node = passed_trees[i]
                node.parent = None
                node.child_index = None
                node.p = None

                self.runs[i] = self.runs[i] - node.n
                self.trees[i] = node
        for i, fin in enumerate(finished_games):
            if fin:
                self.runs[i] = 1

        self.games = games
        self.max_runs = max(self.runs)

    def to_run(self, i):
        return self.runs[i] > 0

    def parallel_pass(self, boards: List[BlankBoard]):
        """
        returns 2 vectors, p_vecs and evals

        outputs random if the board is None, meaning that move calculation is out of runs
        if board is in terminal state, outputs None in that entry of p_vecs
        """
        tensors = torch.stack(
            [
                (
                    board.to_tensor()
                    if board is not None
                    else torch.randn(self.board_shape)
                )
                for board in boards
            ],
            dim=0,
        ).to(device)
        p_vecs, evals = self.net(tensors)
        p_vecs = p_vecs.detach().cpu().numpy()
        evals = evals.detach().cpu().numpy()[:, 0]
        player_perspective_evals = self.parallel_player_perspective_evals(boards)
        for i, board in enumerate(boards):
            if board is not None and player_perspective_evals[i] != 2:
                p_vecs[i] = None
                evals[i] = player_perspective_evals[i]
        return p_vecs, evals
    
    def parallel_player_perspective_evals(self, boards):
        results = [3] * len(boards)
        
        def player_perspective_eval_wrapper(board, index):
            try:
                results[index] = board.player_perspective_eval()
            except Exception as e:
                results[index] = 2
                print("Error from the following board")
                print(e)
                print(board.board)

        threads = []

        for i, board in enumerate(boards):
            if board is None:
                results[i] = 2
            elif board.terminal_slow():
                thread = threading.Thread(target=player_perspective_eval_wrapper, args=(board, i))
                threads.append(thread)
                thread.start()
            else:
                results[i] = board.player_perspective_eval()
                
        while 3 in results:
            continue

        for thread in threads:
            thread.join(timeout=1)
        return results


    def play(self):
        """
        gets all best moves

        returns List[next board], List[normalized values], List[index of move], List[next tree]
        """

        st = time.time()
        ppt = 0
        boards = [None] * self.games
        vals = [None] * self.games
        indices = [-1] * self.games
        next_trees = [None] * self.games
        # for _ in tqdm(range(self.max_runs)):
        for _ in range(self.max_runs):
            sim_nodes = [None] * self.games
            sim_boards = [None] * self.games
            for i, head in enumerate(self.trees):
                if self.to_run(i):
                    n = select(head)
                    sim_nodes[i] = n
                    sim_boards[i] = get_board(n)

            ppts = time.time()
            p_vecs, evals = self.parallel_pass(sim_boards)
            ppt += time.time() - ppts
            for i, node in enumerate(sim_nodes):
                if self.to_run(i):
                    node.back_propagate(evals[i])
                    if not np.any(np.isnan(p_vecs[i])):
                        node.make_children(add_dirichlet(np.array(p_vecs[i])))

            self.runs -= 1

            for i, runs_left in enumerate(self.runs):
                if runs_left == 0:
                    values = np.array(
                        [
                            -node.value_score() if node is not None else -np.inf
                            for node in self.trees[i].children
                        ]
                    )
                    es = np.exp(values)
                    values = es / sum(es)
                    indices[i] = np.argmax(values)
                    boards[i] = get_board(self.trees[i].children[indices[i]])
                    vals[i] = values
                    next_trees[i] = self.trees[i].children[indices[i]]
        et = time.time() - st

        if self.benchmark and self.telemetry:
            with open(BENCHMARK_FILE, "a") as f:
                f.write(
                    f"Total Time: {round(et, 3)}, Forward Time: {round(ppt, 3)}, F/T = {round(ppt/et*100, 2)}%\n"
                )
                append_to_recent_key("percent_forward", ppt / et * 100)
        return self.first_boards, boards, vals, indices, next_trees
