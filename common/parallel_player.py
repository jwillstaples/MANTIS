import torch
from tqdm import tqdm
from common.board import BlankBoard
from common.pmcts import Parallel_MCTS
from common.training_player import TrainingPlayer
from common.training_util import GameDataset, save_idxs
from typing import Type
import torch.nn as nn


from torch.multiprocessing import Queue, Process, Value

device = "cuda" if torch.cuda.is_available() else "cpu"


class ParallelPlayer(TrainingPlayer):
    def __init__(
        self,
        mcts_iter: int,
        old_exists: bool,
        SAVE_DIR: str,
        multicore: int,
        Net: Type[nn.Module],
        Board: Type[BlankBoard],
    ):
        super().__init__(mcts_iter, old_exists, SAVE_DIR, multicore, Net, Board)

    def generate_self_games(self, num):
        if self.multicore > 1:
            return self.parallel_self_play(num)
        return self.serial_self_play(num)

    def generate_eval_games(
        self, num, iteration, candidate_net, old_net, candidate_net_file
    ):
        if self.multicore > 1:
            return self.parallel_eval(
                num, iteration, candidate_net, old_net, candidate_net_file
            )
        return self.serial_eval(num, iteration, candidate_net, old_net)

    def play_games_in_parallel(
        self, num, net0, net1, self_play=False, telemetry=False, desc="Eval"
    ):
        games = [self.Board.from_start() for _ in range(num)]
        results = [2 for _ in range(num)]
        finished = [False for _ in range(num)]
        turn = 0

        boards = [[] for _ in range(num)]  # boards[i] is the list of moves from game i
        pis = [[] for _ in range(num)]
        idxs = [[] for _ in range(num)]
        current_trees = [None for _ in range(num)]
        if telemetry:
            pbar = tqdm(desc=f"{desc} - Moves Played", total=100)
        while 2 in results:
            for i in range(num):
                if results[i] != 2:
                    games[i] = self.Board.from_start()
                    current_trees[i] = None
                    finished[i] = True

            mcts = Parallel_MCTS(
                games=num,
                net=net0 if turn == 0 else net1,
                runs=self.mcts_iter,
                first_boards=games,
                passed_trees=current_trees,
                finished_games=finished,
            )
            first_boards, moves, mpis, midxs, mcurrent_trees = mcts.play()
            turn = 1 if turn == 0 else 0
            games = moves
            for i, (fb, pi, idx, tree) in enumerate(
                zip(first_boards, mpis, midxs, mcurrent_trees)
            ):
                if results[i] == 2:
                    boards[i].append(fb)
                    pis[i].append(pi)
                    idxs[i].append(idx)
                    if self_play:
                        current_trees[i] = tree
                    results[i] = games[i].terminal_eval()
            if telemetry:
                pbar.update(1)

        if not self_play:
            return results, idxs[0]

        t_datas = [[] for _ in range(num)]
        for i, (result, seq_boards, seq_pis) in enumerate(zip(results, boards, pis)):
            for j, (board, pi) in enumerate(zip(seq_boards, seq_pis)):
                reward = result if j % 2 == 0 else -result
                t_datas[i].append((board, pi, [float(reward)]))

        training_data = []
        [training_data.extend(t_data) for t_data in t_datas]

        ret_idxs = idxs[0]

        return training_data, ret_idxs

    def serial_self_play(self, num):
        net = self.Net().to(device)
        if not self.old_exists:
            net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
        net.eval()

        all_data, idxs = self.play_games_in_parallel(
            num, net, net, self_play=True, desc="SP"
        )

        return net, GameDataset(all_data), idxs

    def parallel_self_play(self, num):
        net = self.Net().to(device)
        if not self.old_exists:
            net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
        net.eval()

        queues = [Queue() for _ in range(self.multicore - 1)]
        stop_sigs = [Value("i", 0) for _ in range(self.multicore - 1)]

        processes = [Process(
            target=self.self_play_games_wrapper, args=(stop_sigs[i], queues[i], num, True)
        ) for i in range(self.multicore - 1)]

        [process.start() for process in processes]

        all_data, idxs = self.play_games_in_parallel(
            num, net, net, self_play=True, telemetry=True, desc="SP"
        )

        while (
            not all(stop_sig.value == 0 for stop_sig in stop_sigs)
        ): 
            continue
            
        [process.join(timeout=1) for process in processes]

        for queue in queues:
            proc_data = queue.get()
            all_data.extend(proc_data)
        
        return net, GameDataset(all_data), idxs

    def self_play_games_wrapper(self, stop_sig, queue, num, self_play):
        # may not be able to pass the network into the new function because it can't be pickled

        net = self.Net().to(device)
        if not self.old_exists:
            net.load_state_dict(torch.load("old.pt", map_location=torch.device(device)))
        net.eval()

        print("Parallel Start")
        all_data, _ = self.play_games_in_parallel(num, net, net, self_play, desc="SP")
        queue.put(all_data)
        stop_sig.value = 1

    def serial_eval(self, num, iteration, net, old_net):
        score = 0
        res = [0] * 3

        w_results, idxs = self.play_games_in_parallel(
            num // 2, net, old_net, False, telemetry=True
        )
        b_results, _ = self.play_games_in_parallel(
            num // 2, old_net, net, False, telemetry=True
        )

        for result in w_results:
            if result == 1:
                res[0] += 1
            if result == 0:
                res[1] += 1
            if result == -1:
                res[2] += 1
            score += result

        for result in b_results:
            if result == -1:
                res[0] += 1
            if result == 0:
                res[1] += 1
            if result == 1:
                res[2] += 1
            score -= result
        save_idxs(self.save_dir, idxs, f"e{iteration}")

        return score, res

    def parallel_eval(self, num, iteration, net, old_net, fp):
        score = 0
        res = [0] * 3

        queue = Queue()
        stop_signal = Value("i", 0)

        process = Process(
            target=self.eval_games_wrapper,
            args=(queue, stop_signal, num, fp),
        )
        process.start()

        w_results, idxs = self.play_games_in_parallel(
            num // 2, net, old_net, False, telemetry=True
        )

        while stop_signal.value == 0:
            continue
        process.join(timeout=1)
        b_results = queue.get()

        for result in w_results:
            if result == 1:
                res[0] += 1
            if result == 0:
                res[1] += 1
            if result == -1:
                res[2] += 1
            score += result

        for result in b_results:
            if result == -1:
                res[0] += 1
            if result == 0:
                res[1] += 1
            if result == 1:
                res[2] += 1
            score -= result
        save_idxs(self.save_dir, idxs, f"e{iteration}")

        return score, res

    def eval_games_wrapper(self, queue, stop_sig, GAMES_TO_EVAL, fp):
        net = self.Net().to(device)
        net.load_state_dict(torch.load(fp, map_location=torch.device(device)))

        old_net = self.Net().to(device)
        if self.old_exists:
            old_net.load_state_dict(torch.load("old.pt"))

        b_results, _ = self.play_games_in_parallel(
            GAMES_TO_EVAL // 2, old_net, net, False
        )
        queue.put(b_results)
        stop_sig.value = 1
