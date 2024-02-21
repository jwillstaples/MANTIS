import numpy as np
from common.board import BlankBoard
import torch
from typing import List
from tqdm import tqdm

import time


device = "cuda" if torch.cuda.is_available() else "cpu"


class Node:
    def __init__(
        self, parent: "Node" = None, p: np.float64 = None, child_index: int = None
    ):
        self.parent = parent
        self.children = None
        self.board = None
        self.child_index = child_index
        self.n = 0
        self.w = 0
        self.p = p

    def make_children(self, ps: np.ndarray):
        self.children = np.empty(ps.shape, dtype=Node)
        legals = self.board.legal_moves()
        for i, pc in enumerate(ps):
            if legals[i] == True:
                self.children[i] = Node(self, pc, i)

    def value_score(self):
        if self.n > 0:
            return self.w / self.n
        else:
            return 0

    def back_propagate(self, eval):
        self.n += 1
        self.w += eval

        if not (self.parent is None):
            self.parent.back_propagate(-1 * eval)


def get_output(board: BlankBoard, nnet: torch.nn.Module):  # Optional 7x1, float
    if board.terminal_eval() == 2:
        p_vec, eval = nnet(board.to_tensor().unsqueeze(0).to(device))
        p_vec = p_vec.detach().cpu().numpy()[0, :]
        eval = eval.detach().cpu().numpy()[0]
        return p_vec, np.float64(eval)
    return None, board.player_perspective_eval()


def add_dirichlet(p_vec: np.ndarray) -> np.ndarray:
    epsilon = 0.25  # hyper-parameter for exploration
    noise = np.random.dirichlet(0.03 * np.ones(p_vec.shape))
    return (1 - epsilon) * p_vec + epsilon * noise


def mcts(
    head_board: BlankBoard,
    nnet: torch.nn.Module,
    runs: int = 500,
    head_node: Node = None,
):
    """
    gets best move

    returns next board, normalized values, index of move
    """
    start = time.time()
    forward_time = 0

    sf = time.time()
    p_vec, eval = get_output(head_board, nnet)
    forward_time += time.time() - sf

    if head_node is None:
        head = Node()
        head.board = head_board
        head.make_children(add_dirichlet(np.array(p_vec)))
        head.n = 0
    else:
        head = head_node
        head.parent = None
        head.child_index = None
        head.p = None

        runs = runs - head_node.n

    for _ in range(runs):
        sim_node = select(head)
        sim_board = get_board(sim_node)
        sf = time.time()
        p_vec, eval = get_output(sim_board, nnet)
        forward_time += time.time() - sf
        if p_vec is None:
            sim_node.back_propagate(eval)
        else:
            sim_node.back_propagate(eval)
            sim_node.make_children(add_dirichlet(np.array(p_vec)))

    values = np.array(
        [-node.value_score() if node is not None else -np.inf for node in head.children]
    )
    es = np.exp(values)
    values = es / sum(es)

    index = np.argmax(values)

    tot = time.time() - start
    # print(f"Tot: {tot}, f: {forward_time}, percent = {forward_time/tot*100}%")
    return get_board(head.children[index]), values, index, head.children[index]


def select(tree: Node) -> Node:
    def _ucb(tree: Node) -> np.float64:
        # cbase = 19562
        # cinit = 1.25

        # this is classic ucb, I think alphazero implements a slightly altered version
        if tree == None:
            return -np.inf

        return tree.p * np.sqrt(tree.parent.n) / (tree.n + 1) - tree.value_score()

    if tree.children is None or sum(tree.board.legal_moves()) == 0:
        return tree

    max_ucb = _ucb(tree.children[0])
    favorite_child = tree.children[0]
    for child in tree.children:
        current_ucb = _ucb(child)
        if current_ucb > max_ucb:
            max_ucb = current_ucb
            favorite_child = child

    if max_ucb == -np.inf:
        raise Exception("No max UCB")

    return select(favorite_child)


def get_board(node: Node) -> BlankBoard:
    if node.board is None:
        assert node.parent.board.legal_moves()[
            node.child_index
        ], "making board from illegal move"
        node.board = node.parent.board.move_from_int(node.child_index)

    return node.board


def print_tree(tree: Node, depth: int = 0):
    for i in range(depth):
        print("   ", end="")

    print(
        f"Move {tree.child_index} -- vists: {tree.n}, value score: {np.round(tree.value_score(), 4)}"
    )

    if tree.children is None:
        return
    elif depth > 1:
        return

    for child in tree.children:
        print_tree(child, depth=depth + 1)
