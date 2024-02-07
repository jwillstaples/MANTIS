import numpy as np
from common.board import BlankBoard
import torch
from connect4.c4net import C4Net
from typing import List
from tqdm import tqdm

from connect4.oracle_c4 import TestNet


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
        for i, pc in enumerate(ps):
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
            self.parent.back_propagate(eval)


def mcts(head_board: BlankBoard, nnet: torch.nn.Module, runs: int = 1000) -> BlankBoard:

    p_vec, eval = nnet.forward(head_board)

    head = Node()
    head.board = head_board
    head.make_children(head_board.legal_moves() * np.array(p_vec))
    head.n = 1

    for i in tqdm(range(runs)):

        sim_node = select(head)
        sim_board = get_board(sim_node)

        p_vec, eval = nnet.forward(sim_board)
        sim_node.back_propagate(np.float64(eval))
        p_vec_legalized = sim_board.legal_moves() * np.array(p_vec)
        sim_node.make_children(sim_board.legal_moves() * np.array(p_vec))

    print_tree(head)

    max_value = -np.inf
    for child in head.children:
        if child.value_score() > max_value:
            max_value = child.value_score()
            favorite_child = child

    print(max_value)

    return get_board(favorite_child)


def select(tree: Node) -> Node:

    def _ucb(tree: Node) -> np.float64:
        # cbase = 19562
        # cinit = 1.25

        # this is classic ucb, I think alphazero implements a slightly altered version

        return tree.p * np.sqrt(tree.parent.n) / (tree.n + 1) - tree.value_score()

    if tree.children is None:
        return tree

    max_ucb = -np.inf
    favorite_child = tree.children[0]
    for child in tree.children:
        current_ucb = _ucb(child)
        if current_ucb > max_ucb and child.p > 1e-8:
            max_ucb = current_ucb
            favorite_child = child

    if max_ucb == -np.inf:
        tree.p = 0
        return tree

    return select(favorite_child)


def get_board(node: Node) -> BlankBoard:

    if node.board is None:
        node.board = node.parent.board.move_from_int(node.child_index)

    return node.board


def print_tree(tree: Node, depth: int = 0):

    for i in range(depth):
        print("   ", end="")

    print(f"Move {tree.child_index} -- vists: {tree.n}, wins: {tree.w}")

    if tree.children is None:
        return
    elif depth > 1:
        return

    for child in tree.children:
        print_tree(child, depth=depth + 1)
