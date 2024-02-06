import numpy as np
from common.board import BlankBoard
import torch
from c4net import C4Net
from typing import List


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

    def make_children(self, p_vec: np.ndarray):
        self.children = np.empty(p_vec.shape, dtype=Node)
        for i, p in enumerate(p_vec):
            self.children[i] = Node(self, p, i)

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
    head.make_children(p_vec)

    for i in range(runs):

        sim_node = select(head)
        sim_board = get_board(sim_node)
        p_vec, eval = nnet.forward(sim_board)
        sim_node.back_propagate(eval)
        p_vec_legalized = int(sim_board.legal_moves()) * p_vec
        sim_node.make_children(p_vec_legalized)

    final_eval = [(child.w / child.n, child) for child in head.children]
    final_eval.sort()
    favorite_child = final_eval[0][1]

    return get_board(favorite_child)


def select(tree: Node) -> Node:

    def _ucb(tree: Node) -> np.float64:
        # cbase = 19562
        # cinit = 1.25

        # this is classic ucb, I think alphazero implements a slightly altered version

        return tree.prior * np.sqrt(tree.parent.n) / (tree.n + 1) - tree.value_score()

    if tree.children is None:
        return tree
    else:
        max_ucb = -np.inf
        for child in tree.children:
            current_ucb = _ucb(child)
            if current_ucb > max_ucb:
                max_ucb = current_ucb
                favorite_child = child

    return select(favorite_child)


def get_board(node: Node) -> BlankBoard:

    if node.board is None:
        node.board = node.parent.board.move_from_int(node.child_index)

    return node.board
