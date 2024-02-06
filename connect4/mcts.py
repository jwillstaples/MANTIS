import numpy as np 
from common.board import BlankBoard
from c4net import C4Net
from typing import List

class Node: 

    def __init__(self, parent: "Node"=None, p: np.float64 = None, child_index: int = None): 
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

    def back_propagate(self, eval): 
        
        self.n += 1 
        self.w += eval

        if not (self.parent is None): 
            self.parent.back_propagate(eval)


def mcts(head_board: BlankBoard, runs: int = 1000) -> BlankBoard: 

    nnet = C4Net()
    p_vec, eval = nnet.forward(head_board)

    head = Node()
    head.make_children(p_vec)
    
    for i in range(runs): 

        sim_node = select(head)
        sim_board = get_board(sim_node)
        p_vec, eval = nnet.forward(sim_board)
        sim_node.back_propagate(eval)
        p_vec_legalized = int(sim_board.legal_moves()) * p_vec
        sim_node.make_children(p_vec_legalized)

    final_eval = [(child.w/child.n, child) for child in head.children]
    final_eval.sort()
    favorite_child = final_eval[0][1]

    return get_board(favorite_child)

def select(tree: Node) -> Node: 

    def _minimax(tree: Node) -> (np.float64, Node): 
    
        if tree.children is None: 
            return (tree, _ubc(tree))
        else: 
            max_ubc = -np.inf
            favorite_child = None
            for child in tree.children: 
                child_ubc, _ = _minimax(child)
                if child_ubc > max_ubc: 
                    max_ubc = child_ubc
                    favorite_child = child
        
        return (max_ubc, favorite_child)

    def _ubc(leaf: Node) -> np.float64: 
        return 0     
    
    _, sim_node = _minimax(tree)

    return sim_node

def get_board(node: Node) -> BlankBoard: 

    if node.board is None: 
        node.board = node.parent.board.move_from_int(node.child_index)
    
    return node.board
    
    

