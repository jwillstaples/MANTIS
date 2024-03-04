import time
import chess

from prettytable import PrettyTable
from bitarray import bitarray
from bitarray.util import pprint

from chess_standard.utils import *
from chess_standard.board_chess import BoardChess


def perft(board, depth):
    if depth == 0:
        return 1 # LEAF NODE REACHED, RETURN 1 TO COUNT THE LEAF NODE
    
    num_positions = 0
    # board.visualize_current_gamestate()
    legal_moves = board.get_legal_moves() # returns a list of encoded legal moves
    # print(f'Depth: {depth}, # Legal Moves: {len(legal_moves)}')

    for move in legal_moves:
        board.make_move(move)
        num_positions += perft(board, depth - 1)
        board.unmake_move()
    
    return num_positions


def single_perft(board, max_depth):
    table = PrettyTable()
    table.field_names = ['depth', 'nodes', 'total_nodes', 'time']

    start_time = time.time()
    node_count = perft(board, max_depth)
    end_time = time.time()
    elapsed_time = end_time - start_time

    table.add_row([max_depth, node_count, node_count, f'{elapsed_time:.4f}'])

    table.align = 'l'
    print(table)

    return node_count


def table_perft(board, max_depth):
    counter = {}
    total_nodes = 0
    total_elapsed_time = 0

    for depth in range(1, max_depth+1):
        start_time = time.time()
        nodes = perft(board, depth)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_elapsed_time += elapsed_time
        total_nodes += nodes
        counter[depth] = (nodes, total_nodes, elapsed_time)

    table = PrettyTable()
    table.field_names = ['depth', 'nodes', 'total_nodes', 'time (s)']

    for depth, nodes in counter.items():
        node_count, total_node_count, elapsed_time = nodes
        table.add_row([depth, node_count, total_node_count, f'{elapsed_time:.4f}'], divider=(depth==max_depth))
    
    table.add_row(['-', '-', '-', f'{total_elapsed_time:.4f}'])
    
    table.align = 'l'
    print(table)

    return counter

if __name__ == '__main__':
    board = BoardChess(white_move=1)

    perft_depth = 3
    fen = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'

    # board.visualize_current_gamestate()
    board.set_fen_gamestate(fen)
    # board.visualize_current_gamestate()

    single_depth_count = single_perft(board, perft_depth)
    print('')
    all_depth_counts = table_perft(board, perft_depth)



