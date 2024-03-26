import time
import chess

from prettytable import PrettyTable
from bitarray import bitarray
from bitarray.util import pprint

from chess_standard.utils import *
from chess_standard.board_chess import BoardChess


def perft_engine(board, depth):
    if depth == 0:
        return 1

    num_positions = 0
    legal_moves = board.legal_moves

    for move in legal_moves:
        board.push(move)
        num_positions += perft_engine(board, depth - 1)
        board.pop()

    return num_positions


def perft_mine(board, depth):
    if depth == 0:
        return 1  # LEAF NODE REACHED, RETURN 1 TO COUNT THE LEAF NODE

    num_positions = 0
    # board.visualize_current_gamestate()
    legal_moves = board.get_legal_moves()  # returns a list of encoded legal moves
    # print(f'Depth: {depth}, # Legal Moves: {len(legal_moves)}')

    for move in legal_moves:
        board.make_move(move)
        num_positions += perft_mine(board, depth - 1)
        board.unmake_move()

    return num_positions


def single_perft(board_mine, board_engine, max_depth):
    table = PrettyTable()
    table.field_names = ["engine", "depth", "nodes", "total_nodes", "time"]

    start_time_mine = time.time()
    node_count_mine = perft_mine(board_mine, max_depth)
    end_time_mine = time.time()
    elapsed_time_mine = end_time_mine - start_time_mine

    start_time_engine = time.time()
    node_count_engine = perft_engine(board_engine, max_depth)
    end_time_engine = time.time()
    elapsed_time_engine = end_time_engine - start_time_engine

    table.add_row(
        [
            "mine",
            max_depth,
            node_count_mine,
            node_count_mine,
            f"{elapsed_time_mine:.4f}",
        ]
    )
    table.add_row(
        [
            "engine",
            max_depth,
            node_count_engine,
            node_count_engine,
            f"{elapsed_time_engine:.4f}",
        ]
    )

    table.align = "l"
    print(table)

    return (node_count_mine, node_count_engine)


def table_perft(board_mine, board_engine, max_depth):
    counter = {}

    total_nodes_mine = 0
    total_elapsed_time_mine = 0

    total_nodes_engine = 0
    total_elapsed_time_engine = 0

    for depth in range(1, max_depth + 1):
        start_time_mine = time.time()
        node_count_mine = perft_mine(board_mine, depth)
        end_time_mine = time.time()
        elapsed_time_mine = end_time_mine - start_time_mine

        start_time_engine = time.time()
        node_count_engine = perft_engine(board_engine, depth)
        end_time_engine = time.time()
        elapsed_time_engine = end_time_engine - start_time_engine

        total_elapsed_time_mine += elapsed_time_mine
        total_elapsed_time_engine += elapsed_time_engine

        total_nodes_mine += node_count_mine
        total_nodes_engine += node_count_engine

        counter[depth] = (
            node_count_mine,
            total_nodes_mine,
            elapsed_time_mine,
            node_count_engine,
            total_nodes_engine,
            elapsed_time_engine,
        )

    table = PrettyTable()
    table.field_names = ["engine", "depth", "nodes", "total_nodes", "time (s)"]

    for depth, nodes in counter.items():
        (
            node_count_mine,
            total_nodes_mine,
            elapsed_time_mine,
            node_count_engine,
            total_nodes_engine,
            elapsed_time_engine,
        ) = nodes
        table.add_row(
            [
                "mine",
                depth,
                node_count_mine,
                total_nodes_mine,
                f"{elapsed_time_mine:.4f}",
            ]
        )
        table.add_row(
            [
                "engine",
                depth,
                node_count_engine,
                total_nodes_engine,
                f"{elapsed_time_engine:.4f}",
            ],
            divider=True,
        )

    table.add_row(["mine", "-", "-", "-", f"{total_elapsed_time_mine:.4f}"])
    table.add_row(["engine", "-", "-", "-", f"{total_elapsed_time_engine:.4f}"])

    table.align = "l"
    print(table)

    return counter


def helper_get_legal_moves(board):
    legal_moves = []
    if isinstance(board, BoardChess):
        legal_moves = board.get_legal_moves()
    elif isinstance(board, chess.Board):
        legal_moves = list(board.legal_moves)
    return legal_moves


def perft(board, depth):
    if depth == 0:
        return 1  # LEAF NODE REACHED, RETURN 1 TO COUNT THE LEAF NODE

    num_positions = 0
    legal_moves = helper_get_legal_moves(board)  # Ensure this returns a list of moves

    for move in legal_moves:
        # MAKE MOVE
        if isinstance(board, BoardChess):
            board.make_move(move)
        elif isinstance(board, chess.Board):
            board.push(move)

        num_positions += perft(board, depth - 1)

        # UNMAKE MOVE
        if isinstance(board, BoardChess):
            board.unmake_move()
        elif isinstance(board, chess.Board):
            board.pop()

    return num_positions


def perft_divide(board, depth):
    perft_division_res = {"total": 0}

    if depth <= 0:
        return perft_division_res

    legal_moves = helper_get_legal_moves(board)

    for move in legal_moves:
        # MAKE MOVE
        if isinstance(board, BoardChess):
            board.make_move(move)
        elif isinstance(board, chess.Board):
            board.push(move)

        positions_from_move = perft(board, depth - 1)

        # UNMAKE MOVE
        if isinstance(board, BoardChess):
            board.unmake_move()
        elif isinstance(board, chess.Board):
            board.pop()

        # MOVE STRING
        if isinstance(board, BoardChess):
            origin_square, target_square, promotion_piece_type, special_move_flag = (
                decode_move(move)
            )
            origin_filerank = pos_idx_to_filerank(origin_square)
            target_filerank = pos_idx_to_filerank(target_square)
            move_str = origin_filerank + target_filerank
            if special_move_flag == 1:
                if promotion_piece_type == 0:
                    move_str += "r"
                elif promotion_piece_type == 1:
                    move_str += "n"
                elif promotion_piece_type == 2:
                    move_str += "b"
                elif promotion_piece_type == 3:
                    move_str += "q"

        elif isinstance(board, chess.Board):
            move_str = move.uci()

        perft_division_res[move_str] = positions_from_move
        perft_division_res["total"] += positions_from_move

    return perft_division_res


if __name__ == "__main__":
    fpath = "chess_standard/perft_tests.txt"
    depth = 3

    fen_strings = []
    titles = []
    with open(fpath, "r") as file:
        for line in file:
            if line.startswith("Test Case: "):
                titles.append(line.strip()[11:])
            else:
                fen = line.strip()
                if fen:
                    fen_strings.append(fen)

    fen_strings = [fen_strings[4]]
    titles = [titles[4]]

    for i, fen in enumerate(fen_strings):
        # print(fen)
        board_mine = BoardChess(white_move=1)
        board_mine.set_fen_gamestate(fen)
        board_engine = chess.Board(fen)

        # single_depth_count = single_perft(board_mine, board_engine, perft_depth)
        # print('')
        # all_depth_counts = table_perft(board_mine, board_engine, perft_depth)

        start_mine = time.time()
        divide_mine = perft_divide(board_mine, depth)
        end_mine = time.time()

        start_engine = time.time()
        divide_engine = perft_divide(board_engine, depth)
        end_engine = time.time()

        moves = sorted(set(divide_mine.keys()).union(divide_engine.keys()) - {"total"})

        red_start = "\033[91m"
        red_end = "\033[0m"

        same = True
        print(f'\nPERFT Test Case {i+1} -- "{titles[i]}"')
        print(f"\nPERFT comparison for Depth={depth}")
        print(f"{fen}\n")
        for move in moves:
            value1 = divide_mine.get(move, "N/A")
            value2 = divide_engine.get(move, "N/A")

            if value1 == value2:
                print(f"{move}:\t{value1}\t{value2}")
            else:
                same = False
                print(
                    f"{red_start}{move}:\t{value1}\t{value2}\t{value1 - value2}{red_end}"
                )

        total1 = divide_mine["total"]
        total2 = divide_engine["total"]
        if total1 == total2:
            print(f"\nNodes:\t{total1}\t{total2}")
        else:
            print(
                f"\n{red_start}Nodes:\t{total1}\t{total2}\t{total1 - total2}{red_end}"
            )
        print(
            f"Times:\t{(end_mine - start_mine):.3f}\t{(end_engine - start_engine):.3f}"
        )

        if same:
            print(f'PERFT Test Case {i+1} -- "{titles[i]}" passed!')
        else:
            print(f'Error(s) in PERFT Test Case {i+1} -- "{titles[i]}"!')
