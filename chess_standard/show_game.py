from bitarray.util import pprint

from chess_standard.utils import *
from chess_standard.precompute import *
from chess_standard.board_chess import BoardChess


def simulate_human_vs_human(board, num_moves):
    exit_program = False

    symbols = {
        "P": "White Pawn",
        "R": "White Rook",
        "N": "White Knight",
        "B": "White Bishop",
        "Q": "White Queen",
        "K": "White King",
        "p": "Black Pawn",
        "r": "Black Rook",
        "n": "Black Knight",
        "b": "Black Bishop",
        "q": "Black Queen",
        "k": "Black King",
    }

    for _ in range(num_moves):
        # PRINT CURRENT TURN'S GAME STATE
        _ = board.visualize_current_gamestate()

        # PRINT CURRENT GAME STATE'S AVAILABLE PAWN MOVES
        if board.white_move == 1:
            print("\nAvailable Moves for White (format = origin -> target)")
        elif board.white_move == -1:
            print("\nAvailable Moves for Black (format = origin -> target)")

        encoded_moves = board.get_legal_moves()
        for i, test_move in enumerate(encoded_moves):
            origin_square, target_square, promotion_piece_type, special_move_flag = (
                decode_move(test_move)
            )

            piece_type = board.get_piece_of_square(origin_square)

            prior = []
            algebraic_notation_move = []
            if piece_type != "P" and piece_type != "p":
                prior.append(piece_type)
                algebraic_notation_move.append(piece_type)
            algebraic_origin_square = pos_idx_to_filerank(origin_square)
            algebraic_target_square = pos_idx_to_filerank(target_square)
            prior.append(algebraic_origin_square)

            if board.white_move == 1:
                target_bitboard = board.get_bitboard_of_square(target_square)

                if target_bitboard in board.black_piece_bitboards.values():
                    if piece_type == "P":
                        algebraic_notation_move.append(f"{algebraic_origin_square[0]}x")
                    else:
                        algebraic_notation_move.append("x")

                # HANDLE SPECIAL FLAGS
                if special_move_flag == 0:  # REGULAR MOVE
                    algebraic_notation_move.append(algebraic_target_square)

                elif special_move_flag == 1:  # PROMOTION MOVE
                    algebraic_notation_move.append(algebraic_target_square)
                    if promotion_piece_type == 0:
                        algebraic_notation_move.append("=R")
                    elif promotion_piece_type == 1:
                        algebraic_notation_move.append("=N")
                    elif promotion_piece_type == 2:
                        algebraic_notation_move.append("=B")
                    elif promotion_piece_type == 3:
                        algebraic_notation_move.append("=Q")

                elif special_move_flag == 2:  # EN PASSANT MOVE, BLACK CAPTURES WHITE
                    algebraic_notation_move.append(f"{algebraic_origin_square[0]}x")
                    algebraic_notation_move.append(algebraic_target_square)
                    algebraic_notation_move.append(" e.p.")

                elif special_move_flag == 3:  # CASTLING MOVE
                    if target_square > origin_square:  # KINGSIDE CASTLING
                        algebraic_notation_move = ["0-0"]
                    elif target_square < origin_square:  # QUEENSIDE CASTLING
                        algebraic_notation_move = ["0-0-0"]

            elif board.white_move == -1:
                target_bitboard = board.get_bitboard_of_square(target_square)

                if target_bitboard in board.white_piece_bitboards.values():
                    if piece_type == "p":
                        algebraic_notation_move.append(f"{algebraic_origin_square[0]}x")
                    else:
                        algebraic_notation_move.append("x")

                # HANDLE SPECIAL FLAGS
                if special_move_flag == 0:  # REGULAR MOVE
                    algebraic_notation_move.append(algebraic_target_square)

                elif special_move_flag == 1:  # PROMOTION MOVE
                    algebraic_notation_move.append(algebraic_target_square)
                    if promotion_piece_type == 0:
                        algebraic_notation_move.append("=r")
                    elif promotion_piece_type == 1:
                        algebraic_notation_move.append("=n")
                    elif promotion_piece_type == 2:
                        algebraic_notation_move.append("=b")
                    elif promotion_piece_type == 3:
                        algebraic_notation_move.append("=q")

                elif special_move_flag == 2:  # EN PASSANT MOVE, BLACK CAPTURES WHITE
                    algebraic_notation_move.append(f"{algebraic_origin_square[0]}x")
                    algebraic_notation_move.append(algebraic_target_square)
                    algebraic_notation_move.append(" e.p.")

                elif special_move_flag == 3:  # CASTLING MOVE
                    if target_square > origin_square:  # KINGSIDE CASTLING
                        algebraic_notation_move = ["0-0"]
                    elif target_square < origin_square:  # QUEENSIDE CASTLING
                        algebraic_notation_move = ["0-0-0"]

            print(
                f'Input [{i:>2}] - {symbols[piece_type]} moves {"".join(prior)} -> {"".join(algebraic_notation_move)}'
            )

        # SELECT AN INDEX
        while True:
            user_input = input(
                f'Enter a number 0-{len(encoded_moves)-1} (or "exit" to stop): '
            )

            if user_input == "exit":
                print("Program Exited.")
                exit_program = True
                break

            try:
                select_move_idx = int(user_input)
                if 0 <= select_move_idx <= (len(encoded_moves) - 1):
                    selected_move = encoded_moves[select_move_idx]
                    board.make_move(encoded_move=selected_move)
                    break
                else:
                    print(
                        f'Input out of bounds. Enter a number 0-{len(encoded_moves)-1} (or type "exit" to stop): '
                    )

            except ValueError:
                print(
                    f"Invalid Input. Enter a number 0-{len(encoded_moves)-1} (or type -1 to stop): "
                )

        print(
            "\n-------------------------------------------------------------------------------------------------------------"
        )
        if exit_program:
            break


if __name__ == "__main__":
    board = BoardChess(white_move=1)
    fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    board.set_fen_gamestate(fen)

    num_moves = 100
    simulate_human_vs_human(board, num_moves)

    # rook_rays = precompute_rook_rays()
    # occupied = board.occupied_squares
    # print(lerf_bitboard_to_2D_numpy(occupied))

    # potential_blockers = rook_rays['a1'] & occupied
    # print(lerf_bitboard_to_2D_numpy(potential_blockers))
