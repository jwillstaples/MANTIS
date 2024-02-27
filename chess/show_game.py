
from bitarray.util import pprint

from chess.utils import *
from chess.precompute import *
from chess.board_chess import BoardChess


def simulate_human_vs_human(board, num_moves):
    exit_program = False

    for _ in range(num_moves):
        # PRINT CURRENT TURN'S GAME STATE
        _ = board.visualize_current_gamestate()

        # PRINT CURRENT GAME STATE'S AVAILABLE PAWN MOVES
        if board.white_move == 1:
            print('\nAvailable Pawn Moves for White (format = (piece, origin, target))')
        elif board.white_move == -1:
            print('\nAvailable Pawn Moves for Black (format = (piece, origin, target))')
        
        encoded_moves = board.get_pseudolegal_moves()
        for i, test_move in enumerate(encoded_moves):
            origin_square, target_square, promotion_piece_type, special_move_flag = decode_move(test_move)
            # print(origin_square, target_square)
            print(f'Input [{i}] - ({board.get_piece_of_square(origin_square)}, {pos_idx_to_filerank(origin_square)}, {pos_idx_to_filerank(target_square)})')
        
        # SELECT AN INDEX 
        while True:
            user_input = input(f'Enter a number 0-{len(encoded_moves)-1} (or "exit" to stop): ')

            if user_input == 'exit':
                print('Program Exited.')
                exit_program = True
                break

            try:
                select_move_idx = int(user_input)
                if (0 <= select_move_idx <= (len(encoded_moves)-1)):
                    selected_move = encoded_moves[select_move_idx]
                    board.make_move(encoded_move=selected_move)
                    break
                else:
                    print(f'Input out of bounds. Enter a number 0-{len(encoded_moves)-1} (or type "exit" to stop): ')

            except ValueError:
                print(f'Invalid Input. Enter a number 0-{len(encoded_moves)-1} (or type -1 to stop): ')
        
        print('\n-------------------------------------------------------------------------------------------------------------')
        if exit_program:
            break


if __name__ == '__main__':
    board = BoardChess(white_move=1)
                       
    num_moves = 10
    # simulate_human_vs_human(board, num_moves)

    rook_rays = precompute_rook_rays()
    occupied = board.occupied_squares
    # print(lerf_bitboard_to_2D_numpy(occupied))

    potential_blockers = rook_rays['a1'] & occupied
    print(lerf_bitboard_to_2D_numpy(potential_blockers))