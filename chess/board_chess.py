
import numpy as np

from prettytable import PrettyTable
from bitarray.util import zeros

from common.board import BlankBoard
from chess.utils import *
from chess.precompute import *


class BoardChess(BlankBoard):
    '''
    - Board Representation: Little-Endian Rank File
        - Little-Endian: MSB = Last square, h8; LSB = First square, a1
        - Rank-File: Groups of 8 bits are ranks, with each bit being a column
        - Slicing: Python slicing makes MSB = 0 and LSB = 63, so need to do [63 - index]
    - Move Representation: 16-bit bitarray representation
        - Bits   0-5: LERF Destination Square (0 - 63)
        - Bits  6-11: LERF Origin Square (0 - 63)
        - Bits 12-13: Promotion piece type (00 = Rook, 01 = Knight, 10 = Bishop, 11 = Queen)
        - Bits 14-15: Special move flag (01 = promotion, 10 = en passant, 11 = castling)
    '''
    def __init__(self, white_move=1):
        '''
        white_move: 1 = White's turn to move, -1 = Black's turn to move
        '''
        # Initialize bitboards for each piece type and color   
        self.white_pawns = hex_to_bitarray('0x000000000000FF00')
        self.white_rooks = hex_to_bitarray('0x0000000000000081')
        self.white_knights = hex_to_bitarray('0x0000000000000042')
        self.white_bishops = hex_to_bitarray('0x0000000000000024')
        self.white_queens = hex_to_bitarray('0x0000000000000010')
        self.white_king = hex_to_bitarray('0x0000000000000008')

        self.black_pawns = hex_to_bitarray('0x00FF000000000000')
        self.black_rooks = hex_to_bitarray('0x8100000000000000')
        self.black_knights = hex_to_bitarray('0x4200000000000000')
        self.black_bishops = hex_to_bitarray('0x2400000000000000')
        self.black_queens = hex_to_bitarray('0x1000000000000000')
        self.black_king = hex_to_bitarray('0x0800000000000000')
        
        self.white_piece_bitboards = {
            'P': self.white_pawns, 
            'R': self.white_rooks, 
            'N': self.white_knights, 
            'B': self.white_bishops, 
            'Q': self.white_queens, 
            'K': self.white_king
        }
        self.black_piece_bitboards = {
            'p': self.black_pawns, 
            'r': self.black_rooks, 
            'n': self.black_knights, 
            'b': self.black_bishops, 
            'q': self.black_queens, 
            'k': self.black_king
        }
        
        # Initialize bitboards for overall board squares
        self.white_pieces = self.white_pawns | self.white_knights | self.white_bishops | self.white_rooks | self.white_queens | self.white_king
        self.black_pieces = self.black_pawns | self.black_knights | self.black_bishops | self.black_rooks | self.black_queens | self.black_king
        self.occupied_squares = self.white_pieces | self.black_pieces
        self.empty_squares = ~self.occupied_squares
        
        # Initialize bitboards for ranks, files, diagonals, antidiagonals
        self.file = {
            'a': hex_to_bitarray('0x8080808080808080'), 
            'b': hex_to_bitarray('0x4040404040404040'),
            'c': hex_to_bitarray('0x2020202020202020'), 
            'd': hex_to_bitarray('0x1010101010101010'),
            'e': hex_to_bitarray('0x0808080808080808'),
            'f': hex_to_bitarray('0x0404040404040404'),
            'g': hex_to_bitarray('0x0202020202020202'), 
            'h': hex_to_bitarray('0x0101010101010101'),
        }

        self.rank = {
            '1': hex_to_bitarray('0x00000000000000FF'),
            '2': hex_to_bitarray('0x000000000000FF00'),
            '3': hex_to_bitarray('0x0000000000FF0000'),
            '4': hex_to_bitarray('0x00000000FF000000'),
            '5': hex_to_bitarray('0x000000FF00000000'),
            '6': hex_to_bitarray('0x0000FF0000000000'),
            '7': hex_to_bitarray('0x00FF000000000000'),
            '8': hex_to_bitarray('0xFF00000000000000'),
        }

        self.diag_bitboards = {
            '1': hex_to_bitarray('0x0000000000000080'),
            '2': hex_to_bitarray('0x0000000000008040'),
            '3': hex_to_bitarray('0x0000000000804020'),
            '4': hex_to_bitarray('0x0000000080402010'),
            '5': hex_to_bitarray('0x0000008040201008'),
            '6': hex_to_bitarray('0x0000804020100804'),
            '7': hex_to_bitarray('0x0080402010080402'),
            '8': hex_to_bitarray('0x8040201008040201'),
            '9': hex_to_bitarray('0x4020100804020100'),
            '10': hex_to_bitarray('0x2010080402010000'),
            '11': hex_to_bitarray('0x1008040201000000'),
            '12': hex_to_bitarray('0x0804020100000000'),
            '13': hex_to_bitarray('0x0402010000000000'),
            '14': hex_to_bitarray('0x0201000000000000'),
            '15': hex_to_bitarray('0x0100000000000000'),
        }

        self.antidiag_bitboards = {
            '1': hex_to_bitarray('0x0000000000000001'),
            '2': hex_to_bitarray('0x0000000000000102'),
            '3': hex_to_bitarray('0x0000000000010204'),
            '4': hex_to_bitarray('0x0000000001020408'),
            '5': hex_to_bitarray('0x0000000102040810'),
            '6': hex_to_bitarray('0x0000010204081020'),
            '7': hex_to_bitarray('0x0001020408102040'),
            '8': hex_to_bitarray('0x0102040810204080'),
            '9': hex_to_bitarray('0x0204081020408000'),
            '10': hex_to_bitarray('0x0408102040800000'),
            '11': hex_to_bitarray('0x0810204080000000'),
            '12': hex_to_bitarray('0x1020408000000000'),
            '13': hex_to_bitarray('0x2040800000000000'),
            '14': hex_to_bitarray('0x4080000000000000'),
            '15': hex_to_bitarray('0x8000000000000000'),
        }

        # Initialize sliding piece attack rays
        self.pawn_rays = precompute_pawn_rays()
        self.knight_rays = precompute_knight_rays()
        self.king_rays = precompute_king_rays()
        self.rook_rays = precompute_rook_rays()
        self.bishop_rays = precompute_bishop_rays()

        # Initialize game state variables
        self.move_history = []
        self.white_move = white_move # 1 = White's move, -1 = Black's move
        self.turn_counter = 0
        self.en_passant_target_pos_idx = -1
    

    # ----------------------------------- BOARD UPDATE METHODS ------------------------------------------------------------------------------------------------------
    def make_move(self, encoded_move):
        '''
        encoded_move: LEGAL encoded move bitarray
        - Move Representation: 16-bit bitarray representation
            - Bits   0-5: LERF Destination Square (0 - 63)
            - Bits  6-11: LERF Origin Square (0 - 63)
            - Bits 12-13: Promotion piece type (00 = Rook, 01 = Knight, 10 = Bishop, 11 = Queen)
            - Bits 14-15: Special move flag (01 = promotion, 10 = en passant, 11 = castling)
        Can add to move list for every combination for last two flags IF applicable
        i.e., pawn is going from second to last to last rank or whatever the fuck castling rules are
        need to make a method to obtain set of legal moves also
        '''
        origin_square, target_square, promotion_piece_type, special_move_flag = decode_move(encoded_move)
        print(f'MOVE CHOSEN: LERF origin = {origin_square ^ 56}, LERF target = {target_square ^ 56}, promo piece = {promotion_piece_type}, special = {special_move_flag}')
        
        piece_type = self.get_piece_of_square(origin_square)
        
        algebraic_notation_move = []
        if piece_type != 'P' and piece_type != 'p':
            algebraic_notation_move.append(piece_type)
        algebraic_origin_square = pos_idx_to_filerank(origin_square)
        algebraic_target_square = pos_idx_to_filerank(target_square)
        
        if self.white_move == 1:
            origin_bitboard = self.white_piece_bitboards[piece_type]
            target_bitboard = self.get_bitboard_of_square(target_square)
            
            # First set the val of the piece type bitboard at idx origin_square = 0
            # represents the piece moving away from origin_square
            origin_bitboard[origin_square] = 0
            
            # Next, check if target_square is occupied by a black piece, and if so,
            # set the val of the black piece type bitboard at idx target_square = 0
            # represents the black piece being captured
            if target_bitboard in self.black_piece_bitboards.values(): 
                target_bitboard[target_square] = 0
                if piece_type == 'P':
                    algebraic_notation_move.append(f'{algebraic_origin_square[0]}x')
                else:
                    algebraic_notation_move.append('x')
                
            # CHECK IF MOVE OPENS WHITE UP TO GETTING EN PASSANT'ED
            if (
                (piece_type == 'P') and 
                (int(algebraic_target_square[1]) - int(algebraic_origin_square[1]) == 2)
            ):
                en_passant_target_file = algebraic_origin_square[0]
                en_passant_target_rank = str(int(algebraic_origin_square[1]) + 1)
                en_passant_target_filerank = en_passant_target_file + en_passant_target_rank
                self.en_passant_target_pos_idx = filerank_to_pos_idx(en_passant_target_filerank)
            # OTHERWISE, RESET BLACK'S POTENTIAL EN PASSANT TARGET POS IDX
            else:
                self.en_passant_target_pos_idx = -1


            # HANDLE SPECIAL FLAGS
            if special_move_flag == 0: # REGULAR MOVE
                origin_bitboard[target_square] = 1
                algebraic_notation_move.append(algebraic_target_square)
            
            elif special_move_flag == 1: # PROMOTION MOVE
                algebraic_notation_move.append(algebraic_target_square)
                if promotion_piece_type == 0:
                    self.white_rooks[target_square] = 1
                    algebraic_notation_move.append('=R')
                elif promotion_piece_type == 1:
                    self.white_knights[target_square] = 1
                    algebraic_notation_move.append('=N')
                elif promotion_piece_type == 2:
                    self.white_bishops[target_square] = 1
                    algebraic_notation_move.append('=B')
                elif promotion_piece_type == 3:
                    self.white_queens[target_square] = 1
                    algebraic_notation_move.append('=Q')

            elif special_move_flag == 2: # EN PASSANT MOVE, WHITE CAPTURES BLACK
                origin_bitboard[target_square] = 1

                file_t, rank_t = algebraic_target_square[0], algebraic_target_square[1]
                target_pawn = file_t + str(int(rank_t) - 1)
                self.black_pawns[filerank_to_pos_idx(target_pawn)] = 0

                algebraic_notation_move.append(f'{algebraic_origin_square[0]}x')
                algebraic_notation_move.append(algebraic_target_square)


        elif self.white_move == -1:
            origin_bitboard = self.black_piece_bitboards[piece_type]
            target_bitboard = self.get_bitboard_of_square(target_square)
            
            origin_bitboard[origin_square] = 0
            
            if target_bitboard in self.white_piece_bitboards.values(): 
                target_bitboard[target_square] = 0
                if piece_type == 'p':
                    algebraic_notation_move.append(f'{algebraic_origin_square[0]}x')
                else:
                    algebraic_notation_move.append('x')

            # CHECK IF MOVE OPENS BLACK UP TO GETTING EN PASSANT'ED
            if (
                (piece_type == 'p') and 
                (int(algebraic_origin_square[1]) - int(algebraic_target_square[1]) == 2)
            ):
                en_passant_target_file = algebraic_origin_square[0]
                en_passant_target_rank = str(int(algebraic_origin_square[1]) - 1)
                en_passant_target_filerank = en_passant_target_file + en_passant_target_rank
                self.en_passant_target_pos_idx = filerank_to_pos_idx(en_passant_target_filerank)
            # OTHERWISE, RESET WHITE'S POTENTIAL EN PASSANT TARGET POS IDX
            else:
                self.en_passant_target_pos_idx = -1


            # HANDLE SPECIAL FLAGS
            if special_move_flag == 0: # REGULAR MOVE
                origin_bitboard[target_square] = 1
                algebraic_notation_move.append(algebraic_target_square)

            elif special_move_flag == 1: # PROMOTION MOVE
                algebraic_notation_move.append(algebraic_target_square)
                if promotion_piece_type == 0:
                    self.black_rooks[target_square] = 1
                    algebraic_notation_move.append('=r')
                elif promotion_piece_type == 1:
                    self.black_knights[target_square] = 1
                    algebraic_notation_move.append('=n')
                elif promotion_piece_type == 2:
                    self.black_bishops[target_square] = 1
                    algebraic_notation_move.append('=b')
                elif promotion_piece_type == 3:
                    self.black_queens[target_square] = 1
                    algebraic_notation_move.append('=q')
            
            elif special_move_flag == 2: # EN PASSANT MOVE, BLACK CAPTURES WHITE
                origin_bitboard[target_square] = 1

                file_t, rank_t = algebraic_target_square[0], algebraic_target_square[1]
                target_pawn = file_t + str(int(rank_t) + 1)
                self.white_pawns[filerank_to_pos_idx(target_pawn)] = 0

                algebraic_notation_move.append(f'{algebraic_origin_square[0]}x')
                algebraic_notation_move.append(algebraic_target_square)

        # update the game state
        self.move_history.append(''.join(algebraic_notation_move))
        self.update_gamestate()
            
            
    def update_gamestate(self):
        self.white_pieces = self.white_pawns | self.white_knights | self.white_bishops | self.white_rooks | self.white_queens | self.white_king
        self.black_pieces = self.black_pawns | self.black_knights | self.black_bishops | self.black_rooks | self.black_queens | self.black_king
        self.occupied_squares = self.white_pieces | self.black_pieces
        self.empty_squares = ~self.occupied_squares
        self.white_move *= -1
        self.turn_counter += 1

    
    def visualize_current_gamestate(self):
        white_symbols = {1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K'}
        black_symbols = {1: 'p', 2: 'r', 3: 'n', 4: 'b', 5: 'q', 6: 'k'}
        white_keys, black_keys = set(white_symbols.keys()), set(black_symbols.keys())
        np_white, np_black = self.get_numpy_white(), self.get_numpy_black()
        
        table = PrettyTable()
        table.border = False
        table.field_names = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        table.add_row(['--','--','--','--','--','--','--','--','--'])

        game_state_array = []

        for rank in range(8):
            row = [f'{8-rank} |']
            for file in range(8):
                if np_white[rank][file] in white_keys:
                    row.append(white_symbols[np_white[rank][file]])
                elif np_black[rank][file] in black_keys:
                    row.append(black_symbols[np_black[rank][file]])
                else:
                    row.append('.')
            game_state_array.append(row[1:])
            table.add_row(row)
            
        if self.white_move == 1:
            print(f'Turn: {self.turn_counter} | White to Move')
        elif self.white_move == -1:
            print(f'Turn: {self.turn_counter} | Black to Move')
        print(f'Move History: {self.move_history}')
        print(table)
        print('')

        return game_state_array


    # ----------------------------------- BOARD GETTER METHODS -------------------------------------------------------------------------------------------------------
    def get_piece_of_square(self, target_square):
        for piece_symbol, piece_bitboard in self.white_piece_bitboards.items():
            if piece_bitboard[target_square] == 1:
                return piece_symbol
        for piece_symbol, piece_bitboard in self.black_piece_bitboards.items():
            if piece_bitboard[target_square] == 1:
                return piece_symbol
        return None # empty square
    
        
    def get_bitboard_of_square(self, target_square):
        for piece_bitboard in self.white_piece_bitboards.values():
            if piece_bitboard[target_square] == 1:
                return piece_bitboard
        for piece_bitboard in self.black_piece_bitboards.values():
            if piece_bitboard[target_square] == 1:
                return piece_bitboard
        return None
    
    
    def get_numpy_white(self):
        np_board = np.zeros(64, dtype=int)
            
        for i, piece_bitboard in enumerate(self.white_piece_bitboards.values()):
            np_bitboard = lerf_bitboard_to_1D_numpy(piece_bitboard)
            np_board[np_bitboard == 1] = i + 1
            
        return np_board.reshape((8, 8))
    
    
    def get_numpy_black(self):
        np_board = np.zeros(64, dtype=int)
            
        for i, piece_bitboard in enumerate(self.black_piece_bitboards.values()):
            np_bitboard = lerf_bitboard_to_1D_numpy(piece_bitboard)
            np_board[np_bitboard == 1] = i + 1
            
        return np_board.reshape((8, 8))
    
    
    def get_numpy_empty(self):
        np_board = lerf_bitboard_to_1D_numpy(self.empty_squares)
        return np_board.reshape((8, 8))
    

    # ----------------------------------- PIECE MOVE GENERATION METHODS -----------------------------------------------------------------------------------------------
    def get_attack_ray_of_target_square(self, target_square_pos_idx, input_type):
        '''
        For a target_square pos_idx, gets the piece type on that square and returns its UNMASKED bitboard attack ray
        I.e., you'll want to mask attack_ray friendly pieces to filter out illegal attacks
        '''
        attack_ray = zeros(64, endian='little')

        occupied_minus_king = self.occupied_squares.copy()
        if self.white_move == 1:
            occupied_minus_king[self.white_king.search(1)[0]] = 0
        elif self.white_move == -1:
            occupied_minus_king[self.black_king.search(1)[0]] = 0

        if input_type == 'rook':
            for dir4 in [0, 1, 2, 3]:
                rook_attacks = getRookRayAttacks(self.rook_rays, occupied_minus_king, dir4, target_square_pos_idx)
                attack_ray |= rook_attacks

        elif input_type == 'bishop':
            for dir4 in [0, 1, 2, 3]:
                bishop_attacks = getBishopRayAttacks(self.bishop_rays, occupied_minus_king, dir4, target_square_pos_idx)
                attack_ray |= bishop_attacks

        elif input_type == 'queen':
            for dir4 in [0, 1, 2, 3]:
                line_attacks = getRookRayAttacks(self.rook_rays, occupied_minus_king, dir4, target_square_pos_idx)
                diag_attacks = getBishopRayAttacks(self.bishop_rays, occupied_minus_king, dir4, target_square_pos_idx)
                queen_attacks = line_attacks | diag_attacks
                attack_ray |= queen_attacks

        elif input_type == 'white_pawn' or input_type == 'black_pawn':
            if input_type == 'white_pawn':
                attack_ray = self.pawn_rays[0][target_square_pos_idx].copy()
            elif input_type == 'black_pawn':
                attack_ray = self.pawn_rays[1][target_square_pos_idx].copy()

        elif input_type == 'knight':
            attack_ray = self.knight_rays[target_square_pos_idx].copy()

        elif input_type == 'king':
            attack_ray = self.king_rays[target_square_pos_idx].copy()
        
        return attack_ray


    def get_pinned_piece_dict(self, king_pos):
        pinned_piece_dict = {} # pinned_piece_pos_idx : pinner_pos_idx

        king_rook_attack_rays = [zeros(64, endian='little') for _ in range(4)]
        king_bishop_attack_rays = [zeros(64, endian='little') for _ in range(4)]
        king_queen_attack_rays = [zeros(64, endian='little') for _ in range(8)]
        for dir4 in [0, 1, 2, 3]:
            # 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST
            king_rook_attack_rays[dir4] = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, king_pos)
            # 0 = NORTHWEST, 1 = NORTHEAST, 2 = SOUTHEAST, 3 = SOUTHWEST
            king_bishop_attack_rays[dir4] = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, king_pos)
            # 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST, 4 = NORTHWEST, 5 = NORTHEAST, 6 = SOUTHEAST, 7 = SOUTHWEST
            king_queen_attack_rays[dir4] = king_rook_attack_rays[dir4].copy()
            king_queen_attack_rays[dir4+4] = king_bishop_attack_rays[dir4].copy()

        if self.white_move == 1:
            for pinning_attackers in ['r', 'b', 'q']:
                for pidx in self.black_piece_bitboards[pinning_attackers].search(1):
                    attacker_ray = [zeros(64, endian='little') for _ in range(8)]

                    if pinning_attackers == 'r':
                        for dir4 in [0, 1, 2, 3]:
                            # 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST
                            attacker_ray[dir4] = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pidx)

                        king_pinned_from_northern_rook = king_rook_attack_rays[0] & attacker_ray[2] # use rook's southern ray
                        king_pinned_from_eastern_rook = king_rook_attack_rays[1] & attacker_ray[3] # use rook's western ray
                        king_pinned_from_southern_rook = king_rook_attack_rays[2] & attacker_ray[0] # use rook's northern ray
                        king_pinned_from_western_rook = king_rook_attack_rays[3] & attacker_ray[1] # use rook's eastern ray

                        combined = (
                            king_pinned_from_northern_rook | 
                            king_pinned_from_eastern_rook | 
                            king_pinned_from_southern_rook | 
                            king_pinned_from_western_rook
                        )
                        for pinned_piece_idx in combined.search(1):
                            if self.white_pieces[pinned_piece_idx] == 1:
                                pinned_piece_dict[pinned_piece_idx] = pidx

                    elif pinning_attackers == 'b':
                        for dir4 in [0, 1, 2, 3]:
                            # 0 = NORTHWEST, 1 = NORTHEAST, 2 = SOUTHEAST, 3 = SOUTHWEST
                            attacker_ray[dir4] = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pidx)
                        
                        king_pinned_from_NW_bishop = king_bishop_attack_rays[0] & attacker_ray[2] # use bishop's SE ray
                        king_pinned_from_NE_bishop = king_bishop_attack_rays[1] & attacker_ray[3] # use bishop's SW ray
                        king_pinned_from_SE_bishop = king_bishop_attack_rays[2] & attacker_ray[0] # use bishop's NW ray
                        king_pinned_from_SW_bishop = king_bishop_attack_rays[3] & attacker_ray[1] # use bishop's NE ray

                        combined = (
                            king_pinned_from_NW_bishop | 
                            king_pinned_from_NE_bishop | 
                            king_pinned_from_SE_bishop | 
                            king_pinned_from_SW_bishop
                        )
                        for pinned_piece_idx in combined.search(1):
                            if self.white_pieces[pinned_piece_idx] == 1:
                                pinned_piece_dict[pinned_piece_idx] = pidx

                    else:
                        for dir4 in [0, 1, 2, 3]:
                            # 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST, 4 = NORTHWEST, 5 = NORTHEAST, 6 = SOUTHEAST, 7 = SOUTHWEST
                            line_attacks = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pidx)
                            diag_attacks = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pidx)
                            attacker_ray[dir4] = line_attacks
                            attacker_ray[dir4+4] = diag_attacks
                        
                        king_pinned_from_N_queen = king_queen_attack_rays[0] & attacker_ray[2] 
                        king_pinned_from_E_queen = king_queen_attack_rays[1] & attacker_ray[3] 
                        king_pinned_from_S_queen = king_queen_attack_rays[2] & attacker_ray[0]
                        king_pinned_from_W_queen = king_queen_attack_rays[3] & attacker_ray[1]
                        king_pinned_from_NW_queen = king_queen_attack_rays[4] & attacker_ray[6]
                        king_pinned_from_NE_queen = king_queen_attack_rays[5] & attacker_ray[7]
                        king_pinned_from_SE_queen = king_queen_attack_rays[6] & attacker_ray[4]
                        king_pinned_from_SW_queen = king_queen_attack_rays[7] & attacker_ray[5]

                        combined = (
                            king_pinned_from_N_queen | 
                            king_pinned_from_E_queen | 
                            king_pinned_from_S_queen | 
                            king_pinned_from_W_queen |
                            king_pinned_from_NW_queen |
                            king_pinned_from_NE_queen |
                            king_pinned_from_SE_queen |
                            king_pinned_from_SW_queen
                        )
                        for pinned_piece_idx in combined.search(1):
                            if self.white_pieces[pinned_piece_idx] == 1:
                                pinned_piece_dict[pinned_piece_idx] = pidx

                    # print(f'\nPinned piece pos_idxs after {pinning_attackers}: {pinned_piece_pos_idxs}')
        
        elif self.white_move == -1:
            for pinning_attackers in ['R', 'B', 'Q']:
                for pidx in self.white_piece_bitboards[pinning_attackers].search(1):
                    attacker_ray = [zeros(64, endian='little') for _ in range(8)]

                    if pinning_attackers == 'R':
                        for dir4 in [0, 1, 2, 3]:
                            # 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST
                            attacker_ray[dir4] = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pidx)

                        king_pinned_from_northern_rook = king_rook_attack_rays[0] & attacker_ray[2] # use rook's southern ray
                        king_pinned_from_eastern_rook = king_rook_attack_rays[1] & attacker_ray[3] # use rook's western ray
                        king_pinned_from_southern_rook = king_rook_attack_rays[2] & attacker_ray[0] # use rook's northern ray
                        king_pinned_from_western_rook = king_rook_attack_rays[3] & attacker_ray[1] # use rook's eastern ray

                        combined = (
                            king_pinned_from_northern_rook | 
                            king_pinned_from_eastern_rook | 
                            king_pinned_from_southern_rook | 
                            king_pinned_from_western_rook
                        )
                        for pinned_piece_idx in combined.search(1):
                            if self.black_pieces[pinned_piece_idx] == 1:
                                pinned_piece_dict[pinned_piece_idx] = pidx

                    elif pinning_attackers == 'B':
                        for dir4 in [0, 1, 2, 3]:
                            # 0 = NORTHWEST, 1 = NORTHEAST, 2 = SOUTHEAST, 3 = SOUTHWEST
                            attacker_ray[dir4] = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pidx)
                        
                        king_pinned_from_NW_bishop = king_bishop_attack_rays[0] & attacker_ray[2] # use bishop's SE ray
                        king_pinned_from_NE_bishop = king_bishop_attack_rays[1] & attacker_ray[3] # use bishop's SW ray
                        king_pinned_from_SE_bishop = king_bishop_attack_rays[2] & attacker_ray[0] # use bishop's NW ray
                        king_pinned_from_SW_bishop = king_bishop_attack_rays[3] & attacker_ray[1] # use bishop's NE ray

                        combined = (
                            king_pinned_from_NW_bishop | 
                            king_pinned_from_NE_bishop | 
                            king_pinned_from_SE_bishop | 
                            king_pinned_from_SW_bishop
                        )
                        for pinned_piece_idx in combined.search(1):
                            if self.black_pieces[pinned_piece_idx] == 1:
                                pinned_piece_dict[pinned_piece_idx] = pidx

                    else:
                        for dir4 in [0, 1, 2, 3]:
                            # 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST, 4 = NORTHWEST, 5 = NORTHEAST, 6 = SOUTHEAST, 7 = SOUTHWEST
                            line_attacks = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pidx)
                            diag_attacks = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pidx)
                            attacker_ray[dir4] = line_attacks
                            attacker_ray[dir4+4] = diag_attacks
                        
                        king_pinned_from_N_queen = king_queen_attack_rays[0] & attacker_ray[2] 
                        king_pinned_from_E_queen = king_queen_attack_rays[1] & attacker_ray[3] 
                        king_pinned_from_S_queen = king_queen_attack_rays[2] & attacker_ray[0]
                        king_pinned_from_W_queen = king_queen_attack_rays[3] & attacker_ray[1]
                        king_pinned_from_NW_queen = king_queen_attack_rays[4] & attacker_ray[6]
                        king_pinned_from_NE_queen = king_queen_attack_rays[5] & attacker_ray[7]
                        king_pinned_from_SE_queen = king_queen_attack_rays[6] & attacker_ray[4]
                        king_pinned_from_SW_queen = king_queen_attack_rays[7] & attacker_ray[5]

                        combined = (
                            king_pinned_from_N_queen | 
                            king_pinned_from_E_queen | 
                            king_pinned_from_S_queen | 
                            king_pinned_from_W_queen |
                            king_pinned_from_NW_queen |
                            king_pinned_from_NE_queen |
                            king_pinned_from_SE_queen |
                            king_pinned_from_SW_queen
                        )
                        for pinned_piece_idx in combined.search(1):
                            if self.black_pieces[pinned_piece_idx] == 1:
                                pinned_piece_dict[pinned_piece_idx] = pidx

        return pinned_piece_dict


    def get_attackers_of_target_square(self, target_square_pos_idx):
        '''
        Given a target_square pos_idx, returns a bitboard with 1's representing all pieces of the opposite color threatening target_square
        '''
        square_rook_attack_ray = self.get_attack_ray_of_target_square(target_square_pos_idx, input_type='rook')
        square_bishop_attack_ray = self.get_attack_ray_of_target_square(target_square_pos_idx, input_type='bishop')
        square_queen_attack_ray = self.get_attack_ray_of_target_square(target_square_pos_idx, input_type='queen')
        square_knight_attack_ray = self.get_attack_ray_of_target_square(target_square_pos_idx, input_type='knight')
        square_king_attack_ray = self.get_attack_ray_of_target_square(target_square_pos_idx, input_type='king')

        if self.white_move == 1:
            square_pawn_attack_ray = self.get_attack_ray_of_target_square(target_square_pos_idx, input_type='white_pawn')

            rook_attackers = square_rook_attack_ray & self.black_rooks
            bishop_attackers = square_bishop_attack_ray & self.black_bishops
            queen_attackers = square_queen_attack_ray & self.black_queens
            pawn_attackers = square_pawn_attack_ray & self.black_pawns
            knight_attackers = square_knight_attack_ray & self.black_knights
            king_attackers = square_king_attack_ray & self.black_king

        elif self.white_move == -1:
            square_pawn_attack_ray = self.get_attack_ray_of_target_square(target_square_pos_idx, input_type='black_pawn')

            rook_attackers = square_rook_attack_ray & self.white_rooks
            bishop_attackers = square_bishop_attack_ray & self.white_bishops
            queen_attackers = square_queen_attack_ray & self.white_queens
            pawn_attackers = square_pawn_attack_ray & self.white_pawns
            knight_attackers = square_knight_attack_ray & self.white_knights
            king_attackers = square_king_attack_ray & self.white_king
        
        combined_attackers = rook_attackers | bishop_attackers | queen_attackers | pawn_attackers | knight_attackers | king_attackers
            
        return combined_attackers
    

    def get_between_source_target_ray(self, origin_square, target_square):
        '''
        Given an origin_square pos_idx and target_square pos_idx, returns the rank/file/diag/antidiag bitboard of the ray from source -> dest
        This function is used for sliding pieces only
        Exclusive origin_square, inclusive target_square
        '''
        piece_type = self.get_piece_of_square(origin_square)
        files = 'abcdefgh'

        if piece_type in ['R', 'B', 'Q', 'r', 'b', 'q']:
            filerank_o = pos_idx_to_filerank(origin_square)
            filerank_t = pos_idx_to_filerank(target_square)

            file_o, rank_o = filerank_o[0], filerank_o[1]
            file_t, rank_t = filerank_t[0], filerank_t[1]

            occupied_plus_target = self.occupied_squares.copy()
            occupied_plus_target[target_square] = 1

            exclusive_inclusive_ray = zeros(64, endian='little')

            if piece_type == 'R' or piece_type == 'r':
                if file_o == file_t and int(rank_o) < int(rank_t): # NORTH RAY
                    mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 0, origin_square)
                    exclusive_inclusive_ray = mask & self.file[file_o]
                elif files.index(file_o) < files.index(file_t) and rank_o == rank_t: # EAST RAY
                     mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 1, origin_square)
                     exclusive_inclusive_ray = mask & self.rank[rank_o]
                elif file_o == file_t and int(rank_o) > int(rank_t): # SOUTH RAY
                     mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 2, origin_square)
                     exclusive_inclusive_ray = mask & self.file[file_o]
                elif files.index(file_o) > files.index(file_t) and rank_o == rank_t: # WEST RAY
                     mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 3, origin_square)
                     exclusive_inclusive_ray = mask & self.rank[rank_o]

            elif piece_type == 'B' or piece_type == 'b':
                # 0 = NORTHWEST, 1 = NORTHEAST, 2 = SOUTHEAST, 3 = SOUTHWEST
                if files.index(file_o) > files.index(file_t) and int(rank_o) < int(rank_t): # NORTHWEST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 0, origin_square)
                    exclusive_inclusive_ray = mask
                elif files.index(file_o) < files.index(file_t) and int(rank_o) < int(rank_t): # NORTHEAST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 1, origin_square)
                    exclusive_inclusive_ray = mask
                elif files.index(file_o) < files.index(file_t) and int(rank_o) > int(rank_t): # SOUTHEAST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 2, origin_square)
                    exclusive_inclusive_ray = mask
                elif files.index(file_o) > files.index(file_t) and int(rank_o) > int(rank_t): # SOUTHWEST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 3, origin_square)
                    exclusive_inclusive_ray = mask

            elif piece_type == 'Q' or piece_type == 'q':
                if file_o == file_t and int(rank_o) < int(rank_t): # NORTH RAY
                    mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 0, origin_square)
                    exclusive_inclusive_ray = mask & self.file[file_o]
                elif files.index(file_o) < files.index(file_t) and rank_o == rank_t: # EAST RAY
                     mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 1, origin_square)
                     exclusive_inclusive_ray = mask & self.rank[rank_o]
                elif file_o == file_t and int(rank_o) > int(rank_t): # SOUTH RAY
                     mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 2, origin_square)
                     exclusive_inclusive_ray = mask & self.file[file_o]
                elif files.index(file_o) > files.index(file_t) and rank_o == rank_t: # WEST RAY
                     mask = getRookRayAttacks(self.rook_rays, occupied_plus_target, 3, origin_square)
                     exclusive_inclusive_ray = mask & self.rank[rank_o]
                elif files.index(file_o) > files.index(file_t) and int(rank_o) < int(rank_t): # NORTHWEST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 0, origin_square)
                    exclusive_inclusive_ray = mask
                elif files.index(file_o) < files.index(file_t) and int(rank_o) < int(rank_t): # NORTHEAST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 1, origin_square)
                    exclusive_inclusive_ray = mask
                elif files.index(file_o) < files.index(file_t) and int(rank_o) > int(rank_t): # SOUTHEAST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 2, origin_square)
                    exclusive_inclusive_ray = mask
                elif files.index(file_o) > files.index(file_t) and int(rank_o) > int(rank_t): # SOUTHWEST RAY
                    mask = getBishopRayAttacks(self.bishop_rays, occupied_plus_target, 3, origin_square)
                    exclusive_inclusive_ray = mask
                
        return exclusive_inclusive_ray


    def get_legal_moves(self):
        '''
        Case 1: King is in single check
            Option 1 = Move the king out of check
            Option 2 = Capture the checking piece
            Option 3 = Block the checking piece (if checking piece is a sliding piece)

        Case 2: King is in double check
            Option 1 = Move the king out of check

        Case 3: King not in check, but making the move would result in check
                i.e., the absolute pinned piece case
            Option 1 = Move the king somewhere legal
            Option 2 = Move the pinned piece along the ray
        
        Case 4: King not in check, no absolutely pinned pieces
            Option 1 = Move the king somewhere legal
            Option 2 = Move any of the other pieces for any of their pseudolegal moves
        
        ---------------------------------------------------------------------------------------------

        1.) For pseudolegal moves that move the King:
            a.) combined = attacked_by_rooks | ... | attacked_by_king
            b.) If not combined.any(): # there does not exist a piece that attacks that square
                    legal_encoded_moves.append(pseudolegal_move)
        
        2.) For pseudolegal moves that move anything not the King:
            a.) determine if the king is in check or not with attackers
                i.) if len(attackers.search(1)) >= 2: # DOUBLE CHECK
                        # illegal move, can only move the king here, so continue
                        continue
                        
                ii.) if len(attackers.search(1)) == 1: # SINGLE CHECK
                        # pseudolegal move = legal move IF 
                        # 1.) it captures the attacker
                        # 2.) it blocks the incoming attack if the attacker is a sliding piece
                        # 3.) doing (1) or (2) does not expose the king to an attack, i.e. absolutely pinned

                iii.) if len(attackers.search(1)) == 0: # NOT IN CHECK
                        # 1.) the piece is not absolutely pinned:
                        #       pseudolegal move = legal move
                        # 2.) the piece is absolutely pinned:
                        #       pseudolegal move = legal move IF it moves along the attacking ray

        '''
        legal_encoded_moves = []
        pseudolegal_encoded_moves = self.get_pseudolegal_moves()

        # SETUP KING CHECK BITBOARDS
        if self.white_move == 1:
            king_pos = self.white_king.search(1)[0]
            sliding_types = ['r', 'b', 'q']

            enemy_attackers_of_king = self.get_attackers_of_target_square(king_pos) # enemy = for check computing
            for pidx in enemy_attackers_of_king.search(1):
                if self.white_pieces[pidx] == 1:
                    enemy_attackers_of_king[pidx] = 0

        elif self.white_move == -1:
            king_pos = self.black_king.search(1)[0]
            sliding_types = ['R', 'B', 'Q']

            enemy_attackers_of_king = self.get_attackers_of_target_square(king_pos)
            for pidx in enemy_attackers_of_king.search(1):
                if self.black_pieces[pidx] == 1:
                    enemy_attackers_of_king[pidx] = 0
        

        pinned_piece_dict = self.get_pinned_piece_dict(king_pos)
        pinned_piece_idxs = list(pinned_piece_dict.keys())
        pinned_piece_bitboard = zeros(64, endian='little')
        pinned_piece_bitboard[pinned_piece_idxs] = 1
        
        num_attackers = len(enemy_attackers_of_king.search(1))

        print(f'Number of attackers: {num_attackers}')


        # CONVERTING MOVES
        for encoded_move in pseudolegal_encoded_moves:
            origin_square, target_square, promotion_piece_type, special_move_flag = decode_move(encoded_move)
            moving_piece_type = self.get_piece_of_square(origin_square)

            # KING MOVES
            if moving_piece_type == 'K' or moving_piece_type == 'k':
                attackers = self.get_attackers_of_target_square(target_square)

                if self.white_move == 1: # filter out friendly "attackers"
                    for pidx in attackers.search(1):
                        if self.white_pieces[pidx] == 1:
                            attackers[pidx] = 0
                elif self.white_move == -1:
                    for pidx in attackers.search(1):
                        if self.black_pieces[pidx] == 1:
                            attackers[pidx] = 0

                if not attackers.any(): # there does not exist a piece that attacks target_square
                    legal_encoded_moves.append(encoded_move) # i.e., can move the King there
            
            # ANY OTHER PIECE MOVES
            else:     
                if num_attackers >= 2: # DOUBLE CHECK
                    continue

                elif num_attackers == 1: # SINGLE CHECK
                    attacker_pos_idx = enemy_attackers_of_king.search(1)[0]
                    attacker_piece_type = self.get_piece_of_square(attacker_pos_idx)
                    attacker_ray_bitboard = self.get_between_source_target_ray(attacker_pos_idx, king_pos) # exclusive, inclusive
                    # print(f'Move: {origin_square ^ 56} -> {target_square ^ 56}')
                    # lerf_bitboard_to_2D_numpy(pinned_piece_bitboard)
                    # lerf_bitboard_to_2D_numpy(attacker_ray_bitboard)

                    if pinned_piece_bitboard[origin_square] == 0: # two conditions for legality IF the moving piece isn't a pinned piece
                        if target_square == attacker_pos_idx: # pseudolegal = legal if can capture, or 
                            legal_encoded_moves.append(encoded_move)
                        elif attacker_ray_bitboard[target_square] == 1: # we can move it to block the attacker
                            legal_encoded_moves.append(encoded_move) # if target_sq in the attacker_ray, it's blocking
                    else: # if it is a pinned piece, gg just skip
                        continue

                else: # NOT IN CHECK, need to check if moving piece is absolutely pinned
                    if pinned_piece_bitboard[origin_square] == 1: # if a pinned piece, can only move along the pinned ray
                        attacker_pos_idx = pinned_piece_dict[origin_square] # get idx of the attacking piece that is pinning the move piece
                        attacker_ray_bitboard = self.get_between_source_target_ray(attacker_pos_idx, king_pos)

                        if attacker_ray_bitboard[target_square] == 1: # if the move is within the ray, it's legal, otherwise not legal
                            legal_encoded_moves.append(encoded_move)

                    else: # if not pinned, pseudolegal = legal
                        legal_encoded_moves.append(encoded_move)

        return legal_encoded_moves
    

    def get_pseudolegal_moves(self):
        encoded_moves = []

        if self.white_move == 1:
            if self.white_pawns.any():
                encoded_moves += self.generate_pseudolegal_pawn_moves()
            if self.white_knights.any():
                encoded_moves += self.generate_pseudolegal_knight_moves()
            if self.white_king.any():
                encoded_moves += self.generate_pseudolegal_king_moves()

            if self.white_rooks.any():
                encoded_moves += self.generate_pseudolegal_rook_moves()
            if self.white_bishops.any():
                encoded_moves += self.generate_pseudolegal_bishop_moves()
            if self.white_queens.any():
                encoded_moves += self.generate_pseudolegal_queen_moves()

        elif self.white_move == -1:
            if self.black_pawns.any():
                encoded_moves += self.generate_pseudolegal_pawn_moves()
            if self.black_knights.any():
                encoded_moves += self.generate_pseudolegal_knight_moves()
            if self.black_king.any():
                encoded_moves += self.generate_pseudolegal_king_moves()

            if self.black_rooks.any():
                encoded_moves += self.generate_pseudolegal_rook_moves()
            if self.black_bishops.any():
                encoded_moves += self.generate_pseudolegal_bishop_moves()
            if self.black_queens.any():
                encoded_moves += self.generate_pseudolegal_queen_moves()

        return encoded_moves


    def generate_pseudolegal_queen_moves(self):
        encoded_queen_moves = []

        if self.white_move == 1:
            for pos_idx in self.white_queens.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                for dir4 in [0, 1, 2, 3]:
                    line_attacks = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pos_idx)
                    diag_attacks = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pos_idx)
                    queen_attacks = line_attacks | diag_attacks

                    for target_pos_idx in queen_attacks.search(1):
                        if self.white_pieces[target_pos_idx] == 1:
                            continue
                        target_squares.append(pos_idx_to_bitarray(target_pos_idx, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_queen_moves.append(encoded_move)
        
        elif self.white_move == -1:
            for pos_idx in self.black_queens.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                for dir4 in [0, 1, 2, 3]:
                    line_attacks = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pos_idx)
                    diag_attacks = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pos_idx)
                    queen_attacks = line_attacks | diag_attacks

                    for target_pos_idx in queen_attacks.search(1):
                        if self.black_pieces[target_pos_idx] == 1:
                            continue
                        target_squares.append(pos_idx_to_bitarray(target_pos_idx, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_queen_moves.append(encoded_move)

        return encoded_queen_moves


    def generate_pseudolegal_bishop_moves(self):
        encoded_bishop_moves = []

        if self.white_move == 1:
            for pos_idx in self.white_bishops.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                for dir4 in [0, 1, 2, 3]:
                    bishop_attacks = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pos_idx)
                    for target_pos_idx in bishop_attacks.search(1):
                        if self.white_pieces[target_pos_idx] == 1:
                            continue
                        target_squares.append(pos_idx_to_bitarray(target_pos_idx, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_bishop_moves.append(encoded_move)
        
        elif self.white_move == -1:
            for pos_idx in self.black_bishops.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                for dir4 in [0, 1, 2, 3]:
                    bishop_attacks = getBishopRayAttacks(self.bishop_rays, self.occupied_squares, dir4, pos_idx)
                    for target_pos_idx in bishop_attacks.search(1):
                        if self.black_pieces[target_pos_idx] == 1:
                            continue
                        target_squares.append(pos_idx_to_bitarray(target_pos_idx, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_bishop_moves.append(encoded_move)
        
        return encoded_bishop_moves


    def generate_pseudolegal_rook_moves(self):
        encoded_rook_moves = []

        if self.white_move == 1:
            for pos_idx in self.white_rooks.search(1):
                # print(f'pos_idx: {pos_idx}, lerf_idx: {pos_idx ^ 56}')
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                for dir4 in [0, 1, 2, 3]:
                    rook_attacks = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pos_idx)
                    for target_pos_idx in rook_attacks.search(1):
                        if self.white_pieces[target_pos_idx] == 1:
                            continue
                        target_squares.append(pos_idx_to_bitarray(target_pos_idx, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_rook_moves.append(encoded_move)
                
        elif self.white_move == -1:
            for pos_idx in self.black_rooks.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                for dir4 in [0, 1, 2, 3]:
                    rook_attacks = getRookRayAttacks(self.rook_rays, self.occupied_squares, dir4, pos_idx)
                    for target_pos_idx in rook_attacks.search(1):
                        if self.black_pieces[target_pos_idx] == 1:
                            continue
                        target_squares.append(pos_idx_to_bitarray(target_pos_idx, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_rook_moves.append(encoded_move)
        
        return encoded_rook_moves


    def generate_pseudolegal_king_moves(self):
        encoded_king_moves = []

        if self.white_move == 1:
            for pos_idx in self.white_king.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                if ( # north
                    (not self.rank['8'][pos_idx]) and 
                    bool((self.empty_squares[pos_idx - 8] or self.black_pieces[pos_idx - 8]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 8, length=6))
                if ( # northeast
                    (not self.rank['8'][pos_idx]) and 
                    (not self.file['h'][pos_idx]) and
                    bool((self.empty_squares[pos_idx - 7] or self.black_pieces[pos_idx - 7]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 7, length=6))
                if ( # east
                    (not self.file['h'][pos_idx]) and 
                    bool((self.empty_squares[pos_idx + 1] or self.black_pieces[pos_idx + 1]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 1, length=6))
                if ( # southeast
                    (not self.rank['1'][pos_idx]) and 
                    (not self.file['h'][pos_idx]) and
                    bool((self.empty_squares[pos_idx + 9] or self.black_pieces[pos_idx + 9]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 9, length=6))
                if ( # south
                    (not self.rank['1'][pos_idx]) and 
                    bool((self.empty_squares[pos_idx + 8] or self.black_pieces[pos_idx + 8]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 8, length=6))
                if ( # southwest
                    (not self.rank['1'][pos_idx]) and 
                    (not self.file['a'][pos_idx]) and
                    bool((self.empty_squares[pos_idx + 7] or self.black_pieces[pos_idx + 7]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 7, length=6))
                if ( # west
                    (not self.file['a'][pos_idx]) and
                    bool((self.empty_squares[pos_idx - 1] or self.black_pieces[pos_idx - 1]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 1, length=6))
                if ( # northwest
                    (not self.rank['8'][pos_idx]) and 
                    (not self.file['a'][pos_idx]) and
                    bool((self.empty_squares[pos_idx - 9] or self.black_pieces[pos_idx - 9]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 9, length=6))


                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_king_moves.append(encoded_move)

        elif self.white_move == -1:
            for pos_idx in self.black_king.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                if ( # north
                    (not self.rank['8'][pos_idx]) and 
                    bool((self.empty_squares[pos_idx - 8] or self.white_pieces[pos_idx - 8]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 8, length=6))
                if ( # northeast
                    (not self.rank['8'][pos_idx]) and 
                    (not self.file['h'][pos_idx]) and
                    bool((self.empty_squares[pos_idx - 7] or self.white_pieces[pos_idx - 7]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 7, length=6))
                if ( # east
                    (not self.file['h'][pos_idx]) and 
                    bool((self.empty_squares[pos_idx + 1] or self.white_pieces[pos_idx + 1]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 1, length=6))
                if ( # southeast
                    (not self.rank['1'][pos_idx]) and 
                    (not self.file['h'][pos_idx]) and
                    bool((self.empty_squares[pos_idx + 9] or self.white_pieces[pos_idx + 9]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 9, length=6))
                if ( # south
                    (not self.rank['1'][pos_idx]) and 
                    bool((self.empty_squares[pos_idx + 8] or self.white_pieces[pos_idx + 8]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 8, length=6))
                if ( # southwest
                    (not self.rank['1'][pos_idx]) and 
                    (not self.file['a'][pos_idx]) and
                    bool((self.empty_squares[pos_idx + 7] or self.white_pieces[pos_idx + 7]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx + 7, length=6))
                if ( # west
                    (not self.file['a'][pos_idx]) and
                    bool((self.empty_squares[pos_idx - 1] or self.white_pieces[pos_idx - 1]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 1, length=6))
                if ( # northwest
                    (not self.rank['8'][pos_idx]) and 
                    (not self.file['a'][pos_idx]) and
                    bool((self.empty_squares[pos_idx - 9] or self.white_pieces[pos_idx - 9]))
                ):
                    target_squares.append(pos_idx_to_bitarray(pos_idx - 9, length=6))


                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                        
                    encoded_king_moves.append(encoded_move)
        
        return encoded_king_moves

    
    def generate_pseudolegal_knight_moves(self):
        encoded_knight_moves = []

        if self.white_move == 1:
            for pos_idx in self.white_knights.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                # 6:  NWW = ~GH_File, SEE = ~AB_File
                # 15: NNW =  ~H_File, SSE =  ~A_File
                # 17: NNE =  ~A_File, SSW =  ~H_File
                # 10: NEE = ~AB_File, SWW = ~GH_File
                wraparound_masks = {
                    6: ((self.file['g'] | self.file['h']), (self.file['a'] | self.file['b'])),
                    15: (self.file['h'], self.file['a']),
                    17: (self.file['a'], self.file['h']),
                    10: ((self.file['a'] | self.file['b']), (self.file['g'] | self.file['h']))
                }

                # QUIET AND CAPTURE MOVE LOGIC
                for shift in [6, 15, 17, 10]:
                    wraparound_file_north, wraparound_file_south = wraparound_masks[shift]

                    # NORTH LOGIC
                    if (
                        (pos_idx - shift >= 0) and # handles going out of range
                        (not wraparound_file_north[pos_idx]) and # handles wraparound
                        bool((self.empty_squares[pos_idx - shift] or self.black_pieces[pos_idx - shift])) # handles valid move
                    ): 
                        target_squares.append(pos_idx_to_bitarray(pos_idx - shift, length=6))
                    
                    # SOUTH LOGIC
                    if (
                        (pos_idx + shift <= 63) and
                        (not wraparound_file_south[pos_idx]) and 
                        bool((self.empty_squares[pos_idx + shift] or self.black_pieces[pos_idx + shift]))
                    ):
                        target_squares.append(pos_idx_to_bitarray(pos_idx + shift, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                    
                    encoded_knight_moves.append(encoded_move)

        elif self.white_move == -1:
            for pos_idx in self.black_knights.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                # 6:  NWW = ~GH_File, SEE = ~AB_File
                # 15: NNW =  ~H_File, SSE =  ~A_File
                # 17: NNE =  ~A_File, SSW =  ~H_File
                # 10: NEE = ~AB_File, SWW = ~GH_File
                wraparound_masks = {
                    6: ((self.file['a'] | self.file['b']), (self.file['g'] | self.file['h'])),
                    15: (self.file['a'], self.file['h']),
                    17: (self.file['h'], self.file['a']),
                    10: ((self.file['g'] | self.file['h']), (self.file['a'] | self.file['b']))
                }

                # QUIET AND CAPTURE MOVE LOGIC
                for shift in [6, 15, 17, 10]:
                    wraparound_file_north, wraparound_file_south = wraparound_masks[shift]
                    
                    # NORTH LOGIC
                    if (
                        (pos_idx + shift <= 63) and
                        (not wraparound_file_north[pos_idx]) and 
                        bool((self.empty_squares[pos_idx + shift] or self.white_pieces[pos_idx + shift]))
                    ):
                        target_squares.append(pos_idx_to_bitarray(pos_idx + shift, length=6))
                    
                    # SOUTH LOGIC
                    if (
                        (pos_idx - shift >= 0) and # handles going out of range
                        (not wraparound_file_south[pos_idx]) and # handles wraparound
                        bool((self.empty_squares[pos_idx - shift] or self.white_pieces[pos_idx - shift])) # handles valid move
                    ): 
                        target_squares.append(pos_idx_to_bitarray(pos_idx - shift, length=6))

                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                    
                    encoded_knight_moves.append(encoded_move)
        
        return encoded_knight_moves
            

    def generate_pseudolegal_pawn_moves(self):
        '''
        Returns list of 16-bit encoded moves (pushes and captures)
        - Move Representation: 16-bit bitarray representation
            - Bits   0-5: LERF Destination Square (0 - 63)
            - Bits  6-11: LERF Origin Square (0 - 63)
            - Bits 12-13: Promotion piece type (00 = Rook, 01 = Knight, 10 = Bishop, 11 = Queen)
            - Bits 14-15: Special move flag (01 = promotion, 10 = en passant, 11 = castling)
        '''
        encoded_pawn_moves = []
        files = 'abcdefgh'
        
        # Loop through indices that white_pawns == 1
        if self.white_move == 1:
            for pos_idx in self.white_pawns.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                filerank = pos_idx_to_filerank(pos_idx)
                file, rank = filerank[0], filerank[1]
                
                if ~self.rank['8'][pos_idx]: 
                    # QUIET MOVE LOGIC
                    if self.empty_squares[pos_idx - 8]: # SINGLE STEP
                        target_squares.append(pos_idx_to_bitarray(pos_idx - 8, length=6))
                    if self.rank['2'][pos_idx]:
                        if self.empty_squares[pos_idx - 8] and self.empty_squares[pos_idx - 16]: # DOUBLE STEP
                            target_squares.append(pos_idx_to_bitarray(pos_idx - 16, length=6))
                    
                    # CAPTURE MOVE LOGIC
                    if not self.file['a'][pos_idx] and self.black_pieces[pos_idx - 9]: # LEFT CAPTURE
                        target_squares.append(pos_idx_to_bitarray(pos_idx - 9, length=6))
                    if not self.file['h'][pos_idx] and self.black_pieces[pos_idx - 7]: # RIGHT CAPTURE
                        target_squares.append(pos_idx_to_bitarray(pos_idx - 7, length=6))
                
                # EN PASSANT LOGIC, en_passant_target_pos_idx = en passant target square of black pawn
                if self.en_passant_target_pos_idx != -1:
                    filerank_target = pos_idx_to_filerank(self.en_passant_target_pos_idx)
                    file_t, rank_t = filerank_target[0], filerank_target[1]
                    
                    is_adjacent = (abs(files.index(file) - files.index(file_t)) == 1)

                    if (
                        (int(rank) == (int(rank_t) - 1)) and # check for same rank
                        (is_adjacent == True) # check for adjacent file
                    ): 
                        encoded_move = zeros(16, endian='big')
                        encoded_move[0:2] = int2ba(2, 2, endian='big') # EN PASSANT FLAG = 10
                        encoded_move[4:10] = pos_idx_to_bitarray(self.en_passant_target_pos_idx, length=6)
                        encoded_move[10:16] = origin_square
                        encoded_pawn_moves.append(encoded_move)

                # REGULAR MOVE GEN LOGIC
                for target_square in target_squares:
                    # ENCODE PROMOTION MOVES
                    if (bitarray_to_pos_idx(target_square) in range(0, 8)):
                        for i in range(0, 4):  # Loop over promotion piece types
                            encoded_move = zeros(16, endian='big')
                            encoded_move[0:2] = int2ba(1, 2, endian='big')  # PROMOTION FLAG = 01
                            encoded_move[2:4] = int2ba(i, 2, endian='big')  # PROMOTION PIECE TYPE FLAG
                            encoded_move[4:10] = target_square
                            encoded_move[10:16] = origin_square
                            encoded_pawn_moves.append(encoded_move)
                    # ENCODE NON-PROMOTION MOVES
                    else:
                        encoded_move = zeros(16, endian='big')
                        encoded_move[4:10] = target_square
                        encoded_move[10:16] = origin_square
                        encoded_pawn_moves.append(encoded_move)
                    
        elif self.white_move == -1:
            for pos_idx in self.black_pawns.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

                filerank = pos_idx_to_filerank(pos_idx)
                file, rank = filerank[0], filerank[1]

                if not self.rank['1'][pos_idx]: 
                    # QUIET MOVE LOGIC
                    if self.empty_squares[pos_idx + 8]: # SINGLE STEP
                        target_squares.append(pos_idx_to_bitarray(pos_idx + 8, length=6))
                    if self.rank['7'][pos_idx]:
                        if self.empty_squares[pos_idx + 8] and self.empty_squares[pos_idx + 16]: # DOUBLE STEP
                            target_squares.append(pos_idx_to_bitarray(pos_idx + 16, length=6))

                    # CAPTURE MOVE LOGIC
                    if not self.file['a'][pos_idx] and self.white_pieces[pos_idx + 7]: # LEFT CAPTURE
                        target_squares.append(pos_idx_to_bitarray(pos_idx + 7, length=6))
                    if not self.file['h'][pos_idx] and self.white_pieces[pos_idx + 9]: # RIGHT CAPTURE
                        target_squares.append(pos_idx_to_bitarray(pos_idx + 9, length=6))

                # EN PASSANT LOGIC, en_passant_target_pos_idx = en passant target square of white pawn
                if self.en_passant_target_pos_idx != -1:
                    filerank_target = pos_idx_to_filerank(self.en_passant_target_pos_idx)
                    file_t, rank_t = filerank_target[0], filerank_target[1]
                    
                    is_adjacent = (abs(files.index(file) - files.index(file_t)) == 1)

                    if (
                        (int(rank) == (int(rank_t) + 1)) and # check for same rank
                        (is_adjacent == True) # check for adjacent file
                    ): 
                        encoded_move = zeros(16, endian='big')
                        encoded_move[0:2] = int2ba(2, 2, endian='big') # EN PASSANT FLAG = 10
                        encoded_move[4:10] = pos_idx_to_bitarray(self.en_passant_target_pos_idx, length=6)
                        encoded_move[10:16] = origin_square
                        encoded_pawn_moves.append(encoded_move)

                # REGULAR MOVE GEN LOGIC
                for target_square in target_squares:
                    # ENCODE PROMOTION MOVES
                    if (bitarray_to_pos_idx(target_square) in range(56, 64)):
                        for i in range(0, 4):  # Loop over promotion piece types
                            encoded_move = zeros(16, endian='big')
                            encoded_move[0:2] = int2ba(1, 2, endian='big')  # PROMOTION FLAG = 01
                            encoded_move[2:4] = int2ba(i, 2, endian='big')  # PROMOTION PIECE TYPE FLAG
                            encoded_move[4:10] = target_square
                            encoded_move[10:16] = origin_square
                            encoded_pawn_moves.append(encoded_move)
                    # ENCODE NON-PROMOTION MOVES
                    else:
                        encoded_move = zeros(16, endian='big')
                        encoded_move[4:10] = target_square
                        encoded_move[10:16] = origin_square
                        encoded_pawn_moves.append(encoded_move)
                    
        return encoded_pawn_moves
    
    # ----------------------------------- ARCADE HELPER FUNCTIONS -----------------------------------------------------------------------------------------------
    def is_piece_selectable(self, filerank):
        pos_idx = filerank_to_pos_idx(filerank)
        piece_type = self.get_piece_of_square(pos_idx)

        if self.white_move == 1:
            # print(f'White move, {piece_type} @ {pos_idx} @ {filerank}')
            return (piece_type in self.white_piece_bitboards.keys())

        elif self.white_move == -1:
            # print(f'Black move, {piece_type} @ {pos_idx} @ {filerank}')
            return (piece_type in self.black_piece_bitboards.keys())
