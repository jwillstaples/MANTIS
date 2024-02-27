
import numpy as np

from prettytable import PrettyTable
from bitarray.util import zeros, pprint

from common.board import BlankBoard
from chess.utils import *

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
        
        # Initialize bitboards for ranks and files
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

        # Initialize game state variables
        self.move_history = []
        self.white_move = white_move # 1 = White's move, -1 = Black's move
        self.turn_counter = 0
    

    # ----------------------------------- BOARD UPDATE METHODS ------------------------------------------------------------------------------------------------------
    def make_move(self, encoded_move):
        '''
        encoded_move: LEGAL encoded move bitarray
        '''
        origin_square, target_square, promotion_piece_type, special_move_flag = decode_move(encoded_move)
        
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
                
                
            # Finally, set the val of the piece type bitboard at idx target_square = 1
            # represents the piece moving onto target_square
            origin_bitboard[target_square] = 1
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
                
            origin_bitboard[target_square] = 1
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
        return None
    
        
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
    def get_pseudolegal_moves(self):
        encoded_moves = []

        if self.white_move == 1:
            if len(self.white_pawns.search(1)) >= 1:
                encoded_moves += self.generate_pseudolegal_pawn_moves()
            if len(self.white_knights.search(1)) >= 1:
                encoded_moves += self.generate_pseudolegal_knight_moves()
            if len(self.white_king.search(1)) >= 1:
                encoded_moves += self.generate_pseudolegal_king_moves()
        elif self.white_move == -1:
            if len(self.black_pawns.search(1)) >= 1:
                encoded_moves += self.generate_pseudolegal_pawn_moves()
            if len(self.black_knights.search(1)) >= 1:
                encoded_moves += self.generate_pseudolegal_knight_moves()
            if len(self.black_king.search(1)) >= 1:
                encoded_moves += self.generate_pseudolegal_king_moves()

        return encoded_moves

    
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
        
        # Loop through indices that white_pawns == 1
        if self.white_move == 1:
            for pos_idx in self.white_pawns.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []
                
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
                
                for target_square in target_squares:
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = target_square
                    encoded_move[10:16] = origin_square
                    
                    encoded_pawn_moves.append(encoded_move)
                    
        elif self.white_move == -1:
            for pos_idx in self.black_pawns.search(1):
                origin_square = pos_idx_to_bitarray(pos_idx, length=6)
                target_squares = []

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

                for target_square in target_squares:
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
