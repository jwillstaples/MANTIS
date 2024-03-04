import numpy as np

from bitarray import bitarray
from bitarray.util import ba2int, int2ba, zeros


def hex_to_bitarray(hex_string, endian='little'):
    '''
    Given a 16-bit hex string board representation, returns a 64-bit binary string representation
    '''
    return bitarray(bin(int(hex_string, 16))[2:].zfill(64), endian=endian)


def lerf_bitboard_to_1D_numpy(bitboard):
    '''
    Given a length-64 bitboard board representation, returns a length 64 NumPy array representation
    '''
    return np.array(bitboard.tolist())


def lerf_bitboard_to_2D_numpy(bitboard):
    print(lerf_bitboard_to_1D_numpy(bitboard).reshape((8, 8)))


def pos_idx_to_bitarray(pos_idx, length):
    '''
    So a LERF bitarray has the LSB = 0, but python slicing makes the LSB = 63
    So when you do bitboard.search(1), you'll get a python slice'd version of the 1 bit indices
    This function first converts it back into the LERF position index, so back where LSB = 0
    And then it converts the LERF position index into a big-endian binary of length=length
    Example: 
        rank['1'] has 1 bits for LERF positions 0-7.
        rank['1'].search(1) will thus return [56, ..., 63]
        calling ^ 56 on each of these will turn it back into [0, ..., 7]
        calling int2ba on these ints w/ a length=6 will turn them into [000000, ..., 000111] bitarrays
    '''
    lerf_position = pos_idx ^ 56
    return int2ba(lerf_position, length=length, endian='big')


def bitarray_to_pos_idx(bitarr):
    '''
    Given a LERF position bitarray, returns the python slicing index
    Example:
        Given the bitarray for LERF position 7 = bitarray('000111')
        lerf_position = 7
        returns 55
    '''
    lerf_position = ba2int(bitarr)
    return lerf_position ^ 56


def pos_idx_to_filerank(pos_idx):
    '''
    Given a python slicing index position, returns the corresponding LERF {file}{rank} string 
    '''
    lerf_idx = pos_idx ^ 56
    file_letters = 'abcdefgh'
    
    file = file_letters[(lerf_idx % 8)]
    rank = (lerf_idx // 8) + 1
    
    return file + str(rank)


def filerank_to_pos_idx(filerank):
    file_letters = 'abcdefgh'
    
    file = filerank[0]
    rank = int(filerank[1])
    
    file_idx = file_letters.index(file)
    rank_idx = rank - 1
    
    lerf_idx = (rank_idx * 8) + file_idx
    pos_idx = lerf_idx ^ 56
    
    return pos_idx


def decode_move(move_bitarray):
    '''
    Given a 16-bit encoded move, decodes it into the bit flags
    '''
    special_move_flag = bitarray_to_pos_idx(move_bitarray[0:2]) ^ 56
    promotion_piece_type = bitarray_to_pos_idx(move_bitarray[2:4]) ^ 56
    target_square = bitarray_to_pos_idx(move_bitarray[4:10])
    origin_square = bitarray_to_pos_idx(move_bitarray[10:16])
    
    return (origin_square, target_square, promotion_piece_type, special_move_flag)


def getRookRayAttacks(rook_rays, occupied_squares, dir4, pos_idx):
    '''
    rook_rays (bitboard): precomputed 2D array of rook attack rays
    occupied_squares (bitboard): current game state's occupied squares bitboard
    dir4 (int): 0 = North, 1 = East, 2 = South, 3 = West
    pos_idx (int): the pos_idx of the rook's square

    attacks (bitboard): bitboard with 1's for valid moves, considers the nearest blocker an enemy
    need to mask attacks with friendly pieces bitboard
    '''
    attacks = rook_rays[dir4][pos_idx].copy()
    blocker = attacks & occupied_squares
    if blocker.any():
        filerank = pos_idx_to_filerank(pos_idx)
        file, rank = filerank[0], filerank[1]

        if dir4 == 0: # NORTH
            start_pos_idx = 0
            end_pos_idx = pos_idx
            takelast = True
        elif dir4 == 1: # EAST
            start_pos_idx = pos_idx
            end_pos_idx = (((int(rank) * 8) - 1) ^ 56) + 1
            takelast = False
        elif dir4 == 2: # SOUTH
            start_pos_idx = pos_idx
            end_pos_idx = 64
            takelast = False
        elif dir4 == 3: # WEST
            start_pos_idx = ((int(rank) - 1) * 8) ^ 56
            end_pos_idx = pos_idx
            takelast = True
            
        nearest_blocker_pos_idx = blocker.find(1, start_pos_idx, end_pos_idx, takelast)
        attacks ^= rook_rays[dir4][nearest_blocker_pos_idx]
    
    return attacks

def getBishopRayAttacks(bishop_rays, occupied_squares, dir4, pos_idx):
    '''
    dir4 (int): 0 = NORTHWEST, 1 = NORTHEAST, 2 = SOUTHEAST, 3 = SOUTHWEST
    '''
    attacks = bishop_rays[dir4][pos_idx].copy()
    blocker = attacks & occupied_squares
    if blocker.any():
        #  pos_idx:    0  1  2  3  4  5  6  7   ...   56 57 58 59 60 61 62 63
        # lerf_idx: [[56,57,58,59,60,61,62,63], ..., [ 0, 1, 2, 3, 4, 5, 6, 7]]
        if dir4 == 0: # NORTHWEST
            start_pos_idx = 0
            end_pos_idx = pos_idx
            takelast = True
        elif dir4 == 1: # NORTHEAST
            start_pos_idx = 0
            end_pos_idx = pos_idx
            takelast = True
        elif dir4 == 2: # SOUTHEAST
            start_pos_idx = pos_idx
            end_pos_idx = 64
            takelast = False
        elif dir4 == 3: # SOUTHWEST
            start_pos_idx = pos_idx
            end_pos_idx = 64
            takelast = False

        nearest_blocker_pos_idx = blocker.find(1, start_pos_idx, end_pos_idx, takelast)
        attacks ^= bishop_rays[dir4][nearest_blocker_pos_idx]
    
    return attacks


def decode_fen_string(fen):
    fen_gamestate = {}
    pieces, active_color, castling, en_passant, halfmove, fullmove = fen.split(' ')

    piece_bitboards = {
        'P': zeros(64, endian='little'),
        'R': zeros(64, endian='little'), 
        'N': zeros(64, endian='little'),
        'B': zeros(64, endian='little'),
        'Q': zeros(64, endian='little'),
        'K': zeros(64, endian='little'),
        'p': zeros(64, endian='little'),
        'r': zeros(64, endian='little'),
        'n': zeros(64, endian='little'),
        'b': zeros(64, endian='little'),
        'q': zeros(64, endian='little'),
        'k': zeros(64, endian='little'),
    }
    white_move = 1 if active_color == 'w' else -1
    castling_rights = zeros(4, endian='little')
    en_passant_target_pos_idx = filerank_to_pos_idx(en_passant) if en_passant != '-' else -1
    halfmove_counter = int(halfmove)
    fullmove_counter = int(fullmove)
    
    # Mapping FEN rows (reversed because FEN starts from rank 8 to rank 1)
    rows = pieces.split('/')
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)  # Skip empty squares
            else:
                lerf_idx = (7 - row_idx) * 8 + col_idx
                pos_idx = lerf_idx ^ 56
                piece_bitboards[char][pos_idx] = 1
                col_idx += 1

    if 'K' in castling:
        castling_rights[0] = 1
    if 'Q' in castling:
        castling_rights[1] = 1
    if 'k' in castling:
        castling_rights[2] = 1
    if 'q' in castling:
        castling_rights[3] = 1


    fen_gamestate['white_move'] = white_move
    fen_gamestate['halfmove_counter'] = halfmove_counter
    fen_gamestate['fullmove_counter'] = fullmove_counter
    fen_gamestate['en_passant_target_pos_idx'] = en_passant_target_pos_idx
    fen_gamestate['castling_rights'] = castling_rights
    fen_gamestate['piece_bitboards'] = piece_bitboards

    return fen_gamestate