import numpy as np

from bitarray import bitarray
from bitarray.util import ba2int, int2ba, pprint


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
    special_move_flag = move_bitarray[0:2]
    promotion_piece_type = move_bitarray[2:4]
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
