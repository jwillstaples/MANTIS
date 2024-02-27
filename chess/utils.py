import numpy as np

from bitarray import bitarray
from bitarray.util import ba2int, int2ba


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
    return lerf_bitboard_to_1D_numpy(bitboard).reshape((8, 8))


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
