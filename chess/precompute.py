import numpy as np

from chess.utils import *

from bitarray import bitarray
from bitarray.util import ba2int, int2ba, zeros, ones, pprint

# 11000101 = occupied (the occupied squares bitboard)
# 00000100 = slider (the sliding piece of interest)
# 11000001 = occupied - slider (bitwise subtraction)

# 11000101 = occupied
# 00001000 = 2*slider, i.e., shift the slider one position more significant (left shift by 1)

file_bitboards = {
    'a': hex_to_bitarray('0x8080808080808080'), 
    'b': hex_to_bitarray('0x4040404040404040'),
    'c': hex_to_bitarray('0x2020202020202020'), 
    'd': hex_to_bitarray('0x1010101010101010'),
    'e': hex_to_bitarray('0x0808080808080808'),
    'f': hex_to_bitarray('0x0404040404040404'),
    'g': hex_to_bitarray('0x0202020202020202'), 
    'h': hex_to_bitarray('0x0101010101010101'),
}

rank_bitboards = {
    '1': hex_to_bitarray('0x00000000000000FF'),
    '2': hex_to_bitarray('0x000000000000FF00'),
    '3': hex_to_bitarray('0x0000000000FF0000'),
    '4': hex_to_bitarray('0x00000000FF000000'),
    '5': hex_to_bitarray('0x000000FF00000000'),
    '6': hex_to_bitarray('0x0000FF0000000000'),
    '7': hex_to_bitarray('0x00FF000000000000'),
    '8': hex_to_bitarray('0xFF00000000000000'),
}

diag_bitboards = {
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

antidiag_bitboards = {
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

def precompute_rook_rays():
    # [DIRECTION][SQUARE]
    # direction: 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST
    # square: pos_idx position
    # little endian: a[0] = LSB, a[max] = MSB
    rook_rays = [[zeros(64, endian='little') for _ in range(64)] for _ in range(4)]

    for pos_idx in range(64):
        filerank = pos_idx_to_filerank(pos_idx)
        file, rank = filerank[0], filerank[1]
        rank_idxs = rank_bitboards[rank].search(1)
        
        # NORTH
        north_ray = file_bitboards[file] << 8 * int(rank)
        rook_rays[0][pos_idx] = north_ray

        # SOUTH
        south_ray = file_bitboards[file] >> 8 * (9 - int(rank))
        rook_rays[2][pos_idx] = south_ray

        # EAST
        east_ray = zeros(64, endian='little')
        east_ray[[i for i in rank_idxs if i > pos_idx]] = 1
        rook_rays[1][pos_idx] = east_ray

        # WEST
        west_ray = zeros(64, endian='little')
        west_ray[[i for i in rank_idxs if i < pos_idx]] = 1
        rook_rays[3][pos_idx] = west_ray
    
    return rook_rays


def precompute_bishop_rays():
    # [DIRECTION][SQUARE]
    # direction: 0 = NORTHWEST, 1 = NORTHEAST, 2 = SOUTHEAST, 3 = SOUTHWEST
    # square: pos_idx position
    # little endian: a[0] = LSB, a[max] = MSB
    bishop_rays = [[zeros(64, endian='little') for _ in range(64)] for _ in range(4)]
    files = 'abcdefgh'
    
    for pos_idx in range(64):
        filerank = pos_idx_to_filerank(pos_idx)
        file, rank = filerank[0], filerank[1]

        diag = int(rank) + files.index(file)
        antidiag = int(rank) - files.index(file) + 7

        diag_idxs = diag_bitboards[str(diag)].search(1) # NW = 0, SE = 2
        antidiag_idxs = antidiag_bitboards[str(antidiag)].search(1) # NE = 1, SW = 3

        # NORTHWEST
        northwest_ray = zeros(64, endian='little')
        northwest_ray[[i for i in diag_idxs if i < pos_idx]] = 1
        bishop_rays[0][pos_idx] = northwest_ray

        # NORTHEAST
        northeast_ray = zeros(64, endian='little')
        northeast_ray[[i for i in antidiag_idxs if i < pos_idx]] = 1
        bishop_rays[1][pos_idx] = northeast_ray

        # SOUTHEAST
        southeast_ray = zeros(64, endian='little')
        southeast_ray[[i for i in diag_idxs if i > pos_idx]] = 1
        bishop_rays[2][pos_idx] = southeast_ray

        # SOUTHWEST
        southwest_ray = zeros(64, endian='little')
        southwest_ray[[i for i in antidiag_idxs if i > pos_idx]] = 1
        bishop_rays[3][pos_idx] = southwest_ray
    
    return bishop_rays


if __name__ == '__main__':
    bishop_rays = precompute_bishop_rays()

    for pos_idx in range(64):
        northwest_ray = bishop_rays[0][pos_idx]
        northeast_ray = bishop_rays[1][pos_idx]
        southeast_ray = bishop_rays[2][pos_idx]
        southwest_ray = bishop_rays[3][pos_idx]

        combined = northwest_ray | northeast_ray | southeast_ray | southwest_ray

        print(f'pos_idx: {pos_idx}, lerf_idx: {pos_idx ^ 56}')
        lerf_bitboard_to_2D_numpy(combined)
        print(f'--------------------------------------- \n')
