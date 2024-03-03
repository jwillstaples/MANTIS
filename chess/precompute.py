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


if __name__ == '__main__':
    # POS_IDX:
    # [56,57,58,59,60,61,62,63], ..., [0,1,2,3,4,5,6,7]
    rook_rays = precompute_rook_rays()

    # ROOK ON LERF SQUARE = 27, POS_IDX = 35
    pos_idx = 35
    lerf_idx = pos_idx ^ 56
    filerank = pos_idx_to_filerank(pos_idx)
    file, rank = filerank[0], filerank[1]

    # SET OCCUPIED SQUARES
    occupied_squares = zeros(64, endian='little')
    lerf_idx_occupied = [11, 43, 59, 25, 29, 31, 27]
    pos_idx_occupied = [i ^ 56 for i in lerf_idx_occupied]
    occupied_squares[pos_idx_occupied] = 1

    print('Occupied Squares')
    lerf_bitboard_to_2D_numpy(occupied_squares)

    # for pidx in range(64):
    #     print(f'\n Combined Bitboard for pos_idx: {pidx}, lerf_idx: {pidx ^ 56}')
    #     combined = rook_rays[0][pidx] | rook_rays[1][pidx] | rook_rays[2][pidx] | rook_rays[3][pidx]
    #     lerf_bitboard_to_2D_numpy(combined)

    print('\nNorth Attacks')
    north_attacks = getRookRayAttacks(rook_rays, occupied_squares, 0, pos_idx)
    lerf_bitboard_to_2D_numpy(north_attacks)

    print('\nSouth Attacks')
    south_attacks = getRookRayAttacks(rook_rays, occupied_squares, 2, pos_idx)
    lerf_bitboard_to_2D_numpy(south_attacks)

    print('\nEast Attacks')
    east_attacks = getRookRayAttacks(rook_rays, occupied_squares, 1, pos_idx)
    lerf_bitboard_to_2D_numpy(east_attacks)

    print('\nWest Attacks')
    west_attacks = getRookRayAttacks(rook_rays, occupied_squares, 3, pos_idx)
    lerf_bitboard_to_2D_numpy(west_attacks)

    print('\nRook Attacks')
    rook_attacks = north_attacks | south_attacks | east_attacks | west_attacks
    lerf_bitboard_to_2D_numpy(rook_attacks)

    print(filerank_to_pos_idx('a1'))
    lerf_bitboard_to_2D_numpy((rook_rays[0][56]))