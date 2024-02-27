import numpy as np

from chess.utils import *

from bitarray import bitarray
from bitarray.util import ba2int, int2ba, zeros, pprint


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
    rook_rays = {}

    for pos_idx in range(64):
        lerf_idx = pos_idx ^ 56
        filerank = pos_idx_to_filerank(lerf_idx)
        file = filerank[0]
        rank = filerank[1]

        position_bitboard = zeros(64, endian='little')
        position_bitboard[lerf_idx] = 1

        filerank_bitboard = file_bitboards[file] | rank_bitboards[rank]

        ray_bitboard = position_bitboard ^ filerank_bitboard

        rook_rays[filerank] = ray_bitboard
    
    return rook_rays


if __name__ == '__main__':
    rook_rays = precompute_rook_rays()
    for filerank, bitboard in rook_rays.items():
        print(filerank)
        print(lerf_bitboard_to_2D_numpy(bitboard))
        print('--------------------------------------------\n')