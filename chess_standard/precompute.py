from chess_standard.utils import *

from bitarray.util import zeros

file_bitboards = {
    "a": hex_to_bitarray("0x8080808080808080"),
    "b": hex_to_bitarray("0x4040404040404040"),
    "c": hex_to_bitarray("0x2020202020202020"),
    "d": hex_to_bitarray("0x1010101010101010"),
    "e": hex_to_bitarray("0x0808080808080808"),
    "f": hex_to_bitarray("0x0404040404040404"),
    "g": hex_to_bitarray("0x0202020202020202"),
    "h": hex_to_bitarray("0x0101010101010101"),
}

rank_bitboards = {
    "1": hex_to_bitarray("0x00000000000000FF"),
    "2": hex_to_bitarray("0x000000000000FF00"),
    "3": hex_to_bitarray("0x0000000000FF0000"),
    "4": hex_to_bitarray("0x00000000FF000000"),
    "5": hex_to_bitarray("0x000000FF00000000"),
    "6": hex_to_bitarray("0x0000FF0000000000"),
    "7": hex_to_bitarray("0x00FF000000000000"),
    "8": hex_to_bitarray("0xFF00000000000000"),
}

diag_bitboards = {
    "1": hex_to_bitarray("0x0000000000000080"),
    "2": hex_to_bitarray("0x0000000000008040"),
    "3": hex_to_bitarray("0x0000000000804020"),
    "4": hex_to_bitarray("0x0000000080402010"),
    "5": hex_to_bitarray("0x0000008040201008"),
    "6": hex_to_bitarray("0x0000804020100804"),
    "7": hex_to_bitarray("0x0080402010080402"),
    "8": hex_to_bitarray("0x8040201008040201"),
    "9": hex_to_bitarray("0x4020100804020100"),
    "10": hex_to_bitarray("0x2010080402010000"),
    "11": hex_to_bitarray("0x1008040201000000"),
    "12": hex_to_bitarray("0x0804020100000000"),
    "13": hex_to_bitarray("0x0402010000000000"),
    "14": hex_to_bitarray("0x0201000000000000"),
    "15": hex_to_bitarray("0x0100000000000000"),
}

antidiag_bitboards = {
    "1": hex_to_bitarray("0x0000000000000001"),
    "2": hex_to_bitarray("0x0000000000000102"),
    "3": hex_to_bitarray("0x0000000000010204"),
    "4": hex_to_bitarray("0x0000000001020408"),
    "5": hex_to_bitarray("0x0000000102040810"),
    "6": hex_to_bitarray("0x0000010204081020"),
    "7": hex_to_bitarray("0x0001020408102040"),
    "8": hex_to_bitarray("0x0102040810204080"),
    "9": hex_to_bitarray("0x0204081020408000"),
    "10": hex_to_bitarray("0x0408102040800000"),
    "11": hex_to_bitarray("0x0810204080000000"),
    "12": hex_to_bitarray("0x1020408000000000"),
    "13": hex_to_bitarray("0x2040800000000000"),
    "14": hex_to_bitarray("0x4080000000000000"),
    "15": hex_to_bitarray("0x8000000000000000"),
}


def precompute_rook_rays():
    # [DIRECTION][SQUARE]
    # direction: 0 = NORTH, 1 = EAST, 2 = SOUTH, 3 = WEST
    # square: pos_idx position
    # little endian: a[0] = LSB, a[max] = MSB
    rook_rays = [[zeros(64, endian="little") for _ in range(64)] for _ in range(4)]

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
        east_ray = zeros(64, endian="little")
        east_ray[[i for i in rank_idxs if i > pos_idx]] = 1
        rook_rays[1][pos_idx] = east_ray

        # WEST
        west_ray = zeros(64, endian="little")
        west_ray[[i for i in rank_idxs if i < pos_idx]] = 1
        rook_rays[3][pos_idx] = west_ray

    return rook_rays


def precompute_bishop_rays():
    # [DIRECTION][SQUARE]
    # direction: 0 = NORTHWEST, 1 = NORTHEAST, 2 = SOUTHEAST, 3 = SOUTHWEST
    # square: pos_idx position
    # little endian: a[0] = LSB, a[max] = MSB
    bishop_rays = [[zeros(64, endian="little") for _ in range(64)] for _ in range(4)]
    files = "abcdefgh"

    for pos_idx in range(64):
        filerank = pos_idx_to_filerank(pos_idx)
        file, rank = filerank[0], filerank[1]

        diag = int(rank) + files.index(file)
        antidiag = int(rank) - files.index(file) + 7

        diag_idxs = diag_bitboards[str(diag)].search(1)  # NW = 0, SE = 2
        antidiag_idxs = antidiag_bitboards[str(antidiag)].search(1)  # NE = 1, SW = 3

        # NORTHWEST
        northwest_ray = zeros(64, endian="little")
        northwest_ray[[i for i in diag_idxs if i < pos_idx]] = 1
        bishop_rays[0][pos_idx] = northwest_ray

        # NORTHEAST
        northeast_ray = zeros(64, endian="little")
        northeast_ray[[i for i in antidiag_idxs if i < pos_idx]] = 1
        bishop_rays[1][pos_idx] = northeast_ray

        # SOUTHEAST
        southeast_ray = zeros(64, endian="little")
        southeast_ray[[i for i in diag_idxs if i > pos_idx]] = 1
        bishop_rays[2][pos_idx] = southeast_ray

        # SOUTHWEST
        southwest_ray = zeros(64, endian="little")
        southwest_ray[[i for i in antidiag_idxs if i > pos_idx]] = 1
        bishop_rays[3][pos_idx] = southwest_ray

    return bishop_rays


def precompute_pawn_rays():
    pawn_rays = [[zeros(64, endian="little") for _ in range(64)] for _ in range(2)]

    for pos_idx in range(64):
        filerank = pos_idx_to_filerank(pos_idx)
        file, rank = filerank[0], filerank[1]

        white_pawn_ray = zeros(64, endian="little")
        black_pawn_ray = zeros(64, endian="little")

        # WHITE PAWN ATTACK RAYS
        if file == "a":
            if pos_idx - 7 >= 0:
                white_pawn_ray[(pos_idx - 7)] = 1
        elif file == "h":
            if pos_idx - 9 >= 0:
                white_pawn_ray[(pos_idx - 9)] = 1
        else:
            if pos_idx - 9 >= 0:
                white_pawn_ray[[(pos_idx - 9), (pos_idx - 7)]] = 1
        pawn_rays[0][pos_idx] = white_pawn_ray

        # BLACK PAWN ATTACK RAYS
        if file == "a":
            if pos_idx + 9 <= 63:
                black_pawn_ray[(pos_idx + 9)] = 1
        elif file == "h":
            if pos_idx + 7 <= 63:
                black_pawn_ray[(pos_idx + 7)] = 1
        else:
            if pos_idx + 9 <= 63:
                black_pawn_ray[[(pos_idx + 9), (pos_idx + 7)]] = 1
        pawn_rays[1][pos_idx] = black_pawn_ray

    return pawn_rays


def precompute_knight_rays():
    knight_rays = [zeros(64, endian="little") for _ in range(64)]

    wraparound_masks = {
        6: (
            (file_bitboards["g"] | file_bitboards["h"]),
            (file_bitboards["a"] | file_bitboards["b"]),
        ),
        15: (file_bitboards["h"], file_bitboards["a"]),
        17: (file_bitboards["a"], file_bitboards["h"]),
        10: (
            (file_bitboards["a"] | file_bitboards["b"]),
            (file_bitboards["g"] | file_bitboards["h"]),
        ),
    }

    for pos_idx in range(64):
        filerank = pos_idx_to_filerank(pos_idx)
        file, rank = filerank[0], filerank[1]

        knight_ray = zeros(64, endian="little")

        for shift in [6, 15, 17, 10]:
            wraparound_file_north, wraparound_file_south = wraparound_masks[shift]

            # NORTH LOGIC
            if (pos_idx - shift >= 0) and (  # handles going out of range
                not wraparound_file_north[pos_idx]
            ):  # handles wraparound
                knight_ray[pos_idx - shift] = 1

            # SOUTH LOGIC
            if (pos_idx + shift <= 63) and (not wraparound_file_south[pos_idx]):
                knight_ray[pos_idx + shift] = 1

        knight_rays[pos_idx] = knight_ray

    return knight_rays


def precompute_king_rays():
    king_rays = [zeros(64, endian="little") for _ in range(64)]

    for pos_idx in range(64):
        filerank = pos_idx_to_filerank(pos_idx)
        file, rank = filerank[0], filerank[1]

        king_ray = zeros(64, endian="little")

        # northwest = -9
        if file != "a" and rank != "8":
            king_ray[pos_idx - 9] = 1
        # north = -8
        if rank != "8":
            king_ray[pos_idx - 8] = 1
        # northeast = -7
        if file != "h" and rank != "8":
            king_ray[pos_idx - 7] = 1
        # east = +1
        if file != "h":
            king_ray[pos_idx + 1] = 1
        # southeast = +9
        if file != "h" and rank != "1":
            king_ray[pos_idx + 9] = 1
        # south = +8
        if rank != "1":
            king_ray[pos_idx + 8] = 1
        # southwest = +7
        if file != "a" and rank != "1":
            king_ray[pos_idx + 7] = 1
        # west = -1
        if file != "a":
            king_ray[pos_idx - 1] = 1

        king_rays[pos_idx] = king_ray

    return king_rays


if __name__ == "__main__":

    rook_rays = precompute_rook_rays()
    bishop_rays = precompute_bishop_rays()
    pawn_rays = precompute_pawn_rays()
    knight_rays = precompute_knight_rays()
    king_rays = precompute_king_rays()

    lerf_idx = 53
    pos_idx = lerf_idx ^ 56
    print(f"pos_idx: {pos_idx}, lerf_idx: {lerf_idx}")

    filerank = pos_idx_to_filerank(pos_idx)
    file, rank = filerank[0], filerank[1]

    rook_ray = (
        rook_rays[0][pos_idx]
        | rook_rays[1][pos_idx]
        | rook_rays[2][pos_idx]
        | rook_rays[3][pos_idx]
    )

    bishop_ray = (
        bishop_rays[0][pos_idx]
        | bishop_rays[1][pos_idx]
        | bishop_rays[2][pos_idx]
        | bishop_rays[3][pos_idx]
    )

    queen_ray = rook_ray | bishop_ray

    # for pidx in range(64):
    #     print(f'\npos_idx: {pidx}, lerf_idx: {pidx ^ 56}')
    #     ray = knight_rays[pidx]
    #     lerf_bitboard_to_2D_numpy(ray)

    lerf_bitboard_to_2D_numpy(pawn_rays[1][pos_idx])
    # lerf_bitboard_to_2D_numpy(rook_ray)
    # lerf_bitboard_to_2D_numpy(bishop_ray)
    # lerf_bitboard_to_2D_numpy(queen_ray)
    # lerf_bitboard_to_2D_numpy(pawn_ray)
