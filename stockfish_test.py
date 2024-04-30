from chess_standard.board_chess_pypi import BoardPypiChess
import sys

sys.path.append("/home/jovyan/work/MANTIS")

board = BoardPypiChess.from_start()

print(board.board_to_numpy(board.board))
print(board.to_tensor())
print(board.white_move)
print(board.board.fen())
print(board.terminate_from_local_stockfish())

indices = [i for i, element in enumerate(board.legal_moves()) if element != 0]
print(indices)

board = board.move_from_int(34)

print(board.board_to_numpy(board.board))
print(board.to_tensor())
print(board.white_move)
print(board.board.fen())

print(board.terminal_eval())
print(board.terminate_from_local_stockfish())