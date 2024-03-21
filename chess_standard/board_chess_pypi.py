import numpy as np
import chess

class BoardPypiChess(BlankBoard):
    def __init__(self, fen=''):
        '''
        Initializes the chess board and all the included internal information from the chess.Board() class
        '''
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.white_move = self.board.turn
        self.all_legal_moves = self.legal_moves()

    def legal_moves(self) -> np.ndarray:
        """Returns a binary mask of length 7 for C4, 1858 for chess that represents the legal moves"""
        # andy give me the array :[
        return []

    def terminal_eval(self) -> int:
        """
        Returns dumb evaluation (win, loss, etc)
        -1 : player two (black) wins
        0 : draw
        1 : player one (white) wins
        2 : unterminated
        """
        if self.board.is_game_over():
            outcome = self.board.outcome()
            winner = outcome.winner
            if winner == None:
                return 0
            if winner:
                return 1
            else:
                return -1
        return 2

    def move_from_int(self, num: int) -> "BlankBoard":
        """
        Returns a board instance based on the integer position
        of the move in the p-vector
        """
        move_uci = self.all_legal_moves[num]
        move = chess.Move.from_uci(move_uci)

        self.board.push(move)
        new_board_fen = self.board.fen()
        self.board.pop()

        return BoardPypiChess(fen=new_board_fen)

    def to_tensor(self) -> torch.tensor:
        """
        Returns a tensor that can be inputted to a neural network
        """
        tb = torch.from_numpy(self.board_to_numpy(self.board))
        t1 = (tb > 0).float()
        t0 = (tb == 0).float()
        tn1 = (tb < 0).float()
        if self.white_move:
            return torch.stack([t1, t0, tn1])
        return torch.stack([tn1, t0, t1])

    def player_perspective_eval(self) -> int:
        """
        Returns dumb evaluation (win, loss, etc)
        -1 : current player losing
        0 : draw
        1 : current player winning
        2 : unterminated
        """
        terminal = self.terminal_eval()

        if terminal != 2:
            mult = 1 if self.white_move else -1
            return mult * terminal

        return terminal

    # Internal Helper Functions
    def board_to_numpy(self, board):
        '''
        Params:
            board (chess.Board) : a chess.Board object that you want to turn into a numpy array representation

        Returns:
            board_array (np.ndarray) : A 2D NP array with a custom piece_to_value encoding for representation

        NOTE: In the NP array, index 0 is A1 and index 63 is H8. While this seems intuitive, if you visualize the board,
                you'd think that it was flipped horizontally, but this is the correct representation.
        '''
        piece_map = self.board.piece_map()
        board_array = np.zeros((8, 8))

        # WHITE PIECES ARE POSITIVE
        # BLACK PIECES ARE NEGATIVE
        # EMPTY SQUARES ARE ZERO
        piece_to_value = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
    
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            value = piece_to_value.get(piece.piece_type, 0)
            if piece.color == chess.BLACK:
                value = -value
            board_array[row, col] = value
    
        return board_array
