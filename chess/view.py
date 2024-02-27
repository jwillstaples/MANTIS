import arcade

from bitarray.util import zeros

from chess.board_chess import BoardChess
from chess.utils import *

SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 800
SCREEN_TITLE = 'Chess'

BOARD_ROWS = 8
BOARD_COLS = 8
SQUARE_SIZE = SCREEN_HEIGHT // BOARD_COLS  # Ensure the board fits exactly into the window
LIGHT_COLOR = arcade.color.WHEAT
DARK_COLOR = arcade.color.BURLYWOOD

class ChessGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        arcade.set_background_color((24, 24, 24))

    def setup(self):
        self.board = BoardChess(white_move=1)
        self.textures = {
            'P': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/wP.png'),
            'R': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/wR.png'),
            'N': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/wN.png'),
            'B': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/wB.png'),
            'Q': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/wQ.png'),
            'K': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/wK.png'),
            'p': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/bP.png'),
            'r': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/bR.png'),
            'n': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/bN.png'),
            'b': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/bB.png'),
            'q': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/bQ.png'),
            'k': arcade.load_texture('C:/Users/Elliot/Desktop/MANTIS/MANTIS/chess/static/bK.png'),
            '.': None
        }
        self.game_state_array = self.board.visualize_current_gamestate()
        self.piece_sprites = arcade.SpriteList()
        self.load_pieces()

        self.origin_square = None
        self.valid_moves = None


    def on_draw(self):
        arcade.start_render()
        self.draw_board()

        if self.origin_square:
            self.highlight_origin_square(self.origin_square)
        if self.valid_moves:
            self.highlight_valid_moves(self.valid_moves)
        
        self.draw_annotations()
        self.piece_sprites.draw()
        self.draw_move_history()


    def load_pieces(self):
        for rank in range(BOARD_ROWS):
            for file in range(BOARD_COLS):
                    piece = self.game_state_array[rank][file]
                    texture = self.textures[piece]
                    if texture:
                        sprite = arcade.Sprite()
                        sprite.texture = texture
                        sprite.scale = 0.05
                        sprite.center_x = file * SQUARE_SIZE + SQUARE_SIZE / 2
                        sprite.center_y = SCREEN_HEIGHT - (rank * SQUARE_SIZE + SQUARE_SIZE / 2) 
                        self.piece_sprites.append(sprite)


    def draw_move_history(self):
        table_x = 8 * SQUARE_SIZE + 20  
        row_height = 20  
        max_rows = 25  
        table_y_start = SCREEN_HEIGHT - 20  
        background_color = (40, 40, 40) 
        move_width = 80
        arcade.draw_rectangle_filled(table_x + 130, table_y_start - (max_rows / 2 * row_height), 260, max_rows * row_height, background_color)

        counter = 1
        for i in range(0, len(self.board.move_history[-max_rows*2:]), 2):
            move_y = table_y_start - (i // 2 * row_height)
            arcade.draw_text(f'{counter}.) ', table_x, move_y, (167, 167, 167), 14, anchor_x="left", anchor_y="top", bold=True)

            move_text = self.board.move_history[i]
            arcade.draw_text(move_text, table_x + move_width, move_y, (167, 167, 167), 14, anchor_x="left", anchor_y="top", bold=True)
            
            if i + 1 < len(self.board.move_history):
                move_text = self.board.move_history[i + 1]
                arcade.draw_text(move_text, table_x + 2*move_width, move_y, (167, 167, 167), 14, anchor_x="left", anchor_y="top", bold=True)

            counter += 1

        turn = 'White'
        if self.board.white_move == 1:
            turn = 'White'
        elif self.board.white_move == -1:
            turn = 'Black'

        footer_y = table_y_start - 30 - (max_rows * row_height)
        arcade.draw_text(f'{turn} to move', table_x + 130, footer_y, (167, 167, 167), 14, anchor_x="right", anchor_y="bottom", bold=True)



    def draw_annotations(self):
        file_letters = 'abcdefgh'
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                x = col * SQUARE_SIZE
                y = row * SQUARE_SIZE
                text_color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                if row == 7:
                    arcade.draw_text(file_letters[col], x + SQUARE_SIZE - 5, y + SQUARE_SIZE, text_color, 12, anchor_x="right", anchor_y="top", bold=True)
                if col == 0:
                    arcade.draw_text(row+1, x + 12, y + 20, text_color, 12, anchor_x="right", anchor_y="top", bold=True)


    def draw_board(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                x = col * SQUARE_SIZE
                y = row * SQUARE_SIZE
                color = DARK_COLOR if (row + col) % 2 == 0 else LIGHT_COLOR
                arcade.draw_rectangle_filled(x + SQUARE_SIZE / 2, y + SQUARE_SIZE / 2, SQUARE_SIZE, SQUARE_SIZE, color)
                

    def get_valid_moves(self, filerank):
        filtered_moves = []

        for i, test_move in enumerate(self.board.get_pseudolegal_moves()):
            origin_square, target_square, _, _ = decode_move(test_move)
            if pos_idx_to_filerank(origin_square) == filerank:
                filtered_moves.append(pos_idx_to_filerank(target_square))

        return filtered_moves


    def highlight_origin_square(self, origin_square):
        rank = int(origin_square[1])
        file = origin_square[0]

        col = ord(file) - ord('a')
        row = rank - 1

        center_x = col * SQUARE_SIZE + SQUARE_SIZE / 2
        center_y = row * SQUARE_SIZE + SQUARE_SIZE / 2
        arcade.draw_rectangle_filled(center_x, center_y, SQUARE_SIZE, SQUARE_SIZE, arcade.color.GREEN)


    def highlight_valid_moves(self, moves):
        for target_square in moves:
            rank = int(target_square[1])
            file = target_square[0]

            col = ord(file) - ord('a')
            row = rank - 1

            center_x = col * SQUARE_SIZE + SQUARE_SIZE / 2
            center_y = row * SQUARE_SIZE + SQUARE_SIZE / 2

            arcade.draw_circle_filled(center_x, center_y, SQUARE_SIZE / 4, arcade.color.GREEN)


    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            col = x // SQUARE_SIZE
            row = (SCREEN_HEIGHT - y) // SQUARE_SIZE

            rank = 8 - row
            file_letters = 'abcdefgh'
            file = file_letters[col]

            filerank = file + str(rank)

            # on mouse click, first check if the clicked square is a selectable piece
            if self.board.is_piece_selectable(filerank):
                # if the clicked selectable piece is the one we already have selected, reset highlight
                if self.origin_square == filerank:
                    self.origin_square = None
                    self.valid_moves = None
                # else, assign the filerank and its valid moves to the class variables
                else:
                    self.origin_square = filerank
                    self.valid_moves = self.get_valid_moves(filerank)

            # if the clicked square is NOT a selectable piece
            else:
                # then, first check if we already have a selected piece and 
                # if the clicked square is in the valid moves for that piece
                if self.origin_square and filerank in self.valid_moves:
                    # if it is, then THAT'S A VALID MOVE BAYBEEEEE
                    encoded_move = zeros(16, endian='big')
                    encoded_move[4:10] = pos_idx_to_bitarray(filerank_to_pos_idx(filerank), length=6)
                    encoded_move[10:16] = pos_idx_to_bitarray(filerank_to_pos_idx(self.origin_square), length=6)

                    self.board.make_move(encoded_move)
                    self.game_state_array = self.board.visualize_current_gamestate()
                    self.piece_sprites.clear()
                    self.load_pieces()

                # reset the selection
                self.origin_square = None
                self.valid_moves = None
                

def main():
    game = ChessGame()
    game.setup()
    arcade.run()

if __name__ == '__main__':
    main()