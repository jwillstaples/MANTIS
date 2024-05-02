import chess
import arcade
import time

from collections import defaultdict
from chess_standard.board_chess_pypi import BoardPypiChess
from chess_standard.utils import *

from chess_standard.mantis_chess import MantisChess

# ---- PARAMS ---------------------------------------------------------------------------
SCREEN_TITLE = "MANTIS CHESS"
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BOARD_ROWS = 8
BOARD_COLS = 8
SQUARE_SIZE = SCREEN_HEIGHT // BOARD_COLS
LIGHT_COLOR = arcade.color.WHEAT
DARK_COLOR = arcade.color.BURLYWOOD
FONT_SIZE = 12
STATUS_FONT_SIZE = 36

DELAY_TIME = 1.0

# ---- CHESS FRONTEND -------------------------------------------------------------------
class ChessGame(arcade.Window):
    # ---- MAIN FUNCTIONS ---------------------------------------------------------------
    def __init__(
            self,
            # mode select: 0 = human v human, 1 = human v bot, 2 = bot v bot
            mode=0,

            # human v bot params
            player_clr=True,

            # bot1 params
            white_bot=None,

            # bot2 params
            black_bot=None,

            # game state signals
            board_init_fen="",

            # only for bot v bot win percentage
            random_eval_iters=100,
            bot_clr=True
        ):
        '''
        initializes the **arcade** window w/ all the PARAMS
        '''
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color((24, 24, 24))
        self.textures = {
            1: arcade.load_texture("chess_standard/static/wP.png"),
            2: arcade.load_texture("chess_standard/static/wN.png"),
            3: arcade.load_texture("chess_standard/static/wB.png"),
            4: arcade.load_texture("chess_standard/static/wR.png"),
            5: arcade.load_texture("chess_standard/static/wQ.png"),
            6: arcade.load_texture("chess_standard/static/wK.png"),
            -1: arcade.load_texture("chess_standard/static/bP.png"),
            -2: arcade.load_texture("chess_standard/static/bN.png"),
            -3: arcade.load_texture("chess_standard/static/bB.png"),
            -4: arcade.load_texture("chess_standard/static/bR.png"),
            -5: arcade.load_texture("chess_standard/static/bQ.png"),
            -6: arcade.load_texture("chess_standard/static/bK.png"),
            0: None,
        }

        self.mode = mode
        self.player_clr = player_clr
        self.board_init_fen = board_init_fen
        self.random_eval_iters = random_eval_iters

        # bot init
        self.bot_clr = bot_clr
        self.white_bot = white_bot
        self.black_bot = black_bot

        self.match_iter = 1
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0

        self.chess_setup()
    
    def chess_setup(self):
        '''
        initializes the actual chess game setup
        '''
        # board vars
        self.chess_board = BoardPypiChess(fen=self.board_init_fen)

        # sprites vars
        self.chess_piece_sprites = arcade.SpriteList()
        self.update_board()

        # game state/selection ctrl vars
        self.origin_square = None # type = chess.Square
        self.target_square = None # type = chess.Square
        self.valid_moves = None
        self.is_promotion_event = False
        self.is_gameover_event = False

        # helper vars
        self.mouse_x = 0
        self.mouse_y = 0
        self.file_letters = "abcdefgh"

    # ---- MAIN EVENT HANDLERS ----------------------------------------------------------
    def on_draw(self):
        '''
        event triggers CONTINUALLY
        clears, redraws all game objects, and updates display w/ new frame
        '''
        arcade.start_render()
        self.draw_board()

        if (self.origin_square):
            self.draw_origin_highlight()
        if (self.valid_moves):
            self.draw_valid_moves_highlights()

        self.chess_piece_sprites.draw()

        if (self.is_promotion_event):
            self.draw_promotion_screen()
            self.draw_promotion_boxes()
            self.promotion_piece_sprites.draw()
        
        if (self.is_gameover_event):
            self.draw_gameover_screen()


    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        '''
        event triggers on mouse press
        in charge of core game loop of interacting w/ pieces/board
        '''
        if (
            (button == arcade.MOUSE_BUTTON_LEFT) and
            (
                (self.mode == 0) or
                (self.mode == 1 and self.chess_board.board.turn == self.player_clr)
            ) and 
            (bool(self.chess_board.board.legal_moves))
        ):
            
            if self.is_promotion_event:
                rook_sprite = self.promotion_piece_sprites[0]
                knight_sprite = self.promotion_piece_sprites[1]
                bishop_sprite = self.promotion_piece_sprites[2]
                queen_sprite = self.promotion_piece_sprites[3]

                promo = None

                # CLICKED TYPE ROOK = 4
                if self.is_mouse_hovering_promotion_sprite(rook_sprite, x, y):
                    promo = 4
                # CLICKED TYPE KNIGHT = 2
                elif self.is_mouse_hovering_promotion_sprite(knight_sprite, x, y):
                    promo = 2
                # CLICKED TYPE BISHOP = 3
                elif self.is_mouse_hovering_promotion_sprite(bishop_sprite, x, y):
                    promo = 3
                # CLICKED TYPE QUEEN = 5
                elif self.is_mouse_hovering_promotion_sprite(queen_sprite, x, y):
                    promo = 5

                self.is_promotion_event = False
                self.promotion_piece_sprites.clear()

                if promo != None:
                    move = self.chess_board.board.find_move(self.origin_square, self.target_square, promo)
                    self.make_human_move_to_board(move=move)

            else:
                row, col = (SCREEN_HEIGHT - y) // SQUARE_SIZE, x // SQUARE_SIZE
                rank, file = 8 - row, self.file_letters[col]

                clicked_sq_filerank = file + str(rank)
                clicked_sq_num = chess.parse_square(clicked_sq_filerank)

                # on mouse click, first check if clicked square is a selectable piece
                if (
                    (self.chess_board.board.color_at(clicked_sq_num) != None) and 
                    (   
                        (self.mode == 0 and self.chess_board.board.color_at(clicked_sq_num) == self.chess_board.board.turn) or
                        (self.mode == 1 and self.chess_board.board.color_at(clicked_sq_num) == self.player_clr)
                    )
                ):  
                    # if clicked squared is a selectable piece that we've already selected, reset highlight
                    if self.origin_square == clicked_sq_num:
                        self.reset_selection()
                    # else, assign the piece's number and its valid moves to the class vars
                    else:
                        self.origin_square = clicked_sq_num
                        self.valid_moves = self.get_valid_moves()

                # else, clicked square is NOT a selectable piece, so potentially is a move!
                else:
                    # if we have a selected piece already (self.origin_square != None) and
                    # the clicked square (target square) is in the valid moves list for that piece
                    if (
                        (self.origin_square != None) and
                        (clicked_sq_num in self.valid_moves.keys()) and 
                        (   
                            (self.mode == 0) or
                            (self.mode == 1 and self.chess_board.board.turn == self.player_clr)
                        )
                    ):
                        self.target_square = clicked_sq_num

                        promotion_type = None
                        if (self.valid_moves[self.target_square][self.origin_square]):
                            self.is_promotion_event = True
                            promotion_type = self.valid_moves[self.target_square][self.origin_square]
                        
                        if (self.is_promotion_event == False):
                            move = self.chess_board.board.find_move(
                                from_square=self.origin_square, 
                                to_square=self.target_square,
                                promotion=promotion_type
                            )

                            self.make_human_move_to_board(move=move)
                    
                    if (self.is_promotion_event == False):
                        self.reset_selection()


    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        '''
        event triggers on mouse motion
        in charge of triggering "hover" events
        '''
        self.mouse_x = x
        self.mouse_y = y
    
    # ---- DRAWING & SPRITES ------------------------------------------------------------
    def load_piece_sprites(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                piece_num = int(self.chess_board_array[row][col])
                if (piece_num != 0):
                    sprite = arcade.Sprite(
                        scale=0.05,
                        center_x=col * SQUARE_SIZE + SQUARE_SIZE / 2,
                        center_y=SCREEN_HEIGHT - (row * SQUARE_SIZE + SQUARE_SIZE / 2),
                        texture=self.textures[piece_num]
                    )
                    self.chess_piece_sprites.append(sprite)
    
    def draw_board(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
                text_color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                square_color = DARK_COLOR if (row + col) % 2 == 0 else LIGHT_COLOR
                arcade.draw_rectangle_filled(
                    center_x=x + SQUARE_SIZE / 2, 
                    center_y=y + SQUARE_SIZE / 2,
                    width=SQUARE_SIZE,
                    height=SQUARE_SIZE,
                    color=square_color,
                )
                if row == 7:
                    arcade.draw_text(
                        text=self.file_letters[col],
                        start_x=x + SQUARE_SIZE - 5,
                        start_y=y + SQUARE_SIZE,
                        color=text_color,
                        font_size=FONT_SIZE,
                        anchor_x="right",
                        anchor_y="top",
                        bold=True,
                    )
                if col == 0:
                    arcade.draw_text(
                        text=row + 1,
                        start_x=x + 12,
                        start_y=y + 20,
                        color=text_color,
                        font_size=FONT_SIZE,
                        anchor_x="right",
                        anchor_y="top",
                        bold=True,
                    )

    def draw_origin_highlight(self):
        row, col = chess.square_rank(square=self.origin_square), chess.square_file(square=self.origin_square)
        arcade.draw_rectangle_filled(
            center_x=col * SQUARE_SIZE + SQUARE_SIZE / 2, 
            center_y=row * SQUARE_SIZE + SQUARE_SIZE / 2, 
            width=SQUARE_SIZE, 
            height=SQUARE_SIZE, 
            color=arcade.color.GREEN
        )

    def draw_valid_moves_highlights(self):
        for target_sq, mapping_dict in self.valid_moves.items():
            for origin_sq, _ in mapping_dict.items():
                if origin_sq == self.origin_square:
                    row, col = chess.square_rank(square=target_sq), chess.square_file(square=target_sq)
                    arcade.draw_circle_filled(
                        center_x=col * SQUARE_SIZE + SQUARE_SIZE / 2, 
                        center_y=row * SQUARE_SIZE + SQUARE_SIZE / 2, 
                        radius=SQUARE_SIZE / 4, 
                        color=arcade.color.GREEN
                    )

    def is_mouse_hovering_promotion_sprite(self, sprite, x, y):
        left = sprite.center_x - sprite.width / 2
        right = sprite.center_x + sprite.width / 2
        bottom = sprite.center_y - sprite.height / 2
        top = sprite.center_y + sprite.height / 2

        return left < x < right and bottom < y < top
    
    def draw_promotion_screen(self):
        x_pos = 8 * SQUARE_SIZE + 226
        y_pos = (SCREEN_HEIGHT - 20) - 50 - (25 * 20)
        arcade.draw_lrtb_rectangle_filled(0, 800, SCREEN_HEIGHT, 0, (0, 0, 0, 127))
        arcade.draw_text(
            f"Select promotion type:",
            x_pos,
            y_pos,
            (167, 167, 167),
            14,
            anchor_x="right",
            anchor_y="bottom",
            bold=True,
        )

        promotion_options = None
        if (
            (self.mode == 0 and self.chess_board.board.turn == True) or 
            (self.mode == 1 and self.chess_board.board.turn == self.player_clr and self.player_clr == True)
        ):
            promotion_options = [4, 2, 3, 5]
        elif (
            (self.mode == 0 and self.chess_board.board.turn == False) or
            (self.mode == 1 and self.chess_board.board.turn == self.player_clr and self.player_clr == False)
        ):
            promotion_options = [-4, -2, -3, -5]

        if (promotion_options):
            for i, option_name in enumerate(promotion_options):
                texture = self.textures[option_name]
                sprite = arcade.Sprite()
                sprite.texture = texture
                sprite.scale = 0.03
                sprite.center_x = (x_pos - (112)) + (62 * (i - 1))
                sprite.center_y = y_pos - 50
                self.promotion_piece_sprites.append(sprite)

    def draw_promotion_boxes(self):
        fill_hover = (40, 40, 40)
        outline_hover = (24, 24, 24)
        fill_default = (167, 167, 167)
        outline_default = (255, 255, 255)

        rook = self.promotion_piece_sprites[0]
        knight = self.promotion_piece_sprites[1]
        bishop = self.promotion_piece_sprites[2]
        queen = self.promotion_piece_sprites[3]

        def draw_boxes_filled(sprite):
            arcade.draw_rectangle_filled(
                sprite.center_x,
                sprite.center_y,
                sprite.width,
                sprite.height,
                fill_hover,
            )
            arcade.draw_rectangle_outline(
                sprite.center_x,
                sprite.center_y,
                sprite.width,
                sprite.height,
                outline_hover,
                2,
            )

        def draw_boxes_default(sprite):
            arcade.draw_rectangle_filled(
                sprite.center_x,
                sprite.center_y,
                sprite.width,
                sprite.height,
                fill_default,
            )
            arcade.draw_rectangle_outline(
                sprite.center_x,
                sprite.center_y,
                sprite.width,
                sprite.height,
                outline_default,
                2,
            )

        if self.is_mouse_hovering_promotion_sprite(rook, self.mouse_x, self.mouse_y):
            draw_boxes_filled(rook)
        else:
            draw_boxes_default(rook)

        if self.is_mouse_hovering_promotion_sprite(knight, self.mouse_x, self.mouse_y):
            draw_boxes_filled(knight)
        else:
            draw_boxes_default(knight)

        if self.is_mouse_hovering_promotion_sprite(bishop, self.mouse_x, self.mouse_y):
            draw_boxes_filled(bishop)
        else:
            draw_boxes_default(bishop)

        if self.is_mouse_hovering_promotion_sprite(queen, self.mouse_x, self.mouse_y):
            draw_boxes_filled(queen)
        else:
            draw_boxes_default(queen) 

    def draw_gameover_screen(self):
        arcade.draw_lrtb_rectangle_filled(
            left=0, 
            right=SCREEN_WIDTH, 
            top=SCREEN_HEIGHT, 
            bottom=0, 
            color=(0, 0, 0, 127)
        )
        gameover_text = ""
        if (self.chess_board.terminal_eval() == 1):
            gameover_text = "WHITE WINS!"
            self.white_wins += 1
        elif (self.chess_board.terminal_eval() == -1):
            gameover_text = "BLACK WINS!"
            self.black_wins += 1
        elif (self.chess_board.terminal_eval() == 0):
            gameover_text = "DRAW."
            self.draws += 1
        print(f'Results of Match #{self.match_iter}: {gameover_text} | Score = (W: {self.white_wins}, B: {self.black_wins}, D: {self.draws})')
        
        white_win_percentage = self.white_wins / self.match_iter
        black_win_percentage = self.black_wins / self.match_iter
        draw_percentage = self.draws / self.match_iter

        if (self.bot_clr):
            print(f"Our bot\'s win percentage = {self.white_wins}/{self.match_iter} = {white_win_percentage:.2f}")
        else:
            print(f"Our bot\'s win percentage = {self.black_wins}/{self.match_iter} = {black_win_percentage:.2f}")

        self.match_iter += 1

        arcade.draw_text(
            text=gameover_text,
            start_x=400,
            start_y=self.height / 2,
            color=(255, 255, 255),
            font_size=STATUS_FONT_SIZE,
            anchor_x="center",
            anchor_y="center",
            bold=True
        )

        if (self.match_iter < self.random_eval_iters + 1):
            self.reset_game()
        else:
            arcade.exit()


    # ---- EVENT HANDLING LOGIC ---------------------------------------------------------
    def get_valid_moves(self):
        all_valid_moves = defaultdict(dict) # {to_square(s) : {from_square(s) : promotion(s)}}

        for chessMove in self.chess_board.board.legal_moves:
            all_valid_moves[chessMove.to_square][chessMove.from_square] = chessMove.promotion

        return all_valid_moves
    
    def update_board(self):
        self.chess_board_array = self.chess_board.board_to_perspective(self.chess_board.board)

        self.chess_piece_sprites.clear()
        self.load_piece_sprites()

        if (self.chess_board.board.is_game_over()):
            self.is_gameover_event = True
            arcade.unschedule(self.make_bot_move_to_board)
        else:
            if (self.mode == 2):
                arcade.schedule(self.make_bot_move_to_board, DELAY_TIME)
    
    def make_human_move_to_board(self, move: chess.Move):
        self.chess_board.board.push(move)
        self.update_board()

        if (self.mode == 1):
            self.make_bot_move_to_board()
    
    def make_bot_move_to_board(self, delta_time=0):
        if (self.chess_board.board.turn):
            self.chess_board = self.white_bot.move(self.chess_board)
        else:
            self.chess_board = self.black_bot.move(self.chess_board)
        self.update_board()
        
    def reset_selection(self):
        self.origin_square = None
        self.target_square = None
        self.valid_moves = None
        self.is_promotion_event = False
    
    def reset_game(self):
        self.chess_setup()

        

