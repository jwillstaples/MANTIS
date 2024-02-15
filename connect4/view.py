import arcade

from connect4.board_c4 import BoardC4
from connect4.mantis_c4 import MantisC4
from connect4.minimax_c4 import OracleC4, TestNet
from connect4.player_c4 import PlayerC4
from connect4.c4net import C4Net

# from connect4.player_c4 import PlayerC4

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 700


class C4Piece:
    def __init__(self, r, c, color):
        self.r = r
        self.c = c
        self.color = color


class C4Game(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, "Connect 4")
        arcade.set_background_color(arcade.color.WHITE)
        self.pieces = []
        self.selected = -1  # which column player selected

    def setup(self):
        self.board = BoardC4.from_start()
        self.result = 2
        # self.bot = OracleC4(depth=4)
        # nnet = TestNet()
        # self.bot = PlayerC4(nnet)

        self.bot = MantisC4("old.pt")

    def on_draw(self):
        self.clear()

        BORDER = 50
        for i in range(1, 9):
            x_position = i * SCREEN_WIDTH / 9
            arcade.draw_line(
                x_position,
                BORDER,
                x_position,
                SCREEN_HEIGHT - BORDER,
                arcade.color.BLACK,
                2,
            )
        arcade.draw_line(
            1 * SCREEN_WIDTH / 9,
            BORDER,
            8 * SCREEN_WIDTH / 9,
            BORDER,
            arcade.color.BLACK,
            2,
        )
        arcade.draw_line(
            1 * SCREEN_WIDTH / 9,
            SCREEN_HEIGHT - BORDER,
            8 * SCREEN_WIDTH / 9,
            SCREEN_HEIGHT - BORDER,
            arcade.color.BLACK,
            2,
        )

        if self.board.red_move and self.result == 2:
            self.board = self.bot.move(self.board)
            self.update_board(self.board)
        elif self.selected != -1 and self.board.bottom_available(self.selected) != 7:
            self.board = self.board.make_move(
                (self.selected, self.board.bottom_available(self.selected))
            )
            self.update_board(self.board)
            self.selected = -1

        self.result = self.board.terminal_eval()

        for piece in self.pieces:
            arcade.draw_circle_filled(piece.r, piece.c, 50, piece.color)

        if self.result != 2:
            text = "Red wins" if self.result == 1 else "Blue wins"
            text = "Draw" if self.result == 0 else text
            arcade.draw_text(
                text,
                450,
                675,
                arcade.color.BLACK,
                15,
                anchor_x="center",
                anchor_y="center",
            )

    def update_board(self, board):
        board = board.board_matrix
        for i in range(len(board)):
            for j in range(len(board[0])):
                x_pos = 150 + i * 100
                y_pos = 100 + j * 100
                color = arcade.color.RED if board[i][j] == 1 else arcade.color.BLUE
                if board[i][j] != 0:
                    self.pieces.append(C4Piece(x_pos, y_pos, color))

    def on_mouse_press(self, x, y, button, key_modifiers):
        if self.result != 2:
            arcade.Window.close(self)
        if 100 < x < 800:
            self.selected = (x - 100) // 100


def main():
    game = C4Game(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
