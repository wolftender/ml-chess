# simple chess playing program made in python
# as a part of artificial intelligence project prepared for
# artificial intelligence course on mini faculty

import math
import sys
import ctypes
from threading import Thread
from time import sleep

# sdl
from sdl2 import *
from sdl2.sdlimage import *
from sdl2.sdlttf import *

# chess library
import chess

# tensorflow
import tensorflow as tf
import numpy as np

# random player
import random

class Player:
    def get_name(self) -> str:
        return "unnamed"

    def make_move(self, board : chess.Board):
        pass

# this is the "stupid" player class that can be used for all kinds of testing
# it works in a very simple way, it simply selects a random moves from a set of
# moves that it can make without regard for any evaluation
class RandomPlayer(Player):
    def get_name(self) -> str:
        return "random player"

    def make_move(self, board : chess.Board):
        sleep(2)

        move_count = board.legal_moves.count()
        move_idx = random.randint(0, move_count - 1)
        selected_move = None
        
        index = 0
        for move in board.legal_moves:
            if index == move_idx:
                selected_move = move

            index = index + 1
                

        board.push(selected_move)

class NeuralNetworkPlayer(Player):
    def __init__(self, checkpoint_name = 't2/checkpoint.ckpt', add_depth = False) -> None:
        super().__init__()

        print("loading tf model...")
        vec_len = 774

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(vec_len,)),
            tf.keras.layers.Dense(vec_len, activation='relu'),
            tf.keras.layers.Dense(vec_len, activation='relu'),
            tf.keras.layers.Dense(vec_len, activation='relu'),
            tf.keras.layers.Dense(vec_len, activation='relu'),
            tf.keras.layers.Dense(vec_len, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [ 'accuracy' ]
        )

        self.model.load_weights('checkpoints/model_v2/' + checkpoint_name)

    def get_name(self) -> str:
        return "tensorflow"

    def find_min_max(self, board : chess.Board, color):
        count_eval = 0
        batch = []

        encoded = self.encode_board(board)

        outcome = board.outcome()
        if outcome != None:
            if outcome.winner == color:
                return (9999, 9999, 0)
            elif outcome.winner == (not color):
                return (-9999, -9999, 0)
            else:
                return (0, 0, 0)

        checkmate_in_two = False
        for move in board.legal_moves:
            board.push(move)

            # check for obvious
            outcome = board.outcome()
            if outcome != None:
                if outcome.winner == (not color):
                    board.pop()
                    return (-9999, -9999, 0)
                elif outcome.winner == color:
                    checkmate_in_two = True

            encoded = self.encode_board(board)
            batch.append(encoded)

            board.pop()
            count_eval = count_eval + 1

        pred_data = np.stack(batch, axis=0)
        pred_labels = self.model.predict_on_batch(pred_data)

        min_eval = pred_labels.min()
        max_eval = pred_labels.max()

        if color == chess.BLACK:
            tmp = min_eval
            min_eval = -max_eval
            max_eval = -tmp
        
        if not checkmate_in_two:
            return (min_eval, max_eval, count_eval)
        else:
            return (min_eval, 9999, count_eval)

    def make_move(self, board : chess.Board):
        clone_board = board.copy()

        best_min_eval = -1000000
        best_max_eval = -1000000
        best_move = None
        evaluated_moves = 0

        print("evaluating moves...")

        for move in board.legal_moves:
            clone_board.push(move)
            min_eval, max_eval, count_eval = self.find_min_max(clone_board, board.turn)

            print("move gen min=", min_eval, "max=", max_eval)

            if min_eval > best_min_eval:
                best_min_eval = min_eval
                best_max_eval = max_eval
                best_move = move
            elif abs(min_eval - best_min_eval) < 0.001:
                if max_eval > best_max_eval:
                    best_min_eval = min_eval
                    best_max_eval = max_eval
                    best_move = move

            evaluated_moves = evaluated_moves + count_eval
            clone_board.pop()

        print("evaluated", evaluated_moves, "moves, chose move min=", best_min_eval, "max=", best_max_eval)
        try:
            board.push(best_move)
        except:
            print("cannot push move")
        

    def bitboard_to_vector(self, bitboard):
        serialized = np.unpackbits(np.array([bitboard], dtype=np.uint64).view(np.uint8)).astype(np.single)
        return serialized

    def encode_board(self, board : chess.Board):
        # this function gets SquareSet for given figure type
        # square set is a 64-bit integer (a bitboard)
        K = self.bitboard_to_vector(int(board.pieces(chess.KING, chess.WHITE)))
        Q = self.bitboard_to_vector(int(board.pieces(chess.QUEEN, chess.WHITE)))
        R = self.bitboard_to_vector(int(board.pieces(chess.ROOK, chess.WHITE)))
        B = self.bitboard_to_vector(int(board.pieces(chess.BISHOP, chess.WHITE)))
        N = self.bitboard_to_vector(int(board.pieces(chess.KNIGHT, chess.WHITE)))
        P = self.bitboard_to_vector(int(board.pieces(chess.PAWN, chess.WHITE)))

        k = self.bitboard_to_vector(int(board.pieces(chess.KING, chess.BLACK)))
        q = self.bitboard_to_vector(int(board.pieces(chess.QUEEN, chess.BLACK)))
        r = self.bitboard_to_vector(int(board.pieces(chess.ROOK, chess.BLACK)))
        b = self.bitboard_to_vector(int(board.pieces(chess.BISHOP, chess.BLACK)))
        n = self.bitboard_to_vector(int(board.pieces(chess.KNIGHT, chess.BLACK)))
        p = self.bitboard_to_vector(int(board.pieces(chess.PAWN, chess.BLACK)))

        isblack = 0
        iswhite = 0

        white_king_castl = 0
        white_queen_castl = 0
        black_king_castl = 0
        black_queen_castl = 0

        if board.has_kingside_castling_rights(chess.WHITE):
            white_king_castl = 1

        if board.has_queenside_castling_rights(chess.WHITE):
            white_queen_castl = 1

        if board.has_kingside_castling_rights(chess.BLACK):
            black_king_castl = 1

        if board.has_queenside_castling_rights(chess.BLACK):
            black_queen_castl = 1

        if board.turn == chess.WHITE:
            iswhite = 1
        elif board.turn == chess.BLACK:
            isblack = 1

        meta = np.array([isblack, iswhite, white_king_castl, white_queen_castl, black_king_castl, black_queen_castl])
        return np.concatenate([K, Q, R, B, N, P, k, q, r, b, n, p, meta], axis=0)

class ChessGame:
    def __init__(self, white : Player = None, black : Player = None, fen = None, start_perspective = chess.WHITE) -> None:
        if fen != None:
            self.board = chess.Board(fen = fen)
        else:
            self.board = chess.Board()

        print(self.board)

        # dimensions for the window
        self.window_width = 720
        self.window_height = 520

        # dimensions for the chessboard
        self.chessboard_dim = 480
        self.chessboard_x = 20
        self.chessboard_y = 20

        self.running = False
        self.piece_sprites = None
        self.font = None

        # mouse events
        self.mouse_down = False
        self.mouse_click = False
        self.mouse_click_x = 0
        self.mouse_click_y = 0

        # change this variable to change the perspective
        self.perspective = start_perspective

        # variables for the ui with chess moving
        self.selection_mask = [0] * 64
        self.selection_square = None

        self.is_player_move = True

        # players, todo: introduce the neural network from colab here
        self.white_player = white
        self.black_player = black

        self.waiting_white = False
        self.waiting_black = False

        self.game_active = True
        self.game_message = ""
        self.calculate_material()

    def coords_to_square(self, x, y):
        if self.perspective == chess.WHITE:
            return (7 - y) * 8 + x
        else:
            return y * 8 + x

    def square_to_coord(self, square):
        x = square % 8
        y = 0

        if self.perspective == chess.WHITE:
            y = 7 - math.floor(square / 8)
        else:
            y = math.floor(square / 8)

        return (x, y)

    def clear_selection(self):
        self.selection_mask = [0] * 64
        self.selection_square = None

    def mark_square(self, square):
        x, y = self.square_to_coord(square)
        index = y * 8 + x
        self.selection_mask[index] = 1

    def create_text_texture(self, font : TTF_Font, text : bytes, color : SDL_Color):
        surface = TTF_RenderText_Solid(font, text, color)
        width = surface.contents.w
        height = surface.contents.h

        texture = SDL_CreateTextureFromSurface(self.renderer, surface)
        SDL_FreeSurface(surface)

        return (texture, width, height)
    
    def render_text(self, x : int, y : int, font : TTF_Font, text : bytes, color : SDL_Color, center = False):
        texture, width, height = self.create_text_texture(font, text, color)

        if center:
            dst_rect = SDL_Rect(x - math.floor(width / 2), y, width, height)
        else:
            dst_rect = SDL_Rect(x, y, width, height)

        SDL_RenderCopy(self.renderer, texture, None, dst_rect)
        SDL_DestroyTexture(texture)

    def run(self) -> None:
        # dont initialize if already initialized
        if self.running: pass
        SDL_Init(SDL_INIT_VIDEO)
        IMG_Init(IMG_INIT_PNG)
        TTF_Init()

        # this enables antialiasing for scaling and vsync
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, b"2")
        SDL_SetHint(SDL_HINT_RENDER_VSYNC, b"1")

        # create window
        self.window = SDL_CreateWindow(b"chess program v0.1", SDL_WINDOWPOS_CENTERED, 
            SDL_WINDOWPOS_CENTERED, self.window_width, self.window_height, SDL_WINDOW_SHOWN)

        # initialize the renderer for this window
        self.renderer = SDL_CreateRenderer(self.window, -1, SDL_RENDERER_ACCELERATED)
        SDL_SetRenderDrawBlendMode(self.renderer, SDL_BLENDMODE_BLEND)

        # load piece sprites
        self.piece_sprites = IMG_LoadTexture(self.renderer, b"./pieces.png")

        # load the ui font
        self.font = TTF_OpenFont(b"./font.ttf", 20)

        w = ctypes.pointer(ctypes.c_int(0))
        h = ctypes.pointer(ctypes.c_int(0))
        SDL_QueryTexture(self.piece_sprites, None, None, w, h)

        self.piece_sprite_width = math.floor(w.contents.value / 6)
        self.piece_sprite_height = math.floor(h.contents.value / 2)

        self.running = True
        event = SDL_Event()
        while self.running:
            # event handling loop
            while SDL_PollEvent(ctypes.byref(event)) != 0:
                if event.type == SDL_QUIT:
                    self.running = False
                    break

                if event.type == SDL_MOUSEBUTTONDOWN:
                    if event.button.button == SDL_BUTTON_LEFT and not self.mouse_down:
                        self.mouse_down = True
                        self.mouse_click = True
                        self.mouse_click_x = event.button.x
                        self.mouse_click_y = event.button.y

                if event.type == SDL_MOUSEBUTTONUP:
                    if event.button.button == SDL_BUTTON_LEFT and self.mouse_down:
                        self.mouse_down = False

            # frame logic
            self.frame_logic()

            # first we clear the frame with black color
            SDL_SetRenderDrawColor(self.renderer, 21, 88, 67, 255)
            SDL_RenderClear(self.renderer)

            # draw the ui
            self.draw()

            # reset input flags if needed
            self.mouse_click = False

            # end the frame, present the window
            SDL_RenderPresent(self.renderer)

        # free resources
        SDL_DestroyTexture(self.piece_sprites)
        SDL_DestroyWindow(self.window)
        SDL_Quit()
        IMG_Quit()
        TTF_Quit()

    def select_piece(self, square):
        self.selection_square = square

        for move in self.board.legal_moves:
            if move.from_square == square:
                self.mark_square(move.to_square)

    def count_pieces(self, color, piece):
        square_set = self.board.pieces(piece, color)
        square_list = square_set.tolist()
        return square_list.count(True)

    def make_material_dict(self, color):
        return {
            chess.QUEEN: max(1 - self.count_pieces(not color, chess.QUEEN), 0),
            chess.ROOK: max(2 - self.count_pieces(not color, chess.ROOK), 0),
            chess.BISHOP: max(2 - self.count_pieces(not color, chess.BISHOP), 0),
            chess.KNIGHT: max(2 - self.count_pieces(not color, chess.KNIGHT), 0),
            chess.PAWN: max(8 - self.count_pieces(not color, chess.PAWN), 0)
        }

    def calculate_material(self):
        self.black_material = self.make_material_dict(chess.BLACK)
        self.white_material = self.make_material_dict(chess.WHITE)

    def check_victory(self):
        outcome = self.board.outcome()
        if outcome != None:    
            self.game_active = False
               
            if outcome.winner == chess.WHITE:
                self.game_message = "white win"
            elif outcome.winner == chess.BLACK:
                self.game_message = "black win"
            else:
                self.game_message = "draw"

    def frame_logic(self):
        if not self.game_active:
            return

        player = None
        
        # select the player and clear the waiting flags, we clear the flag opposite
        # to the current color because we can assume that the async player made a 
        # move already
        if self.board.turn == chess.BLACK:
            player = self.black_player
            
            if self.waiting_white:
                self.check_victory()
                self.calculate_material()
                self.waiting_white = False
        else:
            player = self.white_player
            
            if self.waiting_black:
                self.check_victory()
                self.calculate_material()
                self.waiting_black = False

        # if the current player is None then it means that player has no client
        # thus they are allowed to make a move within the application window
        if player == None:
            self.perspective = self.board.turn

            # waiting status
            if self.board.turn == chess.WHITE and not self.waiting_white:
                self.waiting_white = True
            elif self.board.turn == chess.BLACK and not self.waiting_black:
                self.waiting_black = True

            # really long and convoluted logic related to the ui making a move
            if self.mouse_click:
                mx = self.mouse_click_x - self.chessboard_x
                my = self.mouse_click_y - self.chessboard_y
                square_size = math.floor(self.chessboard_dim / 8)

                # if this is a valid click inside of the chessboard
                if mx > 0 and my > 0 and mx < self.chessboard_dim and my < self.chessboard_dim:
                    sx = math.floor(mx / square_size)
                    sy = math.floor(my / square_size)
                    square = chess.A1 + self.coords_to_square(sx, sy)
                    piece = self.board.piece_at(square)

                    # if square is not selected then the player may only select a piece to move
                    if self.selection_square == None:
                        if piece != None and piece.color == self.board.turn:
                            self.select_piece(square)
                    else:
                        if square == self.selection_square:
                            # if square selected and the same square is clicked then turn the selection off
                            self.clear_selection()
                        elif self.selection_mask[8 * sy + sx] == 1:
                            # if a marked square is selected try to make a move ending in this square
                            try:
                                move = self.board.find_move(from_square=self.selection_square, to_square=square)
                                self.board.push(move)
                                self.clear_selection()
                            except:
                                print("invalid move!")
                        elif piece != None and piece.color == self.board.turn:
                            # if another figure selected change selection to that figure
                            self.clear_selection()
                            self.select_piece(square)

        else: 
            # this is the logic regarding any players extending the Player class
            # basically we want the players to be asynchronous, thus the make_move method
            # is started in separate thread

            if self.board.turn == chess.WHITE and not self.waiting_white:
                thread = Thread(target = player.make_move, args=[self.board])
                thread.start()

                self.waiting_white = True
            elif self.board.turn == chess.BLACK and not self.waiting_black:
                thread = Thread(target = player.make_move, args=[self.board])
                thread.start()

                self.waiting_black = True

    # this method draws a piece on the square defined by x,y,w,h
    def draw_piece(self, x, y, w, h, piece):
        piece_type = piece.piece_type
        color = 0
        if piece.color == chess.BLACK: color = 1
        
        src_rect = SDL_Rect(
            x = (6 - piece_type) * self.piece_sprite_width,
            y = color * self.piece_sprite_height,
            w = self.piece_sprite_width,
            h = self.piece_sprite_height
        )

        SDL_RenderCopy(self.renderer, self.piece_sprites, src_rect, SDL_Rect(x, y, w, h))

    # this is the main drawing method, it draws the chessboard, the pieces
    # and everything related to the game on the screen
    def draw(self):
        square_size = math.floor(self.chessboard_dim / 8)

        # draw the chessboard, we assume it is 8x8
        for sx in range(0,8):
            for sy in range(0,8):
                posx = self.chessboard_x + square_size * sx
                posy = self.chessboard_y + square_size * sy

                rect = SDL_Rect(posx, posy, square_size, square_size)
                color_index = sx + (sy % 2)

                # draw the chessboard rectangle, the palette can be adjusted here
                if color_index % 2 == 0: SDL_SetRenderDrawColor(self.renderer, 240, 217, 181, 255)
                else: SDL_SetRenderDrawColor(self.renderer, 181, 136, 99, 255)

                SDL_RenderFillRect(self.renderer, rect)

                # render piece if there is a piece on this square
                index = 8 * sy + sx
                square = chess.A1 + self.coords_to_square(sx, sy)
                piece = self.board.piece_at(square)

                if piece != None:
                    self.draw_piece(posx, posy, square_size, square_size, piece)

                # if this is the selected square or a square marked as a valid move this turn
                # then draw a "glow" over it
                if square == self.selection_square or self.selection_mask[index] == 1:
                    SDL_SetRenderDrawColor(self.renderer, 190, 255, 0, 90)
                    SDL_RenderFillRect(self.renderer, rect)

        # draw the cool sidebar with playerson the right
        white_name = "player #1"
        black_name = "player #2"

        if self.black_player != None:
            black_name = self.black_player.get_name()

        if self.white_player != None:
            white_name = self.white_player.get_name()

        if self.perspective == chess.WHITE:
            self.draw_player_profile(self.window_height + 10, 10, black_name, chess.BLACK)
            self.draw_player_profile(self.window_height + 10, 340, white_name + " (you)", chess.WHITE)
        else:
            self.draw_player_profile(self.window_height + 10, 10, white_name, chess.WHITE)
            self.draw_player_profile(self.window_height + 10, 340, black_name + " (you)", chess.BLACK)

        # draw game over message if game is over
        if not self.game_active:
            tx = 520 + 90
            ty = math.floor(self.window_height / 2) - 20

            SDL_SetRenderDrawColor(self.renderer, 0, 0, 0, 255)
            SDL_RenderFillRect(self.renderer, SDL_Rect(tx - 80, ty - 40, 160, 100))

            self.render_text(tx, ty, self.font, self.game_message.encode(), SDL_Color(255, 0, 0), True)

    def draw_player_profile(self, x, y, name, color):
        self.draw_piece(x + 35, y, 90, 90, chess.Piece(chess.KING, color))
        self.render_text(x + 80, y + 100, self.font, name.encode(), SDL_Color(), True)
        
        # render material for this player
        material = self.white_material
        if color == chess.BLACK: 
            material = self.black_material

        index = 0
        for piece_type in material:
            for i in range(0, material[piece_type]):
                dx = index % 8
                dy = math.floor(index / 8)
                spr_x = x + 20 + dx * 15
                spr_y = y + 125 + dy * 15

                self.draw_piece(spr_x, spr_y, 25, 25, chess.Piece(piece_type, not color))
                index = index + 1

def main():
    # '8/8/8/4p1K1/2k1P3/8/8/8'
    # 'r7/8/3k4/8/8/4K3/8/5QRR'
    # '3k2R1/7R/8/8/8/4K3/1r6/r4Q2'
    game = ChessGame(white = NeuralNetworkPlayer(checkpoint_name='t3/checkpoint.ckpt'), black = None, fen = None, start_perspective=chess.BLACK)
    #game = ChessGame(white = NeuralNetworkPlayer(checkpoint_name='t3/checkpoint.ckpt'))
    game.run()

if __name__ == "__main__":
    sys.exit(main())