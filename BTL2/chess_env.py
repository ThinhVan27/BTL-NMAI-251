import chess
import numpy as np


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.action_space_size = 4096

        self.weights = {
            # basic piece values used inside _get_potential
            'pawn': 1.0,
            'knight': 3.0,
            'bishop': 3.2,
            'rook': 5.0,
            'queen': 9.0,

            # positional / misc – mostly OFF for now
            'king_safety': 0.0,
            'mobility': 0.0,
            'center': 0.0,
            'pst_scale': 0.0,

            # shaping & misc
            'step_penalty': -0.05,      # small pressure to finish
            'check': 0.0,               # no explicit check bonus
            'castling': 0.5,            # small bonus
            'repetition_penalty': -3.0,
            'promotion': 5.0,
            'pressure': 0.1,
        }

        self.max_moves = 200
        self.move_count = 0
        self.last_move = None

        # simple piece-square tables for shaping (kept but currently scaled down)
        self.pst_pawn = [
             0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
             5,  5, 10, 25, 25, 10,  5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5, -5,-10,  0,  0,-10, -5,  5,
             5, 10, 10,-20,-20, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0,
        ]
        self.pst_knight = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50,
        ]

    def reset(self):
        """Reset environment. Currently: always sets up an endgame position."""
        self.move_count = 0
        self.last_move = None

        # For now: pure endgame curriculum
        self._setup_robust_endgame()
        return self.get_state()

        # If you want a mix later:
        # if np.random.random() < 0.4:
        #     self._setup_robust_endgame()
        # else:
        #     self.board.reset()

    def _setup_robust_endgame(self):
        """Random but legal-ish endgame positions with both kings and a few pieces."""
        self.board.clear()

        def place_piece(piece_type, color):
            tries = 0
            while True:
                sq = np.random.randint(0, 64)
                if self.board.piece_at(sq) is None:
                    self.board.set_piece_at(sq, chess.Piece(piece_type, color))
                    break
                tries += 1
                if tries > 100:
                    break

        # place kings
        while True:
            wk = np.random.randint(0, 64)
            bk = np.random.randint(0, 64)
            if wk != bk and chess.square_distance(wk, bk) > 1:
                self.board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
                self.board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                break

        # add some minor pieces & pawns
        if np.random.random() < 0.5:
            # few pieces
            num_pieces = np.random.randint(1, 3)
            for _ in range(num_pieces):
                pt = np.random.choice(
                    [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                )
                color = chess.WHITE if np.random.random() < 0.5 else chess.BLACK
                place_piece(pt, color)
        else:
            # pawns + minors
            num_pieces = np.random.randint(2, 5)
            for _ in range(num_pieces):
                pt = np.random.choice(
                    [chess.PAWN, chess.KNIGHT, chess.ROOK, chess.BISHOP]
                )
                color = chess.WHITE if np.random.random() < 0.6 else chess.BLACK
                place_piece(pt, color)

        self.board.turn = chess.WHITE

        # ensure not trivial / immediately over
        if self.board.is_game_over() or self.board.is_check():
            self._setup_robust_endgame()

    def get_state(self):
        """
        13x8x8 tensor:
        - 0..5: white pawn, knight, bishop, rook, queen, king
        - 6..11: black pawn..king
        - 12: side-to-move plane
        """
        state = np.zeros((13, 8, 8), dtype=np.float32)
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if not piece:
                continue
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            state[channel, rank, file] = 1.0

        # side to move
        state[12, :, :] = 1.0 if self.board.turn == chess.WHITE else 0.0
        return state

    # === Reward helpers ===

    def _material_score(self, board: chess.Board):
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        w = b = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if not piece:
                continue
            val = values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                w += val
            else:
                b += val
        return w, b

    def _get_potential(self, board: chess.Board) -> float:
        """
        Heuristic evaluation from **white's** perspective.
        Positive = white better; negative = black better.
        """
        w_mat, b_mat = self._material_score(board)
        result = (w_mat - b_mat) * self.weights['pawn']

        # piece-square table shaping (optional, currently scaled by pst_scale=0)
        pst_scale = self.weights['pst_scale']
        if pst_scale != 0.0:
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if not piece:
                    continue
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)

                if piece.piece_type == chess.PAWN:
                    idx = rank * 8 + file
                    val = self.pst_pawn[idx]
                    if piece.color == chess.WHITE:
                        result += pst_scale * val / 100.0
                    else:
                        result -= pst_scale * val / 100.0

                if piece.piece_type == chess.KNIGHT:
                    idx = rank * 8 + file
                    val = self.pst_knight[idx]
                    if piece.color == chess.WHITE:
                        result += pst_scale * val / 100.0
                    else:
                        result -= pst_scale * val / 100.0

        return result

    # === Step ===

    def step(self, action_idx: int):
        move = self.decode_action(action_idx)

        # illegal move → big penalty, terminate
        if move not in self.board.legal_moves:
            return self.get_state(), -10.0, True, {"legal": False}

        prev_potential = self._get_potential(self.board)
        castling_bonus = self.weights['castling'] if self.board.is_castling(move) else 0.0

        # apply move
        self.board.push(move)
        self.move_count += 1

        natural_done = self.board.is_game_over(claim_draw=True)
        reached_max_moves = (self.move_count >= self.max_moves and not natural_done)
        done = natural_done or reached_max_moves

        curr_potential = self._get_potential(self.board)

        # agent_color = side that just moved
        agent_color = not self.board.turn

        diff = curr_potential - prev_potential
        shaped = diff if agent_color == chess.WHITE else -diff

        # main shaping term
        material_scale = 0.1
        reward = material_scale * shaped

        # repetition penalty
        if self.board.is_repetition(2):
            reward += self.weights['repetition_penalty']

        # step penalty
        reward += self.weights['step_penalty']

        # anti-backtracking: directly undo previous move
        if (
            self.last_move is not None
            and move.from_square == self.last_move.to_square
            and move.to_square == self.last_move.from_square
        ):
            reward -= 0.5
        self.last_move = move

        # terminal rewards
        if done:
            w_mat, b_mat = self._material_score(self.board)
            agent_mat = w_mat if agent_color == chess.WHITE else b_mat
            opp_mat = b_mat if agent_color == chess.WHITE else w_mat

            outcome = self.board.outcome(claim_draw=True)

            # true win/loss
            if outcome is not None and outcome.winner is not None:
                if outcome.winner == agent_color:
                    reward += 200.0
                else:
                    reward -= 200.0
            else:
                # draw / stalemate / insufficient / max-move
                if self.board.is_stalemate() or self.board.is_insufficient_material():
                    # stalemate from a clearly winning position → punish
                    if agent_mat > opp_mat + 3:
                        reward -= 20.0

                if reached_max_moves:
                    diff_mat = agent_mat - opp_mat
                    if diff_mat > 0:
                        reward += diff_mat * 1.0 - 20.0  # up material but failed to convert
                    elif diff_mat < 0:
                        reward += diff_mat * 1.0        # losing anyway

        return self.get_state(), reward, done, {"legal": True}

    # === Action encoding/decoding ===

    def encode_action(self, move: chess.Move) -> int:
        return move.from_square * 64 + move.to_square

    def decode_action(self, action_idx: int) -> chess.Move:
        from_square = action_idx // 64
        to_square = action_idx % 64
        move = chess.Move(from_square, to_square)

        # auto-queen promotions
        piece = self.board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_square)
            if (
                (piece.color == chess.WHITE and rank == 7)
                or (piece.color == chess.BLACK and rank == 0)
            ):
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

        return move

    def get_legal_actions(self):
        return [self.encode_action(m) for m in self.board.legal_moves]
