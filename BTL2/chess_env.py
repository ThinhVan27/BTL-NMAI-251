# Chess environment for RL 
import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.action_space_size = 4096
        
        # Reward Config
        # self.weights = {
        #     'pawn': 1.0,
        #     'knight': 3.0,
        #     'bishop': 3.2,
        #     'rook': 5.0,
        #     'queen': 9.0,
        #     # Drastically reduced positional noise. vs Random, only material matters.
        #     'mobility': 0.001, 
        #     'center': 0.01,
        #     'pst_scale': 0.0, # Disable PST initially to focus on pure material capture
        #     'step_penalty': -0.05, # Encourage faster wins
        #     'check': 0.2,
        #     'castling': 1.0, # Good practice
        #     'king_safety': 0.1,
        #     'repetition_penalty': -1.0 # Stronger penalty for stalling
        # }

        self.weights = {
            'pawn': 1.0,
            'knight': 3.0,
            'bishop': 3.2,
            'rook': 5.0,
            'queen': 9.0,
            'king_safety': 0.0,      # Disable for now (noise)
            'mobility': 0.0,         # Disable for now (noise)
            'center': 0.0,           # Disable for now (noise)
            'pst_scale': 0.0,        # Disable PST initially to focus on pure material capture
            'step_penalty': -0.05,   # Small pressure to finish games
            'check': 3.0,            # Helpful tactile feedback
            'castling': 1.0,         # Good safe habit
            'repetition_penalty': -5.0 
        }
        
        # Simple Piece-Square Tables (Pawn & Knight)
        # Scaled 0-100, centered on mid-game principles
        self.pst_pawn = [
             0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
             5,  5, 10, 25, 25, 10,  5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5, -5,-10,  0,  0,-10, -5,  5,
             5, 10, 10,-20,-20, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        ]
        self.pst_knight = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]

    def reset(self):
        self.board.reset()
        return self.get_state()

    def step(self, action_idx):
        move = self.decode_action(action_idx)
        if move not in self.board.legal_moves:
            return self.get_state(), -10.0, True, {"legal": False} # Penalty for illegal

        # 1. Calculate Potential BEFORE move
        prev_potential = self._get_potential(self.board)
        
        # Check for castling bonus (before move)
        castling_bonus = 0.0
        if self.board.is_castling(move):
            castling_bonus = self.weights['castling']

        # 2. Execute Move
        self.board.push(move)
        done = self.board.is_game_over()

        # 3. Calculate Potential AFTER move
        curr_potential = self._get_potential(self.board)
        
        # Check bonus (after move, is opponent in check?)
        check_bonus = 0.0
        if self.board.is_check():
            check_bonus = self.weights['check']
            
        # Repetition Penalty
        repetition_penalty = 0.0
        if self.board.is_repetition(2): # 2-fold repetition
            repetition_penalty = self.weights['repetition_penalty']

        # 4. Reward Shaping (Difference in Potential)
        # If I am white, I want potential to increase.
        # If I am black, I want potential to decrease (since eval is usually White-centric)
        # Note: self.board.turn is now the OPPONENT's turn after push.
        # So if we just moved White, board.turn is Black.
        
        # We need the color of the agent who JUST moved.
        agent_color = not self.board.turn 
        
        # Standard RL perspective: Reward is for the Agent.
        diff = curr_potential - prev_potential
        reward = diff if agent_color == chess.WHITE else -diff
        
        # Add Bonuses and Penalties
        reward += castling_bonus
        reward += check_bonus
        reward += repetition_penalty
        
        # Add Step Penalty (Time pressure)
        reward += self.weights['step_penalty']

        # Terminal Rewards (Override shaping for clear outcomes)
        if done:
            if self.board.is_checkmate():
                # Massive reward for winning.
                reward += 100.0 
                # reward += 20.0
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                # Draw is better than losing, but worse than winning.
                reward += 0.0

        return self.get_state(), reward, done, {"legal": True}

    def _get_potential(self, board):
        """
        Calculates a dense evaluation of the board from White's perspective.
        """
        score = 0
        
        # 1. Material
        pm = board.piece_map()
        for sq, piece in pm.items():
            val = 0
            if piece.piece_type == chess.PAWN: val = self.weights['pawn']
            elif piece.piece_type == chess.KNIGHT: val = self.weights['knight']
            elif piece.piece_type == chess.BISHOP: val = self.weights['bishop']
            elif piece.piece_type == chess.ROOK: val = self.weights['rook']
            elif piece.piece_type == chess.QUEEN: val = self.weights['queen']
            
            # PST
            pst_val = 0
            if self.weights['pst_scale'] > 0:
                # Calculate rank/file (0-7)
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                idx = (7 - rank) * 8 + file # Map to 0=a8, 63=h1 table layout
                
                # Mirror for black
                if piece.color == chess.BLACK:
                    rank = 7 - rank # Mirror rank
                    idx = (7 - rank) * 8 + file

                if piece.piece_type == chess.PAWN: pst_val = self.pst_pawn[idx]
                elif piece.piece_type == chess.KNIGHT: pst_val = self.pst_knight[idx]
                
            total_piece_val = val + (pst_val * 0.01 * self.weights['pst_scale'])
            
            if piece.color == chess.WHITE:
                score += total_piece_val
            else:
                score -= total_piece_val

        # 2. Center Control (Bonus for occupying center)
        # Simple check for pieces on e4, d4, e5, d5
        for sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
            p = board.piece_at(sq)
            if p:
                if p.color == chess.WHITE: score += self.weights['center']
                else: score -= self.weights['center']

        # 3. Mobility (Number of legal moves)
        # Note: python-chess legal_moves is for the current turn side only.
        # Approximating mobility is expensive if we flip turns. 
        # For efficiency, we might skip or just use current side.
        # Here is a simple implementation adding current side mobility:
        legal_count = board.legal_moves.count()
        mobility_score = legal_count * self.weights['mobility']
        
        if board.turn == chess.WHITE:
            score += mobility_score
        else:
            score -= mobility_score
            
        return score

    def get_state(self):
        """
        Converts board to 12x8x8 numpy array.
        Channels:
        0-5: White P, N, B, R, Q, K
        6-11: Black P, N, B, R, Q, K
        """
        state = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                channel = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6
                
                state[channel, rank, file] = 1
                
        return state

    def encode_action(self, move: chess.Move) -> int:
        """Encodes a chess move into an integer 0-4095."""
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square

    def decode_action(self, action_idx: int) -> chess.Move:
        from_square = action_idx // 64
        to_square = action_idx % 64
        move = chess.Move(from_square, to_square)
        
        # --- FIX: Auto-promote to Queen ---
        # Check if this is a pawn move to the last rank
        piece = self.board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and rank == 7) or \
            (piece.color == chess.BLACK and rank == 0):
                move.promotion = chess.QUEEN # Auto-promote
                
        return move

    def get_legal_actions(self):
        """Returns list of legal action indices."""
        legal_moves = []
        for move in self.board.legal_moves:
            legal_moves.append(self.encode_action(move))
        return legal_moves