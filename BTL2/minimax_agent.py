import chess
import random
from chess import Move, Board
from agent import Agent

from utils import *

check_mate = 10000
stale_mate = 0
DEPTH = 3


class MinimaxAgent(Agent):
    def __init__(self, depth=DEPTH):
        super().__init__()
        self.depth = depth
        
    def get_action(self, game_state: Board) -> Move:
        return self._get_best_move(game_state, self.depth)
    
    def _get_best_move(self, game_state: Board, depth: int = DEPTH) -> Move:
        """Get the best move following Minimax Algorithm"""
        
        if depth == 0:
            return None
        
        _, next_move = self._get_minimax_score(game_state, depth)
        return next_move
    
    def _get_minimax_score(self, game_state: Board, depth: int = DEPTH, alpha: int = -check_mate, beta: int = check_mate):
       
        if depth == 0:
            return self._score_board(game_state), None
        
        next_move = None
        legal_moves = list(game_state.legal_moves)
        random.shuffle(legal_moves)
        if game_state.turn: # WHITE turn
            for move in legal_moves:
                game_state.push(move)
                succ_score, _ = self._get_minimax_score(game_state, depth-1, alpha, beta)
                game_state.pop()
                
                if succ_score > alpha:
                    alpha = succ_score
                    next_move = move
                
                if alpha >= beta:
                    return alpha, next_move
            return alpha, next_move
        else: # BLACK turn
            for move in legal_moves:
                game_state.push(move)
                succ_score, _ = self._get_minimax_score(game_state, depth-1, alpha, beta)
                game_state.pop()
                
                if succ_score < beta:
                    beta = succ_score
                    next_move = move
                
                if alpha >= beta:
                    return beta, next_move
            return beta, next_move
    
    
    def _score_board(self, game_state: Board) -> int:
        """Get the score for current gs. WHITE favors positive score, while BLACK favors negative score"""
        if game_state.is_checkmate():
            return -check_mate if game_state.turn else check_mate
        if game_state.is_stalemate():
            return 0
        
        score = 0
        for r in range(8):
            for c in range(8):
                piece = get_piece(game_state, r, c)
                upper_piece = piece.upper()
                if piece != '.':
                    pos_score = 0
                    if upper_piece != "K":
                        if upper_piece == "P":
                            pos_score = piecePosScores[piece][r][c]
                        else:
                            pos_score = piecePosScores[piece.upper()][r][c]
                    if piece == upper_piece: # WHITE piece
                        score += pieceScore[upper_piece] + pos_score
                    else:
                        score -= pieceScore[upper_piece] + pos_score
        return score