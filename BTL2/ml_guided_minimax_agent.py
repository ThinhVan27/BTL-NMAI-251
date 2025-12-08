import random
import chess
import torch

from agent import Agent
from chess_env_v2 import ChessEnv
from policy_net import PolicyNet
from utils import get_piece, pieceScore, piecePosScores


def policy_scores_for_board(policy: PolicyNet, env: ChessEnv, board: chess.Board, device) -> dict:
    """
    Return a mapping from legal chess.Move -> policy logit score for the given board.
    Illegal moves are masked out using the env action encoding.
    """
    env.board = board
    state = env.get_state()
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = policy(state_t)[0]

    legal_idxs = env.get_legal_actions()
    mask = torch.full((policy.num_actions,), float("-inf"), device=device)
    mask[legal_idxs] = 0.0
    masked_logits = logits + mask

    scores = {}
    for idx in legal_idxs:
        move = env.decode_action(idx)
        scores[move] = masked_logits[idx].item()
    return scores


class MLGuidedMinimaxAgent(Agent):
    """
    Minimax/negamax search guided by a policy network for move ordering and pruning.
    """

    def __init__(
        self,
        policy: PolicyNet,
        depth: int = 3,
        top_k: int = 6,
        device=None,
    ):
        super().__init__()
        self.policy = policy
        self.depth = depth
        self.top_k = top_k
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.env = ChessEnv()
        self.check_mate = 10000
        self.stale_mate = 0

    def get_action(self, game_state: chess.Board) -> chess.Move:
        board = game_state.copy()
        if board.is_game_over(claim_draw=True):
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else None

        color_sign = 1 if board.turn == chess.WHITE else -1
        scores = policy_scores_for_board(self.policy, self.env, board, self.device)
        ordered_moves = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
        ordered_moves = ordered_moves[: self.top_k] if self.top_k > 0 else ordered_moves

        best_move = None
        alpha = -float("inf")
        beta = float("inf")

        for move in ordered_moves:
            board.push(move)
            val = -self._negamax(board, self.depth - 1, -beta, -alpha, -color_sign)
            board.pop()
            if val > alpha:
                alpha = val
                best_move = move
            if alpha >= beta:
                break

        if best_move is None:
            legal_moves = list(game_state.legal_moves)
            best_move = random.choice(legal_moves) if legal_moves else None
        return best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, color_sign: int) -> float:
        if depth == 0 or board.is_game_over(claim_draw=True):
            return color_sign * self._evaluate_board(board)

        scores = policy_scores_for_board(self.policy, self.env, board, self.device)
        ordered_moves = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
        ordered_moves = ordered_moves[: self.top_k] if self.top_k > 0 else ordered_moves

        best_val = -float("inf")
        for move in ordered_moves:
            board.push(move)
            val = -self._negamax(board, depth - 1, -beta, -alpha, -color_sign)
            board.pop()
            if val > best_val:
                best_val = val
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break
        return best_val

    def _evaluate_board(self, board: chess.Board) -> float:
        """Material + positional evaluation, white-positive, mirroring minimax agent."""
        if board.is_checkmate():
            return -self.check_mate if board.turn else self.check_mate
        if board.is_stalemate():
            return self.stale_mate

        score = 0.0
        for r in range(8):
            for c in range(8):
                piece = get_piece(board, r, c)
                upper = piece.upper()
                if piece != ".":
                    pos_score = 0.0
                    if upper != "K":
                        if upper == "P":
                            pos_score = piecePosScores[piece][r][c]
                        else:
                            pos_score = piecePosScores[upper][r][c]
                    if piece == upper:
                        score += pieceScore[upper] + pos_score
                    else:
                        score -= pieceScore[upper] + pos_score
        return score
