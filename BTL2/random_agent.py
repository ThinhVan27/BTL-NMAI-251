import chess
import random
from chess import Move, Board

from agent import Agent


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
        
    def get_action(self, game_state: Board) -> Move:
        valid_moves = game_state.legal_moves
        return random.choice(list(valid_moves))