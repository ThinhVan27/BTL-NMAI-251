import chess
from chess import Move, Board
from agent import Agent

class RLAgent(Agent):
    def __init__(self):
        super().__init__()
        
    def get_action(self, game_state: Board):
        pass