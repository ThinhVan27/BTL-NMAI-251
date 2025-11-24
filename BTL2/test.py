# Test Engine
import io
import time
import chess
import chess.pgn
from chess import Board, Move

from agent import Agent
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent

def play(agent1: Agent, agent2: Agent, interval: float = 0, pgn: bool = False, verbose: bool = False):
    """Play a chess game with two agents - WHITE and BLACK respectively
    
    Parameters:
        - agent1 (Agent) WHITE player.
        - agent2 (Agent) BLACK player.
        - interval (float) the waiting time between two agents.
        - pgn (Boolean) get PGN string of game.
        - verbose (Boolean) logging game ASCII-based visualization.
    
    Return:
        - str: '1' if WHITE wins, '0' if BLACK wins, '-1' if DRAW
        - PGN: optional, game description
    """
    

    outcome = None
    
    game = chess.pgn.Game()
    
    node = game
    board = Board()
    while True:
        # Agent1 moves
        time.sleep(interval)
        move = agent1.get_action(board)
        try:
           board.push(move)
        except ValueError as e:
            print("[ERROR] No valid move!")
            return
        node = node.add_variation(move)
        if verbose:
            print(board)
            print("="*50)
        if board.is_game_over():
            outcome = board.outcome()
            break
        
        # Agent2 moves
        time.sleep(interval)
        move = agent2.get_action(board)
        try:
            board.push(move)
        except ValueError as e:
            print("[ERROR] No valid move!")
            return
        
        
        node = node.add_variation(move)
        if verbose:
            print(board)
            print("="*50)
        if board.is_game_over():
            outcome = board.outcome()
            break
    
    pgn_text = ""
    if pgn:
        pgn_text = str(game).split('\n\n',1)[1]
            
    if outcome is not None:
        if outcome.winner == True:
            winner = "1"
        elif outcome.winner == False:
            winner = "0"
        else:
            winner = "-1"
    
    return (winner, pgn_text) if pgn else winner
        
    

if __name__ == "__main__":
    a1 = RandomAgent()
    a2 = MinimaxAgent()
    win = 0
    N = 1
    for i in range(N):
        winner, _ = play(a1, a2, 0.1, True, False)
        if winner == "0":
            win += 1
        print(f"[INFO] Complete game {i}.")
        print(f"[INFO] PGNs: {_}")
    
    print(f"[INFO] Minimax winrate: {win/N*100:.2f}%")
    
    # print(chess.__file__)