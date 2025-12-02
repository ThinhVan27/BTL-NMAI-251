# Test Engine
import io
import time
import chess
import chess.pgn
from chess import Board, Move
import argparse

from agent import Agent
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
from rl_agent_big import RLAgent

def play(agent1: Agent, agent2: Agent, interval: float = 0, pgn: bool = False, verbose: bool = False, save: bool = False):
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
    max_steps = 300
    step_count = 0
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
            print("-"*50)
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
        
        step_count += 1
        node = node.add_variation(move)
        if verbose:
            print(board)
            print("-"*50)
            if save:
                with open("results.txt", "a") as f:
                    f.write(f'{board}\n{"="*50}\n')
        if board.is_game_over():
            outcome = board.outcome()
            break
        if step_count >= max_steps:
            break
    
    pgn_text = ""
    length = step_count
    if pgn:
        # length = len(str(game))
        pgn_text = str(game).split('\n\n',1)[1]
        length = len(pgn_text)

    winner = "-1"
            
    if outcome is not None:
        if outcome.winner == True:
            winner = "1"
        elif outcome.winner == False:
            winner = "0"
        else:
            winner = "-1"
    
    return (winner, pgn_text, length)
        
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="models/chess_2000.pth")
parser.add_argument("--interval", type=float, default=0.0)
parser.add_argument("--pgn", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--N", type=int, default=50)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    a1 = MinimaxAgent()
    a2 = RandomAgent()
    
    
    name1 = a1.__class__.__name__
    name2 = a2.__class__.__name__
    
    print("*"*50)
    print(f"Game Play: {name1} vs {name2}")
    print("*"*50)


    if isinstance(a1, RLAgent):
        a1.load(args.path)
    
    if isinstance(a2, RLAgent):
        a2.load(args.path)
        
    win = 0
    N = args.N
    for i in range(N):
        winner, pgn, len = play(a1, a2, args.interval, args.pgn, args.verbose, args.save)
        if winner == "1":
            win += 1
        print(f"[INFO] Complete game {i}.")
        print(f"[INFO] PGNs: {pgn}")
        print(f"[INFO] Winner: {name1}, in: {len} steps.")
        print("=" * 50)
    
    if args.save:
        with open("results.txt", "a") as f:
            f.write(f"PGNs: {pgn}\n ")
    
        
    print(f"[INFO] {name1} winrate: {win/N*100:.2f}%")
    
    # print(chess.__file__)