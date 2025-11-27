import chess
import numpy as np
from chess_env import ChessEnv
from rl_agent import RLAgent
from random_agent import RandomAgent
import argparse
# from minimax_agent import MinimaxAgent # Only needed if you want Phase 2
import os

def evaluate_agent(agent, opponent, num_games=20):
    wins = 0
    draws = 0
    losses = 0
    
    # Force exploitation during evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for _ in range(num_games):
        env = ChessEnv()
        state = env.reset()
        done = False
        board = env.board
        
        while not done:
            if board.turn == chess.WHITE:
                legal_moves = env.get_legal_actions()
                action_idx = agent.get_action(state, legal_moves)
                move = env.decode_action(action_idx)
                
                if move not in board.legal_moves:
                    losses += 1; done = True; break
                board.push(move)
            else:
                if board.is_game_over(): break
                move = opponent.get_action(board)
                board.push(move)
            
            state = env.get_state() # Update state for next step
            
            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner == chess.WHITE: wins += 1
                elif outcome.winner == chess.BLACK: losses += 1
                else: draws += 1
                done = True

    agent.epsilon = original_epsilon
    return wins / num_games

def train(args):
    env = ChessEnv()
    agent = RLAgent()

    episode = 0

    if args.load:
        agent.load(args.load, True)
        episode = int(args.load.split("_")[-1].split(".")[0])
    
    # CRITICAL CHANGE: Train against RandomAgent first!
    train_opponent = RandomAgent()
    eval_opponent = RandomAgent()
    
    episodes = 5000
    target_update_freq = 20
    
    if not os.path.exists("models"): os.makedirs("models")
    
    for episode in range(episode, episodes):
        state = env.reset()
        done = False
        max_steps = 200 # Games vs Random shouldn't take forever
        step_count = 0
        total_reward = 0
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # --- Agent Turn (White) ---
            legal_moves = env.get_legal_actions()
            action_idx = agent.get_action(state, legal_moves)
            
            next_state, reward, done, info = env.step(action_idx)
            # total_reward += reward
            
            # --- Opponent Turn (Black) ---
            if not done:
                opp_move = train_opponent.get_action(env.board)
                opp_action_idx = env.encode_action(opp_move)
                
                # We care about the state AFTER opponent moves
                next_state_final, opp_reward, done, info = env.step(opp_action_idx)
                
                # Reward Logic:
                # My Reward - Opponent Gain. 
                # If Opponent blunders (negative opp_reward), I get a bonus.
                opp_pure_reward = opp_reward - env.weights['step_penalty']
                if opp_pure_reward > 0:
                    combined_reward = reward - opp_pure_reward
                else:
                    combined_reward = reward
                
                agent.remember(state, action_idx, combined_reward, next_state_final, done)
                state = next_state_final
                total_reward += combined_reward
            else:
                agent.remember(state, action_idx, reward, next_state, done)
                state = next_state # Technically terminal
                total_reward += reward
            
            agent.update()

            
        if episode % target_update_freq == 0:
            agent.update_target_network()
        agent.decay_epsilon()
            
        print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Evaluation every 100 episodes
        if episode > 0 and episode % 100 == 0:
            win_rate = evaluate_agent(agent, eval_opponent, num_games=50)
            print(f"--- Eval Episode {episode}: Win Rate {win_rate*100:.1f}% ---")
            
            if win_rate >= 0.90:
                print("GOAL REACHED! Saving model.")
                agent.save(f"models/chess_90_percent.pth")
                # Optional: break or switch to harder opponent here

        if episode % 250 == 0:
            agent.save(f"models/chess_{episode}.pth")   

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, help="Path to model to load and continue training")
args = parser.parse_args()


if __name__ == "__main__":
    train(args)