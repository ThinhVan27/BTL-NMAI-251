import chess
import numpy as np
from chess_env import ChessEnv
from minimax_agent import MinimaxAgent
from rl_agent import RLAgent
from random_agent import RandomAgent
import os

def evaluate_agent(agent, opponent, num_games=5):
    """
    Evaluates the agent against an opponent.
    Returns the win rate (wins / total games).
    """
    wins = 0
    draws = 0
    losses = 0
    
    # Temporarily set epsilon to 0 for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for _ in range(num_games):
        env = ChessEnv()
        state = env.reset()
        done = False
        
        # Reset board
        board = env.board
        
        while not done:
            # White (Agent) moves
            if board.turn == chess.WHITE:
                legal_moves = env.get_legal_actions()
                action_idx = agent.get_action(state, legal_moves)
                move = env.decode_action(action_idx)
                
                if move not in board.legal_moves:
                    # Agent made illegal move -> Loss
                    losses += 1
                    done = True
                    break
                    
                board.push(move)
                
            # Black (Opponent) moves
            else:
                if board.is_game_over():
                    break
                    
                move = opponent.get_action(board)
                board.push(move)
                
            state = env.get_state()
            
            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner == chess.WHITE:
                    wins += 1
                elif outcome.winner == chess.BLACK:
                    losses += 1
                else:
                    draws += 1
                done = True

    agent.epsilon = original_epsilon
    return wins / num_games

def train():
    env = ChessEnv()
    agent = RLAgent()

    agent.load("models/dqn_chess_1950.pth", training=True)
    
    # Training opponent: Minimax (Depth 1 for speed/stability balance)
    train_opponent = MinimaxAgent(depth=2)
    
    # Evaluation opponent: Random (to track original goal)
    eval_opponent = RandomAgent()
    
    episodes = 2000
    target_update_freq = 10
    save_freq = 50
    eval_freq = 50
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # Dynamic Evaluation Config
    num_eval_games = 1
    best_win_rate = 0.0
        
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        max_steps = 200
        step_count = 0
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # --- Agent Turn (White) ---
            legal_moves = env.get_legal_actions()
            action_idx = agent.get_action(state, legal_moves)
            
            # Execute Agent Move
            next_state, reward, done, info = env.step(action_idx)
            
            if not done:
                # --- Opponent Turn (Black) ---
                # Opponent needs the board object to decide
                opp_move = train_opponent.get_action(env.board)
                
                # We need to encode this move to pass to env.step if we want to use env.step
                # But env.step expects action_idx and returns state/reward.
                # We can use env.encode_action(opp_move)
                opp_action_idx = env.encode_action(opp_move)
                
                # Execute Opponent Move
                # Note: env.step returns reward from the perspective of the player who moved.
                # So this reward is for Black.
                next_state_after_opp, opp_reward, done, info = env.step(opp_action_idx)
                
                # Combine rewards?
                # Agent reward (White) = reward
                # Opponent reward (Black) = opp_reward
                # Since it's zero-sum-ish, usually we want to penalize White if Black gains advantage.
                # My env reward is "diff in potential".
                # If Black improves position, opp_reward is positive (for Black).
                # This means White's position got worse.
                # So we should SUBTRACT opp_reward from White's total reward?
                # Wait, let's check env.step again.
                # "reward = diff if agent_color == chess.WHITE else -diff"
                # If Black moves, agent_color is Black.
                # diff = potential_after - potential_before.
                # If Black improves, potential (White-centric) decreases. diff is negative.
                # reward = -(-negative) = positive. Correct.
                # So opp_reward is positive if Black does well.
                # If Black does well, White should be punished.
                # So Total Reward for White = reward (White's move) - opp_reward (Black's move)?
                # Let's verify:
                # White move: improves potential by +1. reward = +1.
                # Black move: decreases potential by -1 (improves for Black). opp_reward = -(-1) = +1.
                # Net change in potential = 0.
                # If we sum: 1 + 1 = 2? No.
                # If we subtract: 1 - 1 = 0. Correct.
                # So we should subtract opp_reward.
                
                combined_reward = reward - opp_reward
                final_next_state = next_state_after_opp
                
            else:
                combined_reward = reward
                final_next_state = next_state
            
            agent.remember(state, action_idx, combined_reward, final_next_state, done)
            agent.update()
            
            state = final_next_state
            total_reward += combined_reward
            
        if episode % target_update_freq == 0:
            agent.update_target_network()
            
        # Evaluation
        if episode > 0 and episode % eval_freq == 0:
            # Evaluate against RandomAgent (as per user goal)
            win_rate = evaluate_agent(agent, eval_opponent, num_games=num_eval_games)
            print(f"Evaluation at Episode {episode}: Win Rate {win_rate*100:.1f}% ({num_eval_games} games)")
            
            # Dynamic adjustment
            if win_rate >= 0.9:
                print(f"Goal Reached! Win Rate >= 90%. Saving best model.")
                agent.save(f"models/best_model_90plus.pth")
                if num_eval_games < 10:
                    num_eval_games = 10
                    print("Increasing eval games to 10 for stability check.")
            elif win_rate >= 0.7:
                if num_eval_games < 10:
                    num_eval_games = 10
                    print("Good progress. Increasing eval games to 10.")
            elif win_rate >= 0.5:
                if num_eval_games < 5:
                    num_eval_games = 5
                    print("Improving. Increasing eval games to 5.")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save(f"models/best_model_{int(win_rate*100)}.pth")

        if episode % save_freq == 0:
            agent.save(f"models/dqn_chess_{episode}.pth")
            
        print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    train()
