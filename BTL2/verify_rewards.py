from chess_env import ChessEnv
import chess

def verify_rewards():
    env = ChessEnv()
    env.reset()
    
    # Setup a board state where a capture is possible
    # e.g., White Pawn at e4, Black Pawn at d5
    env.board.set_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    print("Board state:")
    print(env.board)
    
    # White captures d5 with pawn (exd5)
    move = chess.Move.from_uci("e4d5")
    action_idx = env.encode_action(move)
    
    print(f"\nExecuting move: {move}")
    _, reward, _, _ = env.step(action_idx)
    
    print(f"Reward received: {reward}")
    
    if reward == 10:
        print("SUCCESS: Pawn capture reward is correct (10).")
    else:
        print(f"FAILURE: Expected reward 10, got {reward}.")

if __name__ == "__main__":
    verify_rewards()
