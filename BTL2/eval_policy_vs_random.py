import argparse
import random
import chess
import torch

from chess_env_v2 import ChessEnv
from random_agent import RandomAgent
from policy_net import PolicyNet


def policy_select_move(policy: PolicyNet, env: ChessEnv, board: chess.Board, device) -> chess.Move:
    env.board = board
    state = env.get_state()
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = policy(state_t)[0]

    legal_idxs = env.get_legal_actions()
    mask = torch.full((policy.num_actions,), -1e9, device=device)
    mask[legal_idxs] = 0.0
    masked_logits = logits + mask

    action_idx = int(torch.argmax(masked_logits).item())
    move = env.decode_action(action_idx)

    if move not in board.legal_moves:
        move = random.choice(list(board.legal_moves))
    return move


def evaluate_policy(
    checkpoint_path: str = "policy_supervised.pth",
    n_games: int = 50,
    max_moves: int = 150,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet().to(device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()

    env = ChessEnv()
    random_agent = RandomAgent()

    wins = draws = losses = 0

    for game_idx in range(n_games):
        board = chess.Board()
        policy_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        moves = 0

        while not board.is_game_over(claim_draw=True) and moves < max_moves:
            if board.turn == policy_color:
                move = policy_select_move(policy, env, board, device)
            else:
                move = random_agent.get_action(board)

            board.push(move)
            moves += 1

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
        elif outcome.winner == policy_color:
            wins += 1
        else:
            losses += 1

        if (game_idx + 1) % max(1, n_games // 10) == 0:
            print(f"[INFO] Completed {game_idx + 1}/{n_games} games.")

    winrate = wins / n_games
    print(f"[RESULT] Policy vs random over {n_games} games: winrate={winrate:.3f} (W/D/L={wins}/{draws}/{losses})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy network vs random agent.")
    parser.add_argument("--ckpt", type=str, default="policy_supervised.pth", help="Path to policy checkpoint.")
    parser.add_argument("--games", type=int, default=50, help="Number of evaluation games.")
    parser.add_argument("--max_moves", type=int, default=150, help="Max plies per game.")
    args = parser.parse_args()

    evaluate_policy(
        checkpoint_path=args.ckpt,
        n_games=args.games,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()
