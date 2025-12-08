import argparse
import chess
import torch

from random_agent import RandomAgent
from hybrid_ml_minimax_agent import MLGuidedMinimaxAgent
from policy_net import PolicyNet


def play_match(agent_white, agent_black, max_moves: int = 200) -> int:
    board = chess.Board()
    moves = 0

    while not board.is_game_over(claim_draw=True) and moves < max_moves:
        current_agent = agent_white if board.turn == chess.WHITE else agent_black
        move = current_agent.get_action(board)
        if move is None or move not in board.legal_moves:
            break
        board.push(move)
        moves += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def eval_agent_vs_random(agent_factory, n_games: int = 50, max_moves: int = 200):
    wins = draws = losses = 0
    for i in range(n_games):
        random_agent = RandomAgent()
        if i % 2 == 0:
            agent = agent_factory()
            res = play_match(agent, random_agent, max_moves)
        else:
            agent = agent_factory()
            res = play_match(random_agent, agent, max_moves)
            res *= -1  # flip perspective

        if res > 0:
            wins += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1

        if (i + 1) % max(1, n_games // 10) == 0:
            print(f"[INFO] Completed {i + 1}/{n_games} games.")

    winrate = wins / n_games
    print(f"[RESULT] Agent vs random over {n_games} games: winrate={winrate:.3f} (W/D/L={wins}/{draws}/{losses})")
    return winrate


def make_ml_guided_agent(ckpt_path: str, depth: int, top_k: int, device):
    policy = PolicyNet()
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device).eval()
    return MLGuidedMinimaxAgent(policy=policy, depth=depth, top_k=top_k, device=device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML-guided minimax agent vs random.")
    parser.add_argument("--ckpt", type=str, default="models/policy_supervised.pth", help="Policy checkpoint path.")
    parser.add_argument("--games", type=int, default=50, help="Number of evaluation games.")
    parser.add_argument("--max_moves", type=int, default=200, help="Max plies per game.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for minimax.")
    parser.add_argument("--top_k", type=int, default=6, help="Number of top policy moves to expand.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_agent_vs_random(
        lambda: make_ml_guided_agent(args.ckpt, args.depth, args.top_k, device),
        n_games=args.games,
        max_moves=args.max_moves,
    )


if __name__ == "__main__":
    main()
