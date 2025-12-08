# Test Engine with support for Minimax and ML-guided Minimax agents
import time
import argparse
import chess
import chess.pgn
import torch

from agent import Agent
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
from ml_guided_minimax_agent import MLGuidedMinimaxAgent
from policy_net import PolicyNet


def play_match(agent_white: Agent, agent_black: Agent, max_moves: int = 300, interval: float = 0.0, verbose: bool = False, pgn: bool = False, save: bool = False):
    """Play a game between two agents and return winner (+1 white, -1 black, 0 draw) and optional PGN."""
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    moves_played = 0

    while not board.is_game_over(claim_draw=True) and moves_played < max_moves:
        current_agent = agent_white if board.turn == chess.WHITE else agent_black
        move = current_agent.get_action(board)
        if move is None or move not in board.legal_moves:
            break

        board.push(move)
        moves_played += 1
        node = node.add_variation(move)

        if verbose:
            print(board)
            print("=" * 50)
            if save:
                with open("results.txt", "a") as f:
                    f.write(f"{board}\n{'='*50}\n")

        time.sleep(interval)

    outcome = board.outcome(claim_draw=True)
    if pgn:
        pgn_text = str(game).split("\n\n", 1)[1]
    else:
        pgn_text = ""

    if outcome is None or outcome.winner is None:
        winner = 0
    elif outcome.winner == chess.WHITE:
        winner = 1
    else:
        winner = -1

    return winner, pgn_text, moves_played


def eval_agent_vs_random(agent_factory, games: int, max_moves: int, interval: float, verbose: bool, pgn: bool, save: bool):
    wins = draws = losses = 0
    for i in range(games):
        agent = agent_factory()
        random_agent = RandomAgent()

        # alternate colors
        if i % 2 == 0:
            res, pgn_text, moves = play_match(agent, random_agent, max_moves, interval, verbose, pgn, save)
        else:
            res, pgn_text, moves = play_match(random_agent, agent, max_moves, interval, verbose, pgn, save)
            res *= -1

        if res > 0:
            wins += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1

        if verbose:
            print(f"[INFO] Game {i+1}/{games} result={res} moves={moves}")
        if pgn and save:
            with open("results.txt", "a") as f:
                f.write(f"PGN Game {i+1}:\n{pgn_text}\n")

    winrate = wins / games
    print(f"[RESULT] Agent vs random over {games} games: winrate={winrate:.3f} (W/D/L={wins}/{draws}/{losses})")
    return winrate


def build_agent(agent_name: str, depth: int, top_k: int, ckpt: str, device):
    agent_name = agent_name.lower()
    if agent_name == "random":
        return RandomAgent()
    if agent_name == "minimax":
        return MinimaxAgent(depth=depth)
    if agent_name == "ml_guided":
        policy = PolicyNet()
        policy.load_state_dict(torch.load(ckpt, map_location=device))
        policy.to(device).eval()
        return MLGuidedMinimaxAgent(policy=policy, depth=depth, top_k=top_k, device=device)
    raise ValueError(f"Unknown agent type: {agent_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents (Minimax or ML-guided Minimax) against Random.")
    parser.add_argument("--agent", type=str, default="minimax", choices=["minimax", "ml_guided"], help="Agent type to test against random.")
    parser.add_argument("--games", type=int, default=50, help="Number of evaluation games.")
    parser.add_argument("--max_moves", type=int, default=300, help="Maximum plies per game.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for minimax-based agents.")
    parser.add_argument("--top_k", type=int, default=6, help="Top-k policy moves to expand (ML-guided only).")
    parser.add_argument("--ckpt", type=str, default="models/policy_supervised.pth", help="Policy checkpoint for ML-guided agent.")
    parser.add_argument("--interval", type=float, default=0.0, help="Sleep interval between moves for visualization.")
    parser.add_argument("--pgn", action="store_true", help="Emit PGN strings.")
    parser.add_argument("--verbose", action="store_true", help="Print boards during play.")
    parser.add_argument("--save", action="store_true", help="Append PGNs/boards to results.txt when used with --pgn/--verbose.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_agent_vs_random(
        lambda: build_agent(args.agent, args.depth, args.top_k, args.ckpt, device),
        games=args.games,
        max_moves=args.max_moves,
        interval=args.interval,
        verbose=args.verbose,
        pgn=args.pgn,
        save=args.save,
    )


if __name__ == "__main__":
    main()
