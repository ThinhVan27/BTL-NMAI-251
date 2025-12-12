import argparse
import io
import numpy as np
import chess
import chess.pgn

from chess_env import ChessEnv
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent


def _parse_color(color_str: str) -> bool:
    color_str = color_str.lower()
    if color_str in ["white", "w"]:
        return chess.WHITE
    if color_str in ["black", "b"]:
        return chess.BLACK
    raise ValueError(f"Invalid color: {color_str}")


def _read_pgn_strings(path: str):
    games = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("PGN Game"):
                if current:
                    games.append(" ".join(current))
                    current = []
                continue
            if stripped:
                current.append(stripped)
        if current:
            games.append(" ".join(current))
    return games


def collect_from_pgn_file(
    pgn_path: str,
    out_path: str,
    teacher_color: str = "white",
    alternate_colors: bool = False,
):
    env = ChessEnv()
    states = []
    actions = []

    pgn_strings = _read_pgn_strings(pgn_path)
    base_color = _parse_color(teacher_color)

    for game_idx, pgn_str in enumerate(pgn_strings):
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if game is None:
            continue

        board = game.board()
        # allow optional alternating colors per game (e.g., if dataset alternates teacher side)
        teacher_col = base_color if not alternate_colors else (chess.WHITE if game_idx % 2 == 0 else chess.BLACK)

        for move in game.mainline_moves():
            if board.turn == teacher_col:
                env.board = board
                state = env.get_state()
                action_idx = env.encode_action(move)
                states.append(state)
                actions.append(action_idx)
            board.push(move)

        if (game_idx + 1) % max(1, len(pgn_strings) // 10 or 1) == 0:
            print(f"[INFO] Parsed game {game_idx + 1}/{len(pgn_strings)}, dataset size={len(states)}")

    if not states:
        print("[WARN] No samples collected from PGN file.")
        return

    states_array = np.stack(states, axis=0).astype(np.float32)
    actions_array = np.array(actions, dtype=np.int64)
    np.savez(out_path, states=states_array, actions=actions_array)
    print(f"[INFO] Saved dataset to {out_path} with {len(states_array)} samples (PGN source).")


def collect_teacher_data(
    num_games: int = 200,
    max_moves: int = 100,
    depth: int = 2,
    out_path: str = "teacher_minimax.npz",
):
    env = ChessEnv()
    teacher = MinimaxAgent(depth=depth)
    random_agent = RandomAgent()

    states = []
    actions = []

    for game_idx in range(num_games):
        board = chess.Board()
        teacher_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        move_counter = 0

        while not board.is_game_over(claim_draw=True) and move_counter < max_moves:
            if board.turn == teacher_color:
                env.board = board
                state = env.get_state()
                move = teacher.get_best_move(board)
                if move is None:
                    break
                action_idx = env.encode_action(move)
                states.append(state)
                actions.append(action_idx)
            else:
                move = random_agent.get_action(board)

            board.push(move)
            move_counter += 1

        if (game_idx + 1) % max(1, num_games // 10) == 0:
            print(f"[INFO] Finished game {game_idx + 1}/{num_games}, dataset size={len(states)}")

    states_array = np.stack(states, axis=0).astype(np.float32)
    actions_array = np.array(actions, dtype=np.int64)
    np.savez(out_path, states=states_array, actions=actions_array)
    print(f"[INFO] Saved dataset to {out_path} with {len(states_array)} samples.")


def main():
    parser = argparse.ArgumentParser(description="Collect minimax teacher data.")
    parser.add_argument("--num_games", type=int, default=200, help="Number of games to generate (simulation mode).")
    parser.add_argument("--max_moves", type=int, default=100, help="Max plies per simulated game.")
    parser.add_argument("--depth", type=int, default=2, help="Search depth for minimax (simulation mode).")
    parser.add_argument("--out", type=str, default="teacher_minimax.npz", help="Output npz path.")
    parser.add_argument("--pgn_file", type=str, default=None, help="Optional PGN file to parse instead of simulating games.")
    parser.add_argument("--teacher_color", type=str, default="white", help="Teacher color in PGN file (white/black).")
    parser.add_argument("--alternate_colors", action="store_true", help="Alternate teacher color per game when using PGN input.")
    args = parser.parse_args()

    if args.pgn_file:
        collect_from_pgn_file(
            pgn_path=args.pgn_file,
            out_path=args.out,
            teacher_color=args.teacher_color,
            alternate_colors=args.alternate_colors,
        )
    else:
        collect_teacher_data(
            num_games=args.num_games,
            max_moves=args.max_moves,
            depth=args.depth,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
