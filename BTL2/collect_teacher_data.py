import argparse
import numpy as np
import chess

from chess_env import ChessEnv
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent


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
    parser = argparse.ArgumentParser(description="Collect minimax teacher data vs random.")
    parser.add_argument("--num_games", type=int, default=200, help="Number of games to generate.")
    parser.add_argument("--max_moves", type=int, default=100, help="Max plies per game.")
    parser.add_argument("--depth", type=int, default=2, help="Search depth for minimax.")
    parser.add_argument("--out", type=str, default="teacher_minimax.npz", help="Output npz path.")
    args = parser.parse_args()

    collect_teacher_data(
        num_games=args.num_games,
        max_moves=args.max_moves,
        depth=args.depth,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
