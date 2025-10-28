"""High-level Sokoban skeleton with game rules, solver stubs, and visualization hooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from abc import ABC, abstractmethod
import time
import tracemalloc
import pyautogui
import numpy as np
import json
import os
import shutil
import sys
import math
from tqdm import tqdm


GAME_ROOT = Path(r"D:\1. REFERENCES\1. AI - Machine Learning\10. GenAI\GAs\BTL-NMAI-251\test\game")
SOLUTION_ROOT = Path(r"D:\1. REFERENCES\1. AI - Machine Learning\10. GenAI\GAs\BTL-NMAI-251\test\solution")

Coord = Tuple[int, int]
Move = str
MOVE_VECTORS: Dict[Move, Tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

# ===== OBJECT TO ASCII =====
# PLAYER :          @
# BOX:              $
# GOAL:             .
# WALL:             #
# BOX_AT_GOAL:      *
# PLAYER_AT_GOAL:   +


class LevelFormatError(ValueError):
    """Raised when an ASCII level description cannot be parsed."""


@dataclass(frozen=True)
class SokobanBoard:
    width: int
    height: int
    walls: frozenset[Coord]
    goals: frozenset[Coord]
    initial_boxes: frozenset[Coord]
    initial_player: Coord

    @classmethod
    def from_ascii(cls, ascii_map: Sequence[str]) -> "SokobanBoard":
        if isinstance(ascii_map, str):
            raw_lines = ascii_map.splitlines()
        else:
            raw_lines = list(ascii_map)
        lines = [line.rstrip("\n") for line in raw_lines if line.strip()]
        if not lines:
            raise LevelFormatError("Level description is empty")
        height = len(lines)
        width = max(len(row) for row in lines)
        walls: set[Coord] = set()
        goals: set[Coord] = set()
        boxes: set[Coord] = set()
        player: Optional[Coord] = None
        for y, row in enumerate(lines):
            padded = row.ljust(width)
            for x, char in enumerate(padded):
                if char == "#":
                    walls.add((x, y))
                elif char == ".":
                    goals.add((x, y))
                elif char == "$":
                    boxes.add((x, y))
                elif char == "@":
                    player = (x, y)
                elif char == "*":
                    boxes.add((x, y))
                    goals.add((x, y))
                elif char == "+":
                    goals.add((x, y))
                    player = (x, y)
        if player is None:
            raise LevelFormatError("Player '@' or '+' is required")
        if not goals:
            raise LevelFormatError("At least one goal '.' is required")
        if not boxes:
            raise LevelFormatError("At least one box '$' or '*' is required")
        if len(boxes) > len(goals):
            raise LevelFormatError("Not enough goal tiles for all boxes")
        return cls(
            width=width,
            height=height,
            walls=frozenset(walls),
            goals=frozenset(goals),
            initial_boxes=frozenset(boxes),
            initial_player=player,
        )

    def is_wall(self, coord: Coord) -> bool:
        return coord in self.walls

    def is_goal(self, coord: Coord) -> bool:
        return coord in self.goals

    def in_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height

    def render(self, state: Optional["SokobanState"] = None) -> str:
        active_state = state or SokobanState(self.initial_player, self.initial_boxes)
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.walls:
            grid[y][x] = "#"
        for x, y in self.goals:
            grid[y][x] = "."
        for x, y in active_state.boxes:
            grid[y][x] = "*" if (x, y) in self.goals else "$"
        px, py = active_state.player
        grid[py][px] = "+" if (px, py) in self.goals else "@"
        return "\n".join("".join(row) for row in grid)


@dataclass(frozen=True)
class SokobanState:
    player: Coord
    boxes: frozenset[Coord]

    def move_box(self, source: Coord, target: Coord) -> "SokobanState":
        boxes = set(self.boxes)
        boxes.remove(source)
        boxes.add(target)
        return SokobanState(player=target, boxes=frozenset(boxes))

    def move_player(self, target: Coord) -> "SokobanState":
        return SokobanState(player=target, boxes=self.boxes)


class SokobanGame:
    def __init__(self, board: SokobanBoard):
        self._board = board
        self.initial_state = SokobanState(board.initial_player, board.initial_boxes)

    @property
    def board(self) -> SokobanBoard:
        return self._board

    def is_goal(self, state: SokobanState) -> bool:
        return all(box in self._board.goals for box in state.boxes)
    
    def apply_move(self, state: SokobanState, move: Move) -> Optional[SokobanState]:
        if move not in MOVE_VECTORS:
            raise ValueError(f"Unknown move: {move}")
        dx, dy = MOVE_VECTORS[move]
        px, py = state.player
        nx, ny = px + dx, py + dy
        next_coord = (nx, ny)
        if self._board.is_wall(next_coord):
            return None
        boxes = set(state.boxes)
        if next_coord in boxes:
            bx, by = nx + dx, ny + dy
            box_target = (bx, by)
            if self._board.is_wall(box_target) or box_target in boxes:
                return None
            boxes.remove(next_coord)
            boxes.add(box_target)
            return SokobanState(player=next_coord, boxes=frozenset(boxes))
        return SokobanState(player=next_coord, boxes=state.boxes)

    def legal_moves(self, state: SokobanState) -> List[Move]:
        return [move for move in MOVE_VECTORS if self.apply_move(state, move) is not None]

    def successors(self, state: SokobanState) -> Iterator[tuple[Move, SokobanState]]:
        for move in MOVE_VECTORS:
            next_state = self.apply_move(state, move)
            if next_state is not None:
                yield move, next_state

    def render(self, state: Optional[SokobanState] = None) -> str:
        return self._board.render(state or self.initial_state)

    def replay(self, moves: Iterable[Move]) -> List[SokobanState]:
        history = [self.initial_state]
        current = self.initial_state
        for move in moves:
            next_state = self.apply_move(current, move)
            if next_state is None:
                raise ValueError(f"Illegal move '{move}' encountered during replay")
            history.append(next_state)
            current = next_state
        return history


class ComplexityTracker:
    def __init__(self) -> None:
        self.nodes_expanded = 0
        self.max_frontier = 0
        self._start_time = 0.0
        self.peak_memory = 0

    def start(self) -> None:
        self._start_time = time.perf_counter()
        tracemalloc.start()

    def stop(self) -> float:
        elapsed = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory = peak
        return elapsed

    def record_node(self) -> None:
        self.nodes_expanded += 1

    def track_frontier(self, size: int) -> None:
        if size > self.max_frontier:
            self.max_frontier = size


@dataclass
class SolverLimits:
    time_limit_s: Optional[float] = None
    node_limit: Optional[int] = None


@dataclass
class SolverResult:
    solved: bool
    move_sequence: List[Move]
    elapsed_time_s: float
    nodes_expanded: int
    max_frontier: int
    peak_memory_bytes: int


class Solver(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.tracker = ComplexityTracker()

    def solve(self, game: SokobanGame, limits: Optional[SolverLimits] = None) -> SolverResult:
        limits = limits or SolverLimits()
        self.tracker = ComplexityTracker()
        self.tracker.start()
        try:
            solved, moves = self._search(game, limits)
        finally:
            elapsed = self.tracker.stop()
        return SolverResult(
            solved=solved,
            move_sequence=moves,
            elapsed_time_s=elapsed,
            nodes_expanded=self.tracker.nodes_expanded,
            max_frontier=self.tracker.max_frontier,
            peak_memory_bytes=self.tracker.peak_memory,
        )

    @abstractmethod
    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        raise NotImplementedError

class BFSSolver(Solver):
    name = "bfs"

    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        raise NotImplementedError("BFS solver not implemented yet")


class DFSSolver(Solver):
    name = "dfs"

    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        raise NotImplementedError("DFS solver not implemented yet")


class AStarSolver(Solver):
    name = "astar"

    def __init__(self, heuristic: Optional["Heuristic"] = None) -> None:
        super().__init__()
        self.heuristic = heuristic
        
class Genetic(Solver):
    name = "genetic"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(Genetic.NpEncoder, self).default(obj)

    class Chromosome:
        def __init__(self, rng=None, max_moves=1500):
            self._random = rng if rng is not None else np.random.default_rng()
            self._sequence: List[Move] = []
            self._max_moves = max_moves
            self._fitness: float = -float("inf")
            self._quality: float = -float("inf")
            self._diversity: float = 0.0
            self._controlability: float = 0.0

        # ------------------------------
        # Utility / domain-aware ops
        # ------------------------------
        def _find_path_to_pos(self, game: SokobanGame, state: SokobanState, target_pos: Coord) -> List[Move]:
            # BFS for player-only path avoiding boxes & walls
            from collections import deque
            queue = deque([(state.player, [])])
            visited = {state.player}
            while queue:
                pos, path = queue.popleft()
                if pos == target_pos:
                    return path
                for mv, (dx, dy) in MOVE_VECTORS.items():
                    nxt = (pos[0] + dx, pos[1] + dy)
                    if (game.board.in_bounds(nxt) and not game.board.is_wall(nxt) and nxt not in state.boxes and nxt not in visited):
                        visited.add(nxt)
                        queue.append((nxt, path + [mv]))
            return []

        def is_deadlock(self, game: SokobanGame, state: SokobanState) -> bool:
            # corner/corridor/frozen checks (lightweight)
            for box in state.boxes:
                if box in game.board.goals:
                    continue
                x, y = box
                walls = game.board.walls
                # corner
                if ((x-1, y) in walls and (x, y-1) in walls) or ((x+1, y) in walls and (x, y-1) in walls) or \
                   ((x-1, y) in walls and (x, y+1) in walls) or ((x+1, y) in walls and (x, y+1) in walls):
                    return True
                # frozen (no push directions)
                pushable = 0
                for mv in MOVE_VECTORS.values():
                    from_pos = (x - mv[0], y - mv[1])
                    to_pos = (x + mv[0], y + mv[1])
                    if (game.board.in_bounds(from_pos) and from_pos not in state.boxes and not game.board.is_wall(from_pos) and
                        game.board.in_bounds(to_pos) and to_pos not in state.boxes and not game.board.is_wall(to_pos)):
                        pushable += 1
                if pushable == 0:
                    return True
            return False

        def _generate_macro_action(self, game: SokobanGame, state: SokobanState) -> List[Move]:
            # Find a box not on goal that is nearest (manhattan) to any goal, then attempt a short push sequence.
            boxes = [b for b in state.boxes if b not in game.board.goals]
            if not boxes:
                return []
            goals = list(game.board.goals)
            # choose target box by min distance to nearest goal
            def box_goal_dist(b):
                return min(abs(b[0]-g[0]) + abs(b[1]-g[1]) for g in goals)
            boxes.sort(key=box_goal_dist)
            target_box = boxes[0]
            # choose goal nearest to this box
            target_goal = min(goals, key=lambda g: abs(g[0]-target_box[0]) + abs(g[1]-target_box[1]))
            dx = target_goal[0] - target_box[0]
            dy = target_goal[1] - target_box[1]
            # prefer cardinal axis with larger distance
            push_dir = "right" if dx > 0 else "left" if dx < 0 else "down" if dy > 0 else "up"
            push_vector = MOVE_VECTORS[push_dir]
            # required player pos to push once
            req_pos = (target_box[0] - push_vector[0], target_box[1] - push_vector[1])
            path = self._find_path_to_pos(game, state, req_pos)
            if not path:
                return []
            moves = path + [push_dir]
            # attempt up to 2-3 pushes in same direction if possible
            # simulate locally (best-effort)
            current_state = state
            for mv in moves:
                ns = game.apply_move(current_state, mv)
                if ns is None:
                    break
                current_state = ns
            # attempt additional pushes (cautious)
            for _ in range(2):
                # find new req pos for next push in same dir
                # find box position updated (the one we targeted may move)
                # simple approach: reuse last push_dir and append if possible
                next_req = (current_state.player[0] , current_state.player[1])  # player is at push spot after applying push
                # check if a push in push_dir is legal
                nxt = game.apply_move(current_state, push_dir)
                if nxt is None:
                    break
                moves.append(push_dir)
                current_state = nxt
            return moves[:20]

        # ------------------------------
        # Initialization variants
        # ------------------------------
        def random(self, game: SokobanGame):
            rv = self._random.random()
            if rv < 0.35:
                L = int(self._random.integers(10, 50))
            elif rv < 0.75:
                L = int(self._random.integers(50, 150))
            else:
                L = int(self._random.integers(150, min(300, self._max_moves)))
            self._sequence = [self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(L)]

        def random_with_macro_actions(self, game: SokobanGame):
            self._sequence = []
            state = game.initial_state
            attempts = 0
            while len(self._sequence) < self._max_moves and attempts < 300:
                attempts += 1
                if self._random.random() < 0.75:
                    macro = self._generate_macro_action(game, state)
                    if macro:
                        for mv in macro:
                            if len(self._sequence) >= self._max_moves:
                                break
                            self._sequence.append(mv)
                            ns = game.apply_move(state, mv)
                            if ns is None:
                                break
                            state = ns
                        # quick deadlock check - revert last few if deadlock
                        if self.is_deadlock(game, state):
                            revert = min(6, len(self._sequence))
                            self._sequence = self._sequence[:-revert]
                            state = game.initial_state
                            try:
                                history = game.replay(self._sequence)
                                state = history[-1]
                            except Exception:
                                state = game.initial_state
                                self._sequence = []
                    else:
                        # fallback small random walk
                        n = int(self._random.integers(1, 6))
                        for _ in range(n):
                            if len(self._sequence) >= self._max_moves:
                                break
                            mv = self._random.choice(list(MOVE_VECTORS.keys()))
                            self._sequence.append(mv)
                            ns = game.apply_move(state, mv)
                            if ns is None:
                                self._sequence.pop()
                                break
                            state = ns
                else:
                    # exploratory random chunk
                    n = int(self._random.integers(3, 12))
                    for _ in range(n):
                        if len(self._sequence) >= self._max_moves:
                            break
                        mv = self._random.choice(list(MOVE_VECTORS.keys()))
                        self._sequence.append(mv)
                        ns = game.apply_move(state, mv)
                        if ns is None:
                            self._sequence.pop()
                            break
                        state = ns
            # pad if too short
            if len(self._sequence) < 8:
                self._sequence.extend([self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(8 - len(self._sequence))])

        # ------------------------------
        # Recombination / mutation
        # ------------------------------
        def crossover(self, other: "Genetic.Chromosome") -> "Genetic.Chromosome":
            child = Genetic.Chromosome(self._random, self._max_moves)
            a, b = self._sequence, other._sequence
            if not a:
                child._sequence = list(b)
                return child
            if not b:
                child._sequence = list(a)
                return child
            minlen = min(len(a), len(b))
            if minlen > 2:
                p1 = int(self._random.integers(1, minlen))
                p2 = int(self._random.integers(p1, minlen))
                child._sequence = a[:p1] + b[p1:p2] + (a[p2:] if p2 < len(a) else [])
            else:
                split = int(self._random.integers(0, minlen+1))
                child._sequence = a[:split] + b[split:]
            # trim
            if len(child._sequence) > child._max_moves:
                child._sequence = child._sequence[:child._max_moves]
            return child

        def mutation(self, game: SokobanGame, mut_rate: float) -> "Genetic.Chromosome":
            child = Genetic.Chromosome(self._random, self._max_moves)
            child._sequence = list(self._sequence)

            p = self._random.random()
            # big disruptive
            if p < 0.12 and len(child._sequence) > 2:
                a = int(self._random.integers(0, len(child._sequence)))
                b = int(self._random.integers(a, min(len(child._sequence), a + max(2, len(child._sequence)//3))))
                new_len = int(self._random.integers(1, max(2, (b-a))))
                new_seg = [self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(new_len)]
                child._sequence[a:b] = new_seg
            # swap
            elif p < 0.32 and len(child._sequence) >= 2:
                i = int(self._random.integers(0, len(child._sequence)))
                j = int(self._random.integers(0, len(child._sequence)))
                child._sequence[i], child._sequence[j] = child._sequence[j], child._sequence[i]
            # inversion
            elif p < 0.52 and len(child._sequence) >= 3:
                i = int(self._random.integers(0, len(child._sequence)-1))
                j = int(self._random.integers(i+1, len(child._sequence)))
                child._sequence[i:j] = list(reversed(child._sequence[i:j]))
            # gene-wise
            else:
                for i in range(len(child._sequence)):
                    if self._random.random() < mut_rate:
                        child._sequence[i] = self._random.choice(list(MOVE_VECTORS.keys()))

            # insert guided macro with some prob
            if self._random.random() < mut_rate * 0.9 and len(child._sequence) < child._max_moves:
                insert_pos = int(self._random.integers(0, len(child._sequence)+1)) if len(child._sequence) > 0 else 0
                # try to infer state at insert_pos
                state = game.initial_state
                for mv in child._sequence[:insert_pos]:
                    ns = game.apply_move(state, mv)
                    if ns is None:
                        state = game.initial_state
                        break
                    state = ns
                macro = self._generate_macro_action(game, state)
                if macro:
                    seg = macro[:min(len(macro), max(1, int(self._random.integers(1, 4))))]
                    child._sequence[insert_pos:insert_pos] = seg
                else:
                    # fallback random insert
                    num_ins = int(self._random.integers(1, min(3, child._max_moves - len(child._sequence))))
                    for _ in range(num_ins):
                        pos = int(self._random.integers(0, len(child._sequence)+1))
                        child._sequence.insert(pos, self._random.choice(list(MOVE_VECTORS.keys())))

            # small deletions sometimes
            if len(child._sequence) > 2 and self._random.random() < mut_rate * 0.6:
                del_len = int(self._random.integers(1, min(4, len(child._sequence))))
                start = int(self._random.integers(0, len(child._sequence)-del_len+1))
                for _ in range(del_len):
                    if start < len(child._sequence):
                        child._sequence.pop(start)

            # guided repair: if invalid sequence, truncate at illegal move and try macro fill
            try:
                history = game.replay(child._sequence)
                final_state = history[-1]
                if child.is_deadlock(game, final_state):
                    # rollback a few moves
                    rollback = 0
                    while rollback < 6 and child.is_deadlock(game, final_state):
                        if not child._sequence:
                            break
                        child._sequence = child._sequence[:-1]
                        rollback += 1
                        try:
                            history = game.replay(child._sequence)
                            final_state = history[-1]
                        except Exception:
                            continue
            except ValueError:
                # truncate at first illegal move
                seq = []
                state = game.initial_state
                for mv in child._sequence:
                    ns = game.apply_move(state, mv)
                    if ns is None:
                        break
                    seq.append(mv)
                    state = ns
                # try add a macro
                macro = self._generate_macro_action(game, state)
                if macro:
                    seq.extend(macro[:min(len(macro), 5)])
                child._sequence = seq

            # trim to max
            if len(child._sequence) > child._max_moves:
                child._sequence = child._sequence[:child._max_moves]

            return child

        # ------------------------------
        # Fitness eval / IO
        # ------------------------------
        def evaluate(self, game: SokobanGame, heuristic: "Heuristic", initial_total_dist: Optional[float] = None):
            # compute fitness with smoother shaping and progress signal.
            try:
                from scipy.optimize import linear_sum_assignment
                import numpy as _np
                use_hungarian = True
            except Exception:
                use_hungarian = False
            try:
                history = game.replay(self._sequence)
                final_state = history[-1]
                valid_moves = len(history) - 1
                valid_ratio = valid_moves / (len(self._sequence) if self._sequence else 1)
                self._controlability = valid_ratio * 100.0

                # deadlock lighter penalty
                deadlock_pen = -20000.0 if self.is_deadlock(game, final_state) else 0.0

                if game.is_goal(final_state):
                    self._fitness = 5_000_000.0 - len(self._sequence) * 50.0 + self._controlability * 50.0
                    self._quality = self._fitness
                    return self._fitness

                boxes_on_goals = sum(1 for b in final_state.boxes if b in game.board.goals)
                box_goal_score = boxes_on_goals * 20000.0

                # distance via assignment or greedy
                boxes_list = list(final_state.boxes)
                goals_list = list(game.board.goals)
                total_box_dist = 0.0
                if boxes_list and goals_list:
                    if use_hungarian:
                        cost = _np.zeros((len(boxes_list), len(goals_list)))
                        for i, b in enumerate(boxes_list):
                            for j, g in enumerate(goals_list):
                                cost[i, j] = abs(b[0] - g[0]) + abs(b[1] - g[1])
                        r, c = linear_sum_assignment(cost)
                        total_box_dist = float(cost[r, c].sum())
                    else:
                        # greedy fallback
                        total_box_dist = 0.0
                        remaining_goals = set(goals_list)
                        for b in boxes_list:
                            d, bestg = min(((abs(b[0]-g[0]) + abs(b[1]-g[1])), g) for g in remaining_goals)
                            total_box_dist += d
                            remaining_goals.discard(bestg)

                # progress vs initial
                progress_score = 0.0
                if initial_total_dist is not None:
                    progress_score = (initial_total_dist - total_box_dist) * 250.0

                push_count = sum(1 for i in range(len(history)-1) if history[i].boxes != history[i+1].boxes)
                push_score = push_count * 250.0

                length_penalty = -len(self._sequence) * 2.0

                # heuristic estimate
                try:
                    h_est = heuristic.estimate(final_state, game)
                    heuristic_score = -h_est * 50.0
                except Exception:
                    heuristic_score = 0.0

                self._fitness = box_goal_score + progress_score + push_score + heuristic_score + deadlock_pen + length_penalty

            except ValueError:
                # invalid sequence: reward prefix progress rather than full penalty
                prefix_len = 0
                state = game.initial_state
                for mv in self._sequence:
                    ns = game.apply_move(state, mv)
                    if ns is None:
                        break
                    prefix_len += 1
                    state = ns
                self._fitness = -20000.0 + prefix_len * 100.0
                self._controlability = (prefix_len / (len(self._sequence) if self._sequence else 1)) * 100.0

            self._quality = self._fitness
            return self._fitness

        def quality(self):
            return self._quality if self._quality is not None else -float("inf")

        def save(self, filepath):
            savedObj = {
                "sequence": self._sequence,
                "fitness": self._fitness,
                "quality": self._quality,
                "diversity": self._diversity,
                "controlability": self._controlability
            }
            with open(filepath, "w") as f:
                f.write(json.dumps(savedObj, cls=Genetic.NpEncoder))

        def load(self, filepath):
            with open(filepath, "r") as f:
                saved = json.loads("".join(f.readlines()))
            self._sequence = saved.get("sequence", [])
            self._fitness = saved.get("fitness", None)
            self._quality = saved.get("quality", None)
            self._diversity = saved.get("diversity", 0.0)
            self._controlability = saved.get("controlability", 0.0)

    # ------------------------------
    # Genetic outer methods
    # ------------------------------
    def __init__(self, heuristic: Optional["Heuristic"] = None, pop_size: int = 150, generations: int = 1000, max_moves: int = 1500, parallel_eval: bool = True):
        super().__init__()
        self.heuristic = heuristic or ManhattanHeuristic()
        self.pop_size = pop_size
        self.generations = generations
        self._random = np.random.default_rng()
        self._chromosomes: List[Genetic.Chromosome] = []
        self._fitness_fn = self.fitness_quality
        self._max_moves = max_moves
        self._mut_rate = 0.9
        self._cross_rate = 0.9
        self._tournament = 3
        self._elitism = max(1, math.ceil(self.pop_size * 0.05))
        self._parallel_eval = parallel_eval
        self.best_ever_chrom: Optional[Genetic.Chromosome] = None
        self.best_ever_fitness: float = -float("inf")
        self.elite_archive: List[Genetic.Chromosome] = []

    def _update_best_ever(self):
        best = self._best()
        if best and best._fitness > self.best_ever_fitness:
            self.best_ever_fitness = best._fitness
            # store a deep copy to avoid later mutation side-effects
            import copy
            self.best_ever_chrom = copy.deepcopy(best)

    def _bfs_seeding(self, game: SokobanGame, max_depth: int = 10, max_seeds: int = 50) -> List[List[Move]]:
        from collections import deque
        queue = deque([(game.initial_state, [])])
        visited = {game.initial_state}
        seeds = []
        while queue and len(seeds) < max_seeds:
            state, moves = queue.popleft()
            if len(moves) >= max_depth:
                seeds.append(moves)
                continue
            for mv, ns in game.successors(state):
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, moves + [mv]))
        return seeds

    def _reset(self, game: SokobanGame, **kwargs):
        # adaptive config
        self._tournament = kwargs.get("tournment_size", 3)
        self._cross_rate = kwargs.get("cross_rate", self._cross_rate)
        self._mut_rate = kwargs.get("mut_rate", self._mut_rate)
        self._elitism = max(1, math.ceil(self.pop_size * kwargs.get("elitism_perct", 0.05)))

        self._chromosomes = []
        # BFS seed ~30%
        try:
            bfs = self._bfs_seeding(game, max_depth=12, max_seeds= int(self.pop_size * 0.5))
            n_bfs = min(len(bfs), max(1, int(self.pop_size * 0.3)))
            for i in range(n_bfs):
                c = Genetic.Chromosome(self._random, self._max_moves)
                c._sequence = list(bfs[i])
                self._chromosomes.append(c)
        except Exception:
            pass

        # fill rest with macro-based random
        while len(self._chromosomes) < self.pop_size:
            c = Genetic.Chromosome(self._random, self._max_moves)
            c.random_with_macro_actions(game)
            self._chromosomes.append(c)

        # If we have elite_archive from previous restart, inject them (replace worst)
        if self.elite_archive:
            k = min(len(self.elite_archive), self.pop_size // 5)
            # place elites at front
            self._chromosomes = list(self.elite_archive[:k]) + self._chromosomes[:-k]
            # clear archive after injection
            self.elite_archive = []

        # compute initial assignment distance for progress metric
        try:
            from scipy.optimize import linear_sum_assignment
            import numpy as _np
            boxes_list = list(game.board.initial_boxes)
            goals_list = list(game.board.goals)
            if boxes_list and goals_list:
                cost = _np.zeros((len(boxes_list), len(goals_list)))
                for i, b in enumerate(boxes_list):
                    for j, g in enumerate(goals_list):
                        cost[i, j] = abs(b[0]-g[0]) + abs(b[1]-g[1])
                r, c = linear_sum_assignment(cost)
                self._initial_total_dist = float(cost[r, c].sum())
            else:
                self._initial_total_dist = None
        except Exception:
            self._initial_total_dist = None

        # evaluate initial pop
        self._evaluate(game)

    def _select(self) -> "Genetic.Chromosome":
        # tournament selection
        size = min(self._tournament, self.pop_size)
        indices = list(range(self.pop_size))
        self._random.shuffle(indices)
        contenders = [self._chromosomes[indices[i]] for i in range(size)]
        contenders.sort(key=lambda c: self._fitness_fn(c), reverse=True)
        return contenders[0]

    def _evaluate(self, game: SokobanGame):
        if self._parallel_eval:
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=min(8, self.pop_size)) as ex:
                    futures = {ex.submit(chrom.evaluate, game, self.heuristic, getattr(self, "_initial_total_dist", None)): chrom for chrom in self._chromosomes}
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception:
                            pass
            except Exception:
                for chrom in self._chromosomes:
                    chrom.evaluate(game, self.heuristic, getattr(self, "_initial_total_dist", None))
        else:
            for chrom in self._chromosomes:
                chrom.evaluate(game, self.heuristic, getattr(self, "_initial_total_dist", None))

        # sort population by chosen fitness function
        self._chromosomes.sort(key=lambda c: self._fitness_fn(c), reverse=True)

    def _local_search(self, chromosome: "Genetic.Chromosome", game: SokobanGame, max_iterations: int = 6) -> "Genetic.Chromosome":
        # simple remove-one move local search (fast)
        best = chromosome
        best_f = best._fitness
        for _ in range(max_iterations):
            improved = False
            if len(best._sequence) <= 1:
                break
            for i in range(len(best._sequence)):
                cand = Genetic.Chromosome(self._random, best._max_moves)
                cand._sequence = best._sequence[:i] + best._sequence[i+1:]
                cand.evaluate(game, self.heuristic, getattr(self, "_initial_total_dist", None))
                if cand._fitness > best_f:
                    best = cand
                    best_f = cand._fitness
                    improved = True
                    break
            if not improved:
                break
        return best

    def _intensive_repair_best(self, game: SokobanGame, rounds: int = 10) -> bool:
        best = self._best()
        if not best:
            return False
        improved_any = False
        for _ in range(rounds):
            insert_pos = int(self._random.integers(0, len(best._sequence)+1)) if best._sequence else 0
            # attempt to compute state at insert_pos
            state = game.initial_state
            for mv in best._sequence[:insert_pos]:
                ns = game.apply_move(state, mv)
                if ns is None:
                    state = game.initial_state
                    break
                state = ns
            macro = best._generate_macro_action(game, state)
            if not macro:
                continue
            cand = Genetic.Chromosome(self._random, best._max_moves)
            cand._sequence = best._sequence[:insert_pos] + macro + best._sequence[insert_pos:]
            # small mutation & evaluate
            cand = cand.mutation(game, max(0.05, self._mut_rate * 0.6))
            cand.evaluate(game, self.heuristic, getattr(self, "_initial_total_dist", None))
            if cand._fitness > best._fitness:
                # replace top candidate and re-evaluate population
                self._chromosomes[0] = cand
                self._evaluate(game)
                improved_any = True
                break
        return improved_any

    def _update(self, game: SokobanGame):
        # retain elites
        elites = [c for c in self._chromosomes[:self._elitism]]
        new_pop = elites.copy()

        # immigrants to boost diversity
        num_immigrants = max(1, int(self.pop_size * 0.12))
        for _ in range(num_immigrants):
            im = Genetic.Chromosome(self._random, self._max_moves)
            im.random_with_macro_actions(game)
            new_pop.append(im)

        # adaptive mutation: if stuck indicated externally, it may be increased (handled in _search)
        # generate rest by selection/crossover/mutation
        while len(new_pop) < self.pop_size:
            parent = self._select()
            child = parent
            if self._random.random() < self._cross_rate:
                mate = self._select()
                child = parent.crossover(mate)
            child = child.mutation(game, self._mut_rate)
            new_pop.append(child)

        self._chromosomes = new_pop[:self.pop_size]
        # evaluate and local-search-top
        self._evaluate(game)
        top_k = max(1, int(self.pop_size * 0.05))
        for i in range(min(top_k, len(self._chromosomes))):
            self._chromosomes[i] = self._local_search(self._chromosomes[i], game, max_iterations=3)
        self._chromosomes.sort(key=lambda c: self._fitness_fn(c), reverse=True)
        self._update_best_ever()

    def _best(self) -> Optional["Genetic.Chromosome"]:
        return self._chromosomes[0] if self._chromosomes else None

    # ------------------------------
    # public fitness wrappers
    # ------------------------------
    @staticmethod
    def fitness_quality(chromosome: "Genetic.Chromosome"):
        return chromosome.quality()

    @staticmethod
    def fitness_quality_control(chromosome: "Genetic.Chromosome"):
        result = chromosome.quality()
        if result >= 1:
            result += chromosome.controlability()
        return result / 2.0

    @staticmethod
    def fitness_quality_control_diversity(chromosome: "Genetic.Chromosome"):
        result = chromosome.quality()
        if result >= 1:
            result += chromosome.controlability()
        if result >= 1 and chromosome.controlability() >= 1:
            result += chromosome.diversity()
        return result / 3.0

    # ------------------------------
    # main search
    # ------------------------------
    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        # initialize
        self._reset(game)
        self.best_ever_chrom = None
        self.best_ever_fitness = -float("inf")
        self.elite_archive = []

        # dynamic threshold: moderate estimate based on boxes
        num_boxes = len(game.board.initial_boxes)
        threshold = num_boxes * 100000.0 + 5_000_000.0
        # stagnation detection: 50-generation window
        best_hist: List[float] = []
        stagnation_count = 0
        # external stagnation flag for adaptive mut rate
        self._stagnation_count = 0

        with tqdm(total=self.generations, desc="GA Progress", unit="gen") as pbar:
            for gen in range(self.generations):
                # adaptive mutation decay/increase
                if self._stagnation_count > 2:
                    self._mut_rate = min(0.7, self._mut_rate * 1.15)
                else:
                    self._mut_rate = max(0.05, self._mut_rate * 0.995)

                self._update(game)
                best_chrom = self._best()
                self._stagnation_count = stagnation_count

                # progress bar postfix
                pbar.set_postfix({
                    "fit": f"{best_chrom._fitness:.0f}",
                    "mv": len(best_chrom._sequence),
                    "stg": stagnation_count,
                    "mut": f"{self._mut_rate:.3f}"
                })
                pbar.update(1)

                if gen % 10 == 0:
                    top5 = ", ".join(f"{c._fitness:.0f}" for c in self._chromosomes[:5])
                    tqdm.write(f"[Gen {gen}] top5: {top5}")

                # stagnation logic using 50-gen windows (as requested)
                best_hist.append(best_chrom._fitness)
                if len(best_hist) > 50:
                    recent_best = max(best_hist[-50:])
                    old_best = max(best_hist[-100:-50]) if len(best_hist) >= 100 else 0.0
                    if recent_best - old_best < 50.0:  # threshold 50 (you requested 50)
                        stagnation_count += 1
                    else:
                        stagnation_count = 0

                    # when stuck often, try intensive repair before reset
                    if stagnation_count >= 3:
                        tqdm.write(f"[STUCK] gen {gen}: trying intensive repair of best")
                        repaired = self._intensive_repair_best(game, rounds=25)
                        if not repaired:
                            tqdm.write(f"[RESTART] Stagnation at gen {gen}. Resetting population and keeping elites.")
                            # keep top 10% as archive
                            top_k = max(1, int(0.1 * self.pop_size))
                            self.elite_archive = [c for c in self._chromosomes[:top_k]]
                            self._reset(game)
                            # small increase in mutation to escape next time
                            self._mut_rate = min(0.7, self._mut_rate * 1.2)
                        stagnation_count = 0
                        best_hist = []

                self.tracker.record_node()
                self.tracker.track_frontier(len(self._chromosomes))

                # success check
                if best_chrom._fitness >= threshold:
                    tqdm.write(f"[SUCCESS] Found at gen {gen} (fitness {best_chrom._fitness:.0f})")
                    return True, best_chrom._sequence

        # end generations
        best_chrom = self._best()
        solved = best_chrom and best_chrom._fitness >= threshold
        if not solved and self.best_ever_chrom:
            tqdm.write("[INFO] No exact solution found; returning best-ever chromosome (approximate).")
            return True, self.best_ever_chrom._sequence
        return solved, (best_chrom._sequence if best_chrom else [])
    

class Heuristic(ABC):
    @abstractmethod
    def estimate(self, state: SokobanState, game: SokobanGame) -> float:
        raise NotImplementedError


class ManhattanHeuristic(Heuristic):
    
    def _mahattan_dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def estimate(self, state: SokobanState, game: SokobanGame) -> float:
        boxes = state.boxes
        goals = game.board.goals
        player = state.player
        
        h = 0.0
        min_dist_player2box = min([self._mahattan_dist(player, box) for box in boxes])
        
        mean_heuristic_dist_of_boxes2goals = 0.0
        n_boxes = len(boxes)
        for box in boxes:
            mean_heuristic_dist_of_boxes2goals += min([self._mahattan_dist(box, goal) for goal in goals]) / n_boxes
        
        h = min_dist_player2box + mean_heuristic_dist_of_boxes2goals
        
        return h
    

class SolutionVisualizer(ABC):
    @abstractmethod
    def display(self, game: SokobanGame, moves: List[Move]) -> None:
        raise NotImplementedError


class ConsoleVisualizer(SolutionVisualizer):
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s

    def display(self, game: SokobanGame, moves: List[Move]) -> None:
        history = game.replay(moves)
        for index, state in enumerate(history):
            print(f"\nStep {index}")
            print(game.render(state))
            if self.delay_s:
                time.sleep(self.delay_s)


class PlaywrightVisualizer(SolutionVisualizer):
    def __init__(self, level_url: str = "", delay_s: float = 0.0) -> None:
        self.level_url = level_url
        self.delay_s = delay_s

    def display(self, game: SokobanGame, moves: List[Move]) -> None:
        for move in moves:
            pyautogui.press(move)
            time.sleep(self.delay_s)


class LevelCatalog:
    def __init__(self) -> None:
        self._url_by_id: Dict[str, str] = {}

    def register(self, level_id: str, url: str) -> None:
        self._url_by_id[level_id] = url

    def get(self, level_id: str) -> str:
        if level_id not in self._url_by_id:
            raise KeyError(f"Unknown level identifier: {level_id}")
        return self._url_by_id[level_id]


def load_board_from_path(path: Path) -> SokobanBoard:
    return SokobanBoard.from_ascii(path.read_text(encoding="utf-8"))


def load_board_from_file(path: str | Path) -> SokobanBoard:
    return load_board_from_path(Path(path))


def load_game_from_path(path: Path) -> SokobanGame:
    return SokobanGame(load_board_from_path(path))


__all__ = [
    "SokobanBoard",
    "SokobanState",
    "SokobanGame",
    "Solver",
    "BFSSolver",
    "DFSSolver",
    "AStarSolver",
    "Heuristic",
    "ManhattanHeuristic",
    "SolutionVisualizer",
    "ConsoleVisualizer",
    "PlaywrightVisualizer",
    "LevelCatalog",
    "ComplexityTracker",
    "SolverLimits",
    "SolverResult",
    "load_board_from_path",
    "load_board_from_file",
    "load_game_from_path",
]

class Runner:
    def __init__(self, game_file: str, solver: Solver, visulizer: SolutionVisualizer, limits: Optional[SolverLimits]):
        self.game_file = game_file
        self.solver = solver
        self.visualizer = visulizer
        self.limits = limits
        self.game = load_game_from_path(GAME_ROOT/Path(self.game_file))
        
    def run_solver(self) -> None:
        result = self.solver.solve(self.game, self.limits)
        self._save_solution(result.move_sequence)
        self._visulize(result)
    
    def _save_solution(self, moves: List[str]) -> None:
        game_path = SOLUTION_ROOT/Path(self.game_file[:-4]+"_ga_solution.txt")
        moves = game_path.write_text(", ".join(moves))
    
    def _load_solution(self) -> List[str]:
        solution_path = SOLUTION_ROOT/Path(self.game_file[:-4]+"_ga_solution.txt")
        moves_str = solution_path.read_text(encoding="utf-8")
        return moves_str.split(", ")
    
    def _visulize(self, result: SolverResult) -> None:
        moves = self._load_solution()
        print(f"[INFO] Time elapsed: {result.elapsed_time_s}")
        print(f"[INFO] Used memory: {result.peak_memory_bytes}")
        print(f"[INFO] Nodes: {result.nodes_expanded}")
        print(f"[INFO] Max frontier: {result.max_frontier}")
        self.visualizer.display(self.game, moves)

if __name__ == "__main__":
    game_file = r"D:\1-REFERENCES\01-AI-ML-DL\10-GenAI\GAs\BTL-NMAI-251\test\game\easyv1.txt"
    heuristic = ManhattanHeuristic()
    solver = Genetic(heuristic=heuristic, pop_size=100, generations=500)
    visualizer = ConsoleVisualizer(delay_s=0.1)
    limits = None
    
    runner = Runner(game_file=game_file,
                    solver=solver,
                    visulizer=visualizer,
                    limits=limits)
    runner.run_solver()

    