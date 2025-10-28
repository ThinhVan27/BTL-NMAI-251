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
    """
    Enhanced Genetic Algorithm with simplified fitness function and more aggressive exploration
    """
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
        def __init__(self, rng=None, max_moves=2000):
            self._random = rng if rng is not None else np.random.default_rng()
            self._sequence: List[Move] = []
            self._max_moves = max_moves
            self._fitness: float = -float("inf")
            self._raw_fitness: float = -float("inf")
            self._normalized_fitness: float = 0.0
            self._diversity: float = 0.0
            self._valid_ratio: float = 0.0
            
        def _safe_random_int(self, low, high):
            """Safe random integer generator that handles edge cases"""
            if high <= low:
                return low
            return int(self._random.integers(low, high))
            
        # ========== Domain Knowledge Methods (Keep existing) ==========
        def _find_path_bfs(self, game: SokobanGame, state: SokobanState, target: Coord) -> List[Move]:
            """BFS pathfinding for player (avoiding boxes & walls)"""
            from collections import deque
            queue = deque([(state.player, [])])
            visited = {state.player}
            while queue:
                pos, path = queue.popleft()
                if pos == target:
                    return path
                if len(path) > 30:
                    continue
                for mv, (dx, dy) in MOVE_VECTORS.items():
                    nxt = (pos[0] + dx, pos[1] + dy)
                    if (game.board.in_bounds(nxt) and 
                        not game.board.is_wall(nxt) and 
                        nxt not in state.boxes and 
                        nxt not in visited):
                        visited.add(nxt)
                        queue.append((nxt, path + [mv]))
            return []

        def _is_advanced_deadlock(self, game: SokobanGame, state: SokobanState) -> bool:
            """ENHANCED deadlock detection with multiple patterns"""
            walls = game.board.walls
            goals = game.board.goals
            
            for box in state.boxes:
                if box in goals:
                    continue
                
                x, y = box
                
                # 1. Corner deadlock (4 patterns)
                corner_patterns = [
                    ((x-1, y) in walls and (x, y-1) in walls),
                    ((x+1, y) in walls and (x, y-1) in walls),
                    ((x-1, y) in walls and (x, y+1) in walls),
                    ((x+1, y) in walls and (x, y+1) in walls)
                ]
                if any(corner_patterns):
                    return True
                
                # 2. Edge deadlock (box against wall with no goal on that line)
                # Horizontal wall check
                if (x, y-1) in walls or (x, y+1) in walls:
                    # Check if there's any goal on this row
                    if not any(g[1] == y for g in goals):
                        return True
                
                # Vertical wall check
                if (x-1, y) in walls or (x+1, y) in walls:
                    # Check if there's any goal on this column
                    if not any(g[0] == x for g in goals):
                        return True
                
                # 3. Frozen box (cannot be pushed in any direction)
                pushable_count = 0
                for dx, dy in MOVE_VECTORS.values():
                    from_pos = (x - dx, y - dy)
                    to_pos = (x + dx, y + dy)
                    if (game.board.in_bounds(from_pos) and 
                        from_pos not in state.boxes and 
                        not game.board.is_wall(from_pos) and
                        game.board.in_bounds(to_pos) and 
                        to_pos not in state.boxes and 
                        not game.board.is_wall(to_pos)):
                        pushable_count += 1
                
                if pushable_count == 0:
                    return True
                
                # 4. Two-box freeze pattern (boxes blocking each other)
                for other_box in state.boxes:
                    if other_box == box or other_box in goals:
                        continue
                    
                    ox, oy = other_box
                    # Check if boxes are adjacent
                    if abs(x - ox) + abs(y - oy) == 1:
                        # Check if they form a blocking pattern with walls
                        # Horizontal adjacency
                        if x == ox:
                            if ((x-1, y) in walls and (ox-1, oy) in walls) or \
                               ((x+1, y) in walls and (ox+1, oy) in walls):
                                return True
                        # Vertical adjacency
                        if y == oy:
                            if ((x, y-1) in walls and (ox, oy-1) in walls) or \
                               ((x, y+1) in walls and (ox, oy+1) in walls):
                                return True
            
            return False

        def _calculate_min_matching_distance(self, boxes: List[Coord], goals: List[Coord]) -> float:
            """Calculate minimum matching distance between boxes and goals"""
            if not boxes or not goals:
                return 0.0
                
            try:
                from scipy.optimize import linear_sum_assignment
                cost = np.zeros((len(boxes), len(goals)))
                for i, b in enumerate(boxes):
                    for j, g in enumerate(goals):
                        cost[i, j] = abs(b[0]-g[0]) + abs(b[1]-g[1])
                r, c = linear_sum_assignment(cost)
                return float(cost[r, c].sum())
            except:
                # Fallback: greedy matching
                remaining_goals = set(goals)
                total_dist = 0.0
                for b in boxes:
                    min_dist = float('inf')
                    best_g = None
                    for g in remaining_goals:
                        dist = abs(b[0]-g[0]) + abs(b[1]-g[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_g = g
                    if best_g:
                        total_dist += min_dist
                        remaining_goals.discard(best_g)
                return total_dist

        def _count_state_cycles(self, history: List[SokobanState]) -> int:
            """Count how many times states repeat (cycle detection)"""
            if len(history) < 2:
                return 0
            
            # Use box positions as state signature
            seen_states = {}
            cycle_count = 0
            
            for i, state in enumerate(history):
                state_sig = frozenset(state.boxes)
                if state_sig in seen_states:
                    cycle_count += 1
                else:
                    seen_states[state_sig] = i
            
            return cycle_count

        def _generate_smart_macro(self, game: SokobanGame, state: SokobanState) -> List[Move]:
            """Safer macro-action: try all push directions, simulate, avoid corner/tunnel deadlocks."""
            boxes_not_on_goal = [b for b in state.boxes if b not in game.board.goals]
            if not boxes_not_on_goal:
                return []

            goals = list(game.board.goals)
            best_seq: List[Move] = []
            best_score = float('inf')

            # helper: tunnel deadlock detection (simple)
            def _is_tunnel_deadlock(box: Coord) -> bool:
                x, y = box
                walls = game.board.walls
                # horizontal tunnel (walls above and below)
                if ((x, y-1) in walls or (x, y+1) in walls):
                    # check if any goal on that row
                    if not any(g[1] == y for g in goals):
                        return True
                # vertical tunnel (walls left and right)
                if ((x-1, y) in walls or (x+1, y) in walls):
                    if not any(g[0] == x for g in goals):
                        return True
                return False

            for box in boxes_not_on_goal:
                bx, by = box
                min_dist = min(abs(bx-gx)+abs(by-gy) for gx,gy in goals)
                
                for push_dir, (dx, dy) in MOVE_VECTORS.items():
                    req_pos = (bx - dx, by - dy)
                    box_target = (bx + dx, by + dy)

                    # basic feasibility checks
                    if not game.board.in_bounds(req_pos) or not game.board.in_bounds(box_target):
                        continue
                    if game.board.is_wall(box_target):
                        continue
                    if req_pos in state.boxes or box_target in state.boxes:
                        continue

                    # corner/tunnel check at target
                    walls = game.board.walls
                    corner_check = [
                        ((box_target[0]-1, box_target[1]) in walls and (box_target[0], box_target[1]-1) in walls),
                        ((box_target[0]+1, box_target[1]) in walls and (box_target[0], box_target[1]-1) in walls),
                        ((box_target[0]-1, box_target[1]) in walls and (box_target[0], box_target[1]+1) in walls),
                        ((box_target[0]+1, box_target[1]) in walls and (box_target[0], box_target[1]+1) in walls),
                    ]
                    if any(corner_check) and box_target not in goals:
                        continue
                    if _is_tunnel_deadlock(box_target) and box_target not in goals:
                        continue

                    # path for player to required position
                    path = self._find_path_bfs(game, state, req_pos)
                    if not path:
                        continue

                    # simulate moves (path + push) and check for deadlock after push
                    sim_state = state
                    valid = True
                    simulated_moves = []
                    for mv in path + [push_dir]:
                        ns = game.apply_move(sim_state, mv)
                        if ns is None:
                            valid = False
                            break
                        simulated_moves.append(mv)
                        sim_state = ns
                    if not valid:
                        continue

                    # after push, check advanced deadlock & tunnel
                    if self._is_advanced_deadlock(game, sim_state):
                        continue
                    if any(_is_tunnel_deadlock(b) for b in sim_state.boxes if b not in goals):
                        continue

                    # measure resulting distance to selected goal
                    new_dist = self._calculate_min_matching_distance(list(sim_state.boxes), goals)
                    score = new_dist + 0.1 * len(simulated_moves) + 0.01 * min_dist

                    if score < best_score:
                        best_score = score
                        best_seq = simulated_moves

            # allow a couple of extra safe pushes if available
            if best_seq:
                current = state
                for _ in range(2):
                    last = best_seq[-1]
                    ns = game.apply_move(current, last)
                    if ns is None or self._is_advanced_deadlock(game, ns):
                        break
                    best_seq.append(last)
                    current = ns

            return best_seq[:12]

        # ========== Initialization Methods ==========
        def init_random(self, game: SokobanGame, strategy: str = "balanced"):
            """Initialize with different strategies"""
            if strategy == "short":
                L = self._safe_random_int(5, 30)
                self._sequence = [self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(L)]
            
            elif strategy == "medium":
                L = self._safe_random_int(30, 100)
                self._sequence = [self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(L)]
            
            elif strategy == "macro":
                self._sequence = []
                state = game.initial_state
                attempts = 0
                
                while len(self._sequence) < self._max_moves and attempts < 250:
                    attempts += 1
                    
                    if self._random.random() < 0.75:
                        macro = self._generate_smart_macro(game, state)
                        if macro:
                            for mv in macro:
                                if len(self._sequence) >= self._max_moves:
                                    break
                                self._sequence.append(mv)
                                ns = game.apply_move(state, mv)
                                if ns:
                                    state = ns
                                else:
                                    break
                        else:
                            # Fallback: small random walk
                            for _ in range(self._safe_random_int(2, 6)):
                                if len(self._sequence) >= self._max_moves:
                                    break
                                mv = self._random.choice(list(MOVE_VECTORS.keys()))
                                ns = game.apply_move(state, mv)
                                if ns:
                                    self._sequence.append(mv)
                                    state = ns
                    else:
                        # Random exploration
                        for _ in range(self._safe_random_int(3, 10)):
                            if len(self._sequence) >= self._max_moves:
                                break
                            mv = self._random.choice(list(MOVE_VECTORS.keys()))
                            ns = game.apply_move(state, mv)
                            if ns:
                                self._sequence.append(mv)
                                state = ns
                
                # Ensure minimum length
                if len(self._sequence) < 5:
                    self._sequence.extend([self._random.choice(list(MOVE_VECTORS.keys())) 
                                          for _ in range(5 - len(self._sequence))])
            
            else:  # balanced
                if self._random.random() < 0.5:
                    L = self._safe_random_int(10, 50)
                else:
                    L = self._safe_random_int(50, 150)
                self._sequence = [self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(L)]

        # ========== Genetic Operators ==========
        def crossover(self, other: "Genetic.Chromosome", method: str = "adaptive") -> "Genetic.Chromosome":
            """Enhanced crossover with multiple strategies"""
            child = Genetic.Chromosome(self._random, self._max_moves)
            a, b = self._sequence, other._sequence
            
            if not a or not b:
                child._sequence = list(a or b)
                return child
            
            if method == "two_point":
                minlen = min(len(a), len(b))
                if minlen > 2:
                    p1 = self._safe_random_int(1, minlen)
                    p2 = self._safe_random_int(p1, minlen)
                    child._sequence = a[:p1] + b[p1:p2] + (a[p2:] if p2 < len(a) else [])
                else:
                    child._sequence = a[:len(a)//2] + b[len(b)//2:]
            
            elif method == "uniform":
                maxlen = max(len(a), len(b))
                for i in range(maxlen):
                    if self._random.random() < 0.5:
                        if i < len(a):
                            child._sequence.append(a[i])
                    else:
                        if i < len(b):
                            child._sequence.append(b[i])
            
            else:  # adaptive
                if abs(self._fitness - other._fitness) < 1000:
                    maxlen = max(len(a), len(b))
                    for i in range(min(maxlen, self._max_moves)):
                        if self._random.random() < 0.5 and i < len(a):
                            child._sequence.append(a[i])
                        elif i < len(b):
                            child._sequence.append(b[i])
                else:
                    minlen = min(len(a), len(b))
                    if minlen > 2:
                        p1 = self._safe_random_int(1, minlen)
                        p2 = self._safe_random_int(p1, minlen)
                        child._sequence = a[:p1] + b[p1:p2] + (a[p2:] if p2 < len(a) else [])
            
            if len(child._sequence) > self._max_moves:
                child._sequence = child._sequence[:self._max_moves]
            
            return child

        def mutate(self, game: SokobanGame, mut_rate: float, strategy: str = "adaptive") -> "Genetic.Chromosome":
            """ENHANCED: More aggressive mutation with varied types and safe bounds checking"""
            child = Genetic.Chromosome(self._random, self._max_moves)
            child._sequence = list(self._sequence)
            
            # AGGRESSIVE: Complete recreation with higher probability
            if not child._sequence or self._random.random() < 0.3:
                child.init_random(game, self._random.choice(["macro", "balanced"]))
                return child
            
            mutation_type = self._random.random()
            
            # Large-scale mutation (40% chance) - only if sequence is long enough
            if mutation_type < 0.4 and len(child._sequence) > 10:
                start = self._safe_random_int(0, max(1, len(child._sequence) - 5))
                end = start + self._safe_random_int(5, min(15, len(child._sequence) - start))
                
                state = game.initial_state
                for mv in child._sequence[:start]:
                    ns = game.apply_move(state, mv)
                    if ns: state = ns
                
                new_macro = self._generate_smart_macro(game, state)
                if new_macro:
                    child._sequence[start:end] = new_macro
            
            # Multiple point mutations (30% chance)
            elif mutation_type < 0.7:
                num_mutations = max(1, int(len(child._sequence) * 0.3))
                for _ in range(num_mutations):
                    if child._sequence:  # Only mutate if sequence is not empty
                        idx = self._safe_random_int(0, len(child._sequence))
                        child._sequence[idx] = self._random.choice(list(MOVE_VECTORS.keys()))
            
            # Structural mutations (30% chance) - only if sequence is long enough
            else:
                if len(child._sequence) > 3:
                    # Random segment deletion (30% chance)
                    if self._random.random() < 0.3 and len(child._sequence) > 1:
                        start = self._safe_random_int(0, len(child._sequence) - 1)
                        length = self._safe_random_int(1, min(4, len(child._sequence) - start))
                        del child._sequence[start:start + length]
                    
                    # Random segment duplication (20% chance)
                    if self._random.random() < 0.2 and len(child._sequence) < self._max_moves and len(child._sequence) > 1:
                        start = self._safe_random_int(0, len(child._sequence) - 1)
                        length = self._safe_random_int(1, min(3, len(child._sequence) - start))
                        segment = child._sequence[start:start + length]
                        insert_pos = self._safe_random_int(0, len(child._sequence))
                        child._sequence[insert_pos:insert_pos] = segment

            # Guided insertion - only if sequence has space
            if self._random.random() < mut_rate * 0.5 and len(child._sequence) < self._max_moves:
                pos = self._safe_random_int(0, len(child._sequence) + 1) if child._sequence else 0
                state = game.initial_state
                for mv in child._sequence[:pos]:
                    ns = game.apply_move(state, mv)
                    if ns:
                        state = ns
                macro = self._generate_smart_macro(game, state)
                if macro:
                    seg = macro[:min(len(macro), 5)]
                    child._sequence[pos:pos] = seg
            
            # AGGRESSIVE deadlock repair
            try:
                history = game.replay(child._sequence)
                final = history[-1]
                
                if self._is_advanced_deadlock(game, final):
                    # Rollback more aggressively but safely
                    rollback = min(15, max(1, len(child._sequence)//4))
                    child._sequence = child._sequence[:-rollback] if rollback > 0 and len(child._sequence) > rollback else []
            except ValueError:
                # Truncate at first illegal move
                seq = []
                state = game.initial_state
                for mv in child._sequence:
                    ns = game.apply_move(state, mv)
                    if ns is None:
                        break
                    seq.append(mv)
                    state = ns
                child._sequence = seq
            
            if len(child._sequence) > self._max_moves:
                child._sequence = child._sequence[:self._max_moves]
            
            return child

        # ========== SIMPLIFIED & AGGRESSIVE Fitness Function ==========
        def evaluate(self, game: SokobanGame, heuristic: "Heuristic", 
                    initial_dist: Optional[float] = None,
                    pop_stats: Optional[dict] = None):
            """
            SIMPLIFIED fitness with:
            - Massive reward for complete solutions
            - Quadratic reward for partial progress
            - Reduced complexity
            """
            try:
                history = game.replay(self._sequence)
                final_state = history[-1]
                valid_moves = len(history) - 1
                self._valid_ratio = valid_moves / max(1, len(self._sequence))
                
                # === COMPONENT 1: Goal Achievement (MASSIVE REWARD) ===
                boxes_on_goals = sum(1 for b in final_state.boxes if b in game.board.goals)
                total_boxes = len(final_state.boxes)
                
                # PRIMARY: Complete solution bonus
                if game.is_goal(final_state):
                    self._fitness = 10_000_000 - len(self._sequence)  # Massive reward
                    self._raw_fitness = self._fitness
                    return self._fitness
                
                # SECONDARY: Quadratic reward for boxes on goals
                goal_ratio = boxes_on_goals / max(1, total_boxes)
                goal_score = 1_000_000 * (goal_ratio ** 2)  # Quadratic scaling
                
                # === COMPONENT 2: Distance Improvement ===
                boxes_list = list(final_state.boxes)
                goals_list = list(game.board.goals)
                current_dist = self._calculate_min_matching_distance(boxes_list, goals_list)
                
                dist_improvement = 0.0
                if initial_dist and current_dist < initial_dist:
                    dist_improvement = (initial_dist - current_dist) * 1000
                
                # === COMPONENT 3: Penalties (REDUCED) ===
                deadlock_penalty = -50_000 if self._is_advanced_deadlock(game, final_state) else 0
                length_penalty = -len(self._sequence) * 0.5  # Reduced penalty
                
                # === COMPONENT 4: Cycle Detection ===
                cycle_count = self._count_state_cycles(history)
                cycle_penalty = -cycle_count * 2000
                
                # === COMBINE (SIMPLIFIED) ===
                self._raw_fitness = (
                    goal_score + 
                    dist_improvement + 
                    length_penalty + 
                    deadlock_penalty +
                    cycle_penalty
                )
                
                # Normalize for population statistics
                if pop_stats and 'max_raw' in pop_stats and 'min_raw' in pop_stats:
                    range_val = max(1, pop_stats['max_raw'] - pop_stats['min_raw'])
                    self._normalized_fitness = (self._raw_fitness - pop_stats['min_raw']) / range_val
                else:
                    self._normalized_fitness = self._raw_fitness / 1_000_000
                
                self._fitness = self._raw_fitness
                
            except ValueError:
                # Penalty for invalid sequences
                prefix_len = 0
                state = game.initial_state
                for mv in self._sequence:
                    ns = game.apply_move(state, mv)
                    if ns is None:
                        break
                    prefix_len += 1
                    state = ns
                self._fitness = -100_000 + prefix_len * 50
                self._valid_ratio = prefix_len / max(1, len(self._sequence))
            
            return self._fitness

        def copy(self) -> "Genetic.Chromosome":
            """Deep copy"""
            c = Genetic.Chromosome(self._random, self._max_moves)
            c._sequence = list(self._sequence)
            c._fitness = self._fitness
            c._raw_fitness = self._raw_fitness
            c._normalized_fitness = self._normalized_fitness
            c._diversity = self._diversity
            c._valid_ratio = self._valid_ratio
            return c

    # ========== Main GA Class ==========
    def __init__(
        self,
        heuristic: Optional["Heuristic"] = None,
        pop_size: int = 300,
        generations: int = 5000,
        max_moves: int = 2000,
        mut_rate: float = 0.25,
        cross_rate: float = 0.65,
        elitism_rate: float = 0.03,
        tournament_size: int = 3,
        diversity_weight: float = 0.4
    ):
        super().__init__()
        self.heuristic = heuristic or ManhattanHeuristic()
        self.pop_size = max(20, pop_size)
        self.generations = generations
        self.max_moves = max_moves
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.diversity_weight = diversity_weight
        self._random = np.random.default_rng()
        self.population: List[Genetic.Chromosome] = []
        self.best_ever: Optional[Genetic.Chromosome] = None
        self.best_ever_fitness = -float("inf")
        
    def _init_population(self, game: SokobanGame):
        """Initialize diverse population with more macro-based individuals"""
        self.population = []
        
        n_short = int(self.pop_size * 0.10)   # Reduced
        n_medium = int(self.pop_size * 0.20)  # Reduced  
        n_macro = int(self.pop_size * 0.60)   # Increased macro-based
        n_balanced = self.pop_size - n_short - n_medium - n_macro
        
        strategies = (
            ["short"] * n_short +
            ["medium"] * n_medium +
            ["macro"] * n_macro +
            ["balanced"] * n_balanced
        )
        
        for strategy in strategies:
            c = Genetic.Chromosome(self._random, self.max_moves)
            c.init_random(game, strategy)
            self.population.append(c)
        
        # Calculate initial distance for progress tracking
        try:
            boxes = list(game.board.initial_boxes)
            goals = list(game.board.goals)
            self._initial_dist = self._calculate_min_matching_distance(boxes, goals)
        except:
            self._initial_dist = None
        
        self._evaluate_population(game)
        
    def _calculate_min_matching_distance(self, boxes: List[Coord], goals: List[Coord]) -> float:
        """Calculate minimum matching distance between boxes and goals"""
        if not boxes or not goals:
            return 0.0
            
        try:
            from scipy.optimize import linear_sum_assignment
            cost = np.zeros((len(boxes), len(goals)))
            for i, b in enumerate(boxes):
                for j, g in enumerate(goals):
                    cost[i, j] = abs(b[0]-g[0]) + abs(b[1]-g[1])
            r, c = linear_sum_assignment(cost)
            return float(cost[r, c].sum())
        except:
            # Fallback: greedy matching
            remaining_goals = set(goals)
            total_dist = 0.0
            for b in boxes:
                min_dist = float('inf')
                best_g = None
                for g in remaining_goals:
                    dist = abs(b[0]-g[0]) + abs(b[1]-g[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_g = g
                if best_g:
                    total_dist += min_dist
                    remaining_goals.discard(best_g)
            return total_dist
        
    def _evaluate_population(self, game: SokobanGame):
        """Evaluate entire population with diversity enhancement"""
        for chrom in self.population:
            chrom.evaluate(game, self.heuristic, self._initial_dist)
        
        raw_fits = [c._raw_fitness for c in self.population]
        pop_stats = {
            'max_raw': max(raw_fits),
            'min_raw': min(raw_fits),
            'mean_raw': sum(raw_fits) / len(raw_fits)
        }
        
        # Second pass for diversity and normalization
        for chrom in self.population:
            chrom.evaluate(game, self.heuristic, self._initial_dist, pop_stats)
            
            # Compute diversity (average Hamming distance to sample)
            if len(chrom._sequence) > 0:
                diversity_sum = 0
                sample_size = min(25, len(self.population))
                samples = self._random.choice(self.population, size=sample_size, replace=False)
                for other in samples:
                    if other is chrom:
                        continue
                    min_len = min(len(chrom._sequence), len(other._sequence))
                    if min_len > 0:
                        diff = sum(1 for i in range(min_len) 
                                   if chrom._sequence[i] != other._sequence[i])
                        diversity_sum += diff / min_len
                chrom._diversity = diversity_sum / max(1, sample_size - 1)
            
            # Add diversity to fitness (STRONGER weight)
            chrom._fitness += chrom._diversity * self.diversity_weight * 5000
        
        # Sort population by fitness
        self.population.sort(key=lambda c: c._fitness, reverse=True)
        
        # Update best ever
        if self.population[0]._fitness > self.best_ever_fitness:
            self.best_ever = self.population[0].copy()
            self.best_ever_fitness = self.population[0]._fitness
    
    def _tournament_select(self) -> Genetic.Chromosome:
        """Tournament selection with smaller size for more pressure"""
        size = min(self.tournament_size, len(self.population))
        contestants = self._random.choice(self.population, size=size, replace=False)
        return max(contestants, key=lambda c: c._fitness)
    
    def _local_search(self, chrom: Genetic.Chromosome, game: SokobanGame, iters: int = 5) -> Genetic.Chromosome:
        """Simple local search: try removing redundant moves"""
        best = chrom.copy()
        best_fit = best._fitness
        
        for _ in range(iters):
            if len(best._sequence) <= 1:
                break
            improved = False
            for i in range(len(best._sequence)):
                cand_seq = best._sequence[:i] + best._sequence[i+1:]
                cand = Genetic.Chromosome(self._random, best._max_moves)
                cand._sequence = cand_seq
                cand.evaluate(game, self.heuristic, self._initial_dist)
                if cand._fitness > best_fit:
                    best = cand
                    best_fit = cand._fitness
                    improved = True
                    break
            if not improved:
                break
        
        return best
    
    def _evolve_generation(self, game: SokobanGame):
        """Evolve one generation"""
        # Elitism (reduced)
        elite_count = max(1, int(self.pop_size * self.elitism_rate))
        elites = [c.copy() for c in self.population[:elite_count]]
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.pop_size - elite_count:
            parent1 = self._tournament_select()
            
            if self._random.random() < self.cross_rate:
                parent2 = self._tournament_select()
                child = parent1.crossover(parent2, method="adaptive")
            else:
                child = parent1.copy()
            
            child = child.mutate(game, self.mut_rate, strategy="adaptive")
            offspring.append(child)
        
        # Combine and evaluate
        self.population = elites + offspring
        self._evaluate_population(game)
        
        # Local search on top k (reduced)
        top_k = min(3, len(self.population))
        for i in range(top_k):
            improved = self._local_search(self.population[i], game, iters=2)
            if improved._fitness > self.population[i]._fitness:
                self.population[i] = improved
        
        self.population.sort(key=lambda c: c._fitness, reverse=True)
    
    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        """Main search loop with enhanced progress tracking"""
        self._init_population(game)
        self.best_ever = None
        self.best_ever_fitness = -float("inf")
        
        stagnation_count = 0
        best_history = []
        success_threshold = 5_000_000  # Adjusted for new fitness scale
        
        with tqdm(total=self.generations, desc="GA Progress", unit="gen") as pbar:
            for gen in range(self.generations):
                self.tracker.record_node()
                self.tracker.track_frontier(len(self.population))
                
                self._evolve_generation(game)
                
                best = self.population[0]
                best_history.append(best._fitness)
                
                # Enhanced progress tracking
                if gen % 50 == 0 or gen < 10:
                    try:
                        history = game.replay(best._sequence)
                        final = history[-1]
                        boxes_on_goal = sum(1 for b in final.boxes if b in game.board.goals)
                        tqdm.write(f"[Gen {gen:4d}] Boxes on goals: {boxes_on_goal}/{len(final.boxes)}, Fitness: {best._fitness:.0f}")
                    except:
                        tqdm.write(f"[Gen {gen:4d}] Fitness: {best._fitness:.0f}")
                
                # Adaptive mutation
                if len(best_history) > 50:
                    recent_best = max(best_history[-30:])
                    old_best = max(best_history[-60:-30]) if len(best_history) >= 60 else best_history[0]
                    
                    if recent_best - old_best < 5000:  # More sensitive threshold
                        stagnation_count += 1
                        self.mut_rate = min(0.5, self.mut_rate * 1.2)  # More aggressive increase
                    else:
                        stagnation_count = max(0, stagnation_count - 1)
                        self.mut_rate = max(0.1, self.mut_rate * 0.9)
                
                # Update progress bar
                try:
                    history = game.replay(best._sequence)
                    final = history[-1]
                    boxes_on_goal = sum(1 for b in final.boxes if b in game.board.goals)
                    pbar.set_postfix({
                        "fit": f"{best._fitness:.0f}",
                        "boxes": f"{boxes_on_goal}/{len(final.boxes)}",
                        "moves": len(best._sequence),
                        "mut": f"{self.mut_rate:.3f}",
                        "stag": stagnation_count
                    })
                except:
                    pbar.set_postfix({
                        "fit": f"{best._fitness:.0f}",
                        "moves": len(best._sequence),
                        "mut": f"{self.mut_rate:.3f}",
                        "stag": stagnation_count
                    })
                pbar.update(1)
                
                # Check solution
                if best._fitness >= success_threshold:
                    try:
                        history = game.replay(best._sequence)
                        if game.is_goal(history[-1]):
                            tqdm.write(f"[SUCCESS] Solution found at gen {gen} with {len(best._sequence)} moves!")
                            return True, best._sequence
                    except:
                        pass
                
                # Restart on stagnation (more frequent)
                if stagnation_count >= 50:
                    tqdm.write(f"[RESTART] Stagnation at gen {gen}. Restarting with elite preservation.")
                    elite_archive = [c.copy() for c in self.population[:max(3, int(self.pop_size * 0.1))]]
                    self._init_population(game)
                    for i, elite in enumerate(elite_archive):
                        if i < len(self.population):
                            self.population[i] = elite
                    self._evaluate_population(game)
                    stagnation_count = 0
                    best_history = []
                    self.mut_rate = 0.25
                
                # Diversity injection (more frequent)
                elif stagnation_count > 0 and stagnation_count % 15 == 0:
                    tqdm.write(f"[DIVERSITY] Injecting random immigrants at gen {gen}")
                    n_immigrants = max(8, int(self.pop_size * 0.15))
                    for i in range(n_immigrants):
                        idx = len(self.population) - 1 - i
                        immigrant = Genetic.Chromosome(self._random, self.max_moves)
                        strategy = self._random.choice(["macro", "medium", "balanced"])
                        immigrant.init_random(game, strategy)
                        immigrant.evaluate(game, self.heuristic, self._initial_dist)
                        self.population[idx] = immigrant
                    self._evaluate_population(game)
        
        # Return best found
        best_final = self.population[0] if self.population else None
        if best_final:
            try:
                history = game.replay(best_final._sequence)
                if game.is_goal(history[-1]):
                    tqdm.write("[SUCCESS] Exact solution found at final generation!")
                    return True, best_final._sequence
                else:
                    boxes_on_goal = sum(1 for b in history[-1].boxes if b in game.board.goals)
                    tqdm.write(f"[PARTIAL] Best solution has {boxes_on_goal}/{len(history[-1].boxes)} boxes on goals")
                    return False, best_final._sequence
            except ValueError:
                valid_seq = []
                state = game.initial_state
                for mv in best_final._sequence:
                    ns = game.apply_move(state, mv)
                    if ns is None:
                        break
                    valid_seq.append(mv)
                    state = ns
                tqdm.write(f"[PARTIAL] Returning valid prefix of length {len(valid_seq)}")
                return False, valid_seq
        
        # Fallback to best-ever
        if self.best_ever:
            tqdm.write("[FALLBACK] Returning best-ever chromosome.")
            try:
                history = game.replay(self.best_ever._sequence)
                return game.is_goal(history[-1]), self.best_ever._sequence
            except:
                valid_seq = []
                state = game.initial_state
                for mv in self.best_ever._sequence:
                    ns = game.apply_move(state, mv)
                    if ns is None:
                        break
                    valid_seq.append(mv)
                    state = ns
                return False, valid_seq
        
        return False, []


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
    
    solver = Genetic(
        pop_size=50,           # Larger population
        generations=1000,       # More generations
        max_moves=1000,         # Allow longer sequences
        mut_rate=0.25,          # Higher mutation rate
        cross_rate=0.65,        # Adjusted crossover
        elitism_rate=0.03,      # Less elitism
        tournament_size=3,      # More selection pressure
        diversity_weight=0.4    # Strong diversity emphasis
    )
    
    visualizer = ConsoleVisualizer(delay_s=0.05)  # Faster visualization
    limits = None
    
    runner = Runner(
        game_file=game_file,
        solver=solver,
        visulizer=visualizer,  # Fix typo: visulizer -> visualizer
        limits=limits
    )
    runner.run_solver()

    