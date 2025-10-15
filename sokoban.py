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
    name = 'genetic'

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
        def __init__(self, random=None, max_moves=1500):
            if random:
                self._random = random
            else:
                self._random = np.random.default_rng()
            self._sequence = []
            self._max_moves = max_moves
            self._fitness = None
            self._quality = None
            self._diversity = 0
            self._controlability = 0

        def is_deadlock(self, game: SokobanGame, state: SokobanState) -> bool:
            """Kiểm tra deadlock: box ở góc hoặc sát tường không có goal"""
            for box in state.boxes:
                if box in game.board.goals:
                    continue
                
                x, y = box
                walls = game.board.walls
                
                # Kiểm tra box ở 4 góc (2 cạnh là tường)
                if (x-1, y) in walls and (x, y-1) in walls:
                    return True
                if (x+1, y) in walls and (x, y-1) in walls:
                    return True
                if (x-1, y) in walls and (x, y+1) in walls:
                    return True
                if (x+1, y) in walls and (x, y+1) in walls:
                    return True
                
                # Box sát tường không có goal trên hàng/cột đó
                if (x, y-1) in walls or (x, y+1) in walls:
                    row_has_goal = any(g[1] == y for g in game.board.goals)
                    if not row_has_goal:
                        if (x-1, y) in walls or (x+1, y) in walls:
                            return True
                
                if (x-1, y) in walls or (x+1, y) in walls:
                    col_has_goal = any(g[0] == x for g in game.board.goals)
                    if not col_has_goal:
                        if (x, y-1) in walls or (x, y+1) in walls:
                            return True
            
            return False

        def _generate_macro_action(self, game: SokobanGame, state: SokobanState) -> list:
            """Tạo chuỗi moves để đẩy box về phía goal"""
            player = state.player
            boxes = state.boxes
            goals = game.board.goals
            
            # Tìm box gần goal nhất chưa ở goal
            target_box = None
            min_dist_to_goal = float('inf')
            
            for box in boxes:
                if box not in goals:
                    dist = min([abs(box[0]-g[0]) + abs(box[1]-g[1]) for g in goals])
                    if dist < min_dist_to_goal:
                        min_dist_to_goal = dist
                        target_box = box
            
            if not target_box:
                return None
            
            # Tìm goal gần nhất cho box này
            target_goal = min(goals, key=lambda g: abs(g[0]-target_box[0]) + abs(g[1]-target_box[1]))
            
            # Xác định hướng push box về goal
            dx = target_goal[0] - target_box[0]
            dy = target_goal[1] - target_box[1]
            
            # Chọn hướng push ưu tiên
            push_dir = None
            if abs(dx) > abs(dy):
                push_dir = "right" if dx > 0 else "left"
            else:
                push_dir = "down" if dy > 0 else "up"
            
            # Player cần đến vị trí đối diện với box để push
            push_vector = MOVE_VECTORS[push_dir]
            required_player_pos = (target_box[0] - push_vector[0], target_box[1] - push_vector[1])
            
            # Di chuyển player đến vị trí cần thiết
            moves = []
            px, py = player
            rx, ry = required_player_pos
            
            # Di chuyển theo x trước
            while px != rx and len(moves) < 10:
                if px < rx:
                    moves.append("right")
                    px += 1
                else:
                    moves.append("left")
                    px -= 1
            
            # Di chuyển theo y
            while py != ry and len(moves) < 10:
                if py < ry:
                    moves.append("down")
                    py += 1
                else:
                    moves.append("up")
                    py -= 1
            
            # Push box
            moves.append(push_dir)
            
            return moves[:15]

        def random(self, game):
            """Random thuần túy (fallback)"""
            rand_val = self._random.random()
            if rand_val < 0.3:
                length = self._random.integers(10, 50)
            elif rand_val < 0.7:
                length = self._random.integers(50, 150)
            else:
                length = self._random.integers(150, min(300, self._max_moves))
            
            self._sequence = [self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(length)]

        def random_with_macro_actions(self, game):
            """Tạo sequence sử dụng macro actions - PHƯƠNG THỨC CHÍNH"""
            self._sequence = []
            current_state = game.initial_state
            max_attempts = 200
            attempts = 0
            
            while len(self._sequence) < self._max_moves and attempts < max_attempts:
                attempts += 1
                
                # 60% macro action, 40% exploration
                if self._random.random() < 0.6:
                    macro = self._generate_macro_action(game, current_state)
                    if macro:
                        for move in macro:
                            self._sequence.append(move)
                            next_state = game.apply_move(current_state, move)
                            if next_state:
                                current_state = next_state
                            else:
                                break
                        
                        # Kiểm tra deadlock
                        if self.is_deadlock(game, current_state):
                            # Revert một số moves
                            revert = min(5, len(self._sequence))
                            self._sequence = self._sequence[:-revert]
                            current_state = game.initial_state
                            try:
                                history = game.replay(self._sequence)
                                current_state = history[-1]
                            except:
                                current_state = game.initial_state
                                self._sequence = []
                else:
                    # Random exploration
                    num_moves = self._random.integers(3, 10)
                    for _ in range(num_moves):
                        move = self._random.choice(list(MOVE_VECTORS.keys()))
                        self._sequence.append(move)
                        next_state = game.apply_move(current_state, move)
                        if next_state:
                            current_state = next_state
                            if self.is_deadlock(game, current_state):
                                self._sequence = self._sequence[:-1]
                                break
            
            # Trim if too long
            if len(self._sequence) > self._max_moves:
                self._sequence = self._sequence[:self._max_moves]
            
            # Nếu quá ngắn, thêm random
            if len(self._sequence) < 10:
                self._sequence.extend([self._random.choice(list(MOVE_VECTORS.keys())) for _ in range(10)])

        def crossover(self, chromosome):
            """Two-point crossover"""
            child = Genetic.Chromosome(self._random, self._max_moves)
            
            len1 = len(self._sequence)
            len2 = len(chromosome._sequence)
            
            if len1 == 0 or len2 == 0:
                child._sequence = list(self._sequence if len1 > 0 else chromosome._sequence)
                return child
            
            min_len = min(len1, len2)
            if min_len > 2:
                point1 = self._random.integers(1, min_len)
                point2 = self._random.integers(point1, min_len)
                
                child._sequence = (
                    self._sequence[:point1] + 
                    chromosome._sequence[point1:point2] + 
                    (self._sequence[point2:] if point2 < len1 else [])
                )
            else:
                split = self._random.integers(min_len) if min_len > 0 else 0
                child._sequence = self._sequence[:split] + chromosome._sequence[split:]
            
            return child

        def mutation(self, game, mut_rate):
            """Mutation với big mutation"""
            child = Genetic.Chromosome(self._random, self._max_moves)
            child._sequence = self._sequence.copy()
            
            # Big mutation 20%
            if self._random.random() < 0.2:
                num_changes = self._random.integers(
                    max(1, len(child._sequence) // 3),
                    max(2, len(child._sequence) // 2) + 1
                )
                if len(child._sequence) > 0:
                    positions = self._random.choice(
                        len(child._sequence), 
                        size=min(num_changes, len(child._sequence)), 
                        replace=False
                    )
                    for pos in positions:
                        child._sequence[pos] = self._random.choice(list(MOVE_VECTORS.keys()))
            else:
                # Normal mutation
                for i in range(len(child._sequence)):
                    if self._random.random() < mut_rate:
                        child._sequence[i] = self._random.choice(list(MOVE_VECTORS.keys()))
            
            # Insert operations
            if self._random.random() < mut_rate and len(child._sequence) < self._max_moves:
                num_inserts = self._random.integers(1, 5)
                for _ in range(num_inserts):
                    if len(child._sequence) < self._max_moves:
                        insert_pos = self._random.integers(len(child._sequence) + 1) if len(child._sequence) > 0 else 0
                        child._sequence.insert(insert_pos, self._random.choice(list(MOVE_VECTORS.keys())))
            
            # Delete operation
            if len(child._sequence) > 1 and self._random.random() < mut_rate:
                del child._sequence[self._random.integers(len(child._sequence))]
            
            return child

        def evaluate(self, game: SokobanGame, heuristic: Heuristic):
            """Fitness function với deadlock detection"""
            try:
                history = game.replay(self._sequence)
                final_state = history[-1]
                
                # Check deadlock - phạt nặng
                if self.is_deadlock(game, final_state):
                    self._fitness = -500000
                    self._quality = self._fitness
                    return self._fitness
                
                # GOAL!
                if game.is_goal(final_state):
                    self._fitness = 10000000 - len(self._sequence) * 100
                    self._quality = self._fitness
                    return self._fitness
                
                # Boxes on goals (quan trọng nhất)
                boxes_on_goals = len([b for b in final_state.boxes if b in game.board.goals])
                box_goal_score = boxes_on_goals * 100000
                
                # Tổng khoảng cách boxes đến goals
                total_box_dist = 0
                for box in final_state.boxes:
                    min_dist = min([abs(box[0]-g[0]) + abs(box[1]-g[1]) for g in game.board.goals])
                    total_box_dist += min_dist
                dist_score = -total_box_dist * 1000
                
                # Đếm số lần push
                push_count = 0
                for i in range(len(history) - 1):
                    if history[i].boxes != history[i+1].boxes:
                        push_count += 1
                push_score = push_count * 1000
                
                # Progress so với initial
                initial_total_dist = 0
                for box in game.board.initial_boxes:
                    initial_total_dist += min([abs(box[0]-g[0]) + abs(box[1]-g[1]) for g in game.board.goals])
                progress = (initial_total_dist - total_box_dist) * 500
                
                # Penalty nhẹ cho độ dài
                length_penalty = -len(self._sequence) * 1
                
                self._fitness = (
                    box_goal_score +
                    dist_score +
                    push_score +
                    progress +
                    length_penalty
                )
                
            except ValueError:
                self._fitness = -1000000
            
            self._quality = self._fitness
            return self._fitness

        def quality(self):
            return self._quality if self._quality is not None else -float('inf')

        def diversity(self):
            return self._diversity

        def controlability(self):
            return self._controlability

        def save(self, filepath):
            savedObject = {
                "sequence": self._sequence,
                "fitness": self._fitness,
                "quality": self._quality,
                "diversity": self._diversity,
                "controlability": self._controlability
            }
            with open(filepath, 'w') as f:
                f.write(json.dumps(savedObject, cls=Genetic.NpEncoder))

        def load(self, filepath):
            with open(filepath, 'r') as f:
                savedObject = json.loads("".join(f.readlines()))
                self._sequence = savedObject["sequence"]
                self._fitness = savedObject["fitness"]
                self._quality = savedObject["quality"]
                self._diversity = savedObject["diversity"]
                self._controlability = savedObject["controlability"]

    def __init__(self, heuristic: Optional[Heuristic] = None, pop_size=100, generations=200):
        super().__init__()
        self.heuristic = heuristic or ManhattanHeuristic()
        self.pop_size = pop_size
        self.generations = generations
        self._random = np.random.default_rng()
        self._chromosomes = []
        self._fitness_fn = self.fitness_quality

    def _bfs_seeding(self, game: SokobanGame, max_depth=10) -> list:
        """Dùng BFS để tìm sequences tốt làm seed"""
        from collections import deque
        
        queue = deque([(game.initial_state, [])])
        visited = {game.initial_state}
        good_sequences = []
        
        while queue and len(good_sequences) < 20:
            state, moves = queue.popleft()
            
            if len(moves) >= max_depth:
                good_sequences.append(moves)
                continue
            
            for move, next_state in game.successors(state):
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, moves + [move]))
        
        return good_sequences

    def _reset(self, game: SokobanGame, **kwargs):
        fn_name = kwargs.get('fitness', 'fitness_quality')
        if hasattr(self, fn_name):
            self._fitness_fn = getattr(self, fn_name)
        else:
            raise ValueError(f"{fn_name} doesn't exist")

        self._tournment_size = kwargs.get('tournment_size', 3)
        self._cross_rate = kwargs.get('cross_rate', 0.7)
        self._mut_rate = kwargs.get('mut_rate', 0.3)
        self._elitism = math.ceil(self.pop_size * kwargs.get('elitism_perct', 0.05))

        self._chromosomes = []
        
        # Seed 20% từ BFS
        try:
            bfs_seeds = self._bfs_seeding(game, max_depth=12)
            for i in range(min(len(bfs_seeds), int(self.pop_size * 0.2))):
                chrom = self.Chromosome(self._random)
                chrom._sequence = list(bfs_seeds[i])
                self._chromosomes.append(chrom)
        except:
            pass
        
        # 80% còn lại từ macro actions
        while len(self._chromosomes) < self.pop_size:
            chrom = self.Chromosome(self._random)
            chrom.random_with_macro_actions(game)
            self._chromosomes.append(chrom)
        
        self._evaluate(game)

    def _select(self):
        size = min(self._tournment_size, self.pop_size)
        tournment = list(range(self.pop_size))
        self._random.shuffle(tournment)
        chromosomes = [self._chromosomes[tournment[i]] for i in range(size)]
        chromosomes.sort(key=lambda c: self._fitness_fn(c), reverse=True)
        return chromosomes[0]

    def _evaluate(self, game):
        for chrom in self._chromosomes:
            chrom.evaluate(game, self.heuristic)
        self._chromosomes.sort(key=lambda c: self._fitness_fn(c), reverse=True)

    def _local_search(self, chromosome: Chromosome, game: SokobanGame, max_iterations=5):
        """Local search: xóa moves thừa"""
        best = chromosome
        best_fitness = chromosome._fitness
        
        for _ in range(max_iterations):
            if len(best._sequence) > 1:
                improved = False
                for i in range(len(best._sequence)):
                    candidate = Genetic.Chromosome(self._random, best._max_moves)
                    candidate._sequence = best._sequence[:i] + best._sequence[i+1:]
                    candidate.evaluate(game, self.heuristic)
                    
                    if candidate._fitness > best_fitness:
                        best = candidate
                        best_fitness = candidate._fitness
                        improved = True
                        break
                
                if not improved:
                    break
        
        return best

    def _update(self, game):
        chromosomes = self._chromosomes[:self._elitism]
        
        # Random immigrants 5%
        num_immigrants = max(1, int(self.pop_size * 0.05))
        for _ in range(num_immigrants):
            immigrant = self.Chromosome(self._random)
            immigrant.random_with_macro_actions(game)
            chromosomes.append(immigrant)
        
        # Generate offspring
        while len(chromosomes) < self.pop_size:
            child = self._select()
            if self._random.random() < self._cross_rate:
                parent = self._select()
                child = child.crossover(parent)
            child = child.mutation(game, self._mut_rate)
            chromosomes.append(child)
        
        self._chromosomes = chromosomes
        self._evaluate(game)
        
        # Local search on top 5%
        top_k = max(1, int(self.pop_size * 0.05))
        for i in range(min(top_k, len(self._chromosomes))):
            self._chromosomes[i] = self._local_search(self._chromosomes[i], game, max_iterations=3)
        
        self._chromosomes.sort(key=lambda c: self._fitness_fn(c), reverse=True)

    def _best(self):
        return self._chromosomes[0] if self._chromosomes else None

    def save(self, folderpath):
        if os.path.exists(folderpath):
            shutil.rmtree(folderpath)
        os.makedirs(folderpath)
        for i, chrom in enumerate(self._chromosomes):
            chrom.save(os.path.join(folderpath, f"chromosome_{i}.json"))

    def load(self, folderpath):
        files = [os.path.join(folderpath, fn) for fn in os.listdir(folderpath) if "chromosome" in fn]
        self._chromosomes = []
        for fn in files:
            c = self.Chromosome(self._random)
            c.load(fn)
            self._chromosomes.append(c)
        self._chromosomes.sort(key=lambda c: self._fitness_fn(c), reverse=True)

    @staticmethod
    def fitness_quality(chromosome):
        return chromosome.quality()

    @staticmethod
    def fitness_quality_control(chromosome):
        result = chromosome.quality()
        if result >= 1:
            result += chromosome.controlability()
        return result / 2.0

    @staticmethod
    def fitness_quality_control_diversity(chromosome):
        result = chromosome.quality()
        if result >= 1:
            result += chromosome.controlability()
        if result >= 1 and chromosome.controlability() >= 1:
            result += chromosome.diversity()
        return result / 3.0

    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        self._reset(game)
        
        best_fitness_history = []
        stagnation_count = 0

        with tqdm(total=self.generations, desc="GA Progress", unit="gen") as pbar:
            for gen in range(self.generations):
                self._update(game)
                best_chrom = self._best()
                
                pbar.set_postfix({
                    "fit": f"{best_chrom._fitness:.0f}",
                    "q": f"{best_chrom._quality:.0f}",
                    "mv": len(best_chrom._sequence),
                    "stg": stagnation_count
                })
                pbar.update(1)

                if gen % 50 == 0:
                    tqdm.write(
                        f"Gen {gen}: Fit={best_chrom._fitness:.0f}, "
                        f"Q={best_chrom._quality:.0f}, "
                        f"Moves={len(best_chrom._sequence)}"
                    )

                # Stagnation check
                best_fitness_history.append(best_chrom._fitness)
                if len(best_fitness_history) > 100:
                    recent_best = max(best_fitness_history[-100:])
                    old_best = max(best_fitness_history[-200:-100]) if len(best_fitness_history) >= 200 else 0
                    
                    if abs(recent_best - old_best) < 100.0:
                        stagnation_count += 1
                        if stagnation_count >= 3:
                            tqdm.write(f"[RESTART] Stagnation at gen {gen}")
                            self._reset(game)
                            stagnation_count = 0
                            best_fitness_history = []
                    else:
                        stagnation_count = 0

                self.tracker.record_node()
                self.tracker.track_frontier(len(self._chromosomes))

                # Success check
                if best_chrom._fitness >= 9000000:
                    tqdm.write(f"[SUCCESS] Found at gen {gen}!")
                    break

        best_chrom = self._best()
        solved = best_chrom._fitness >= 9000000
        return solved, best_chrom._sequence
    

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
    game_file = r"D:\1. REFERENCES\1. AI - Machine Learning\10. GenAI\GAs\BTL-NMAI-251\test\game\easyv0.txt"
    heuristic = ManhattanHeuristic()
    solver = Genetic(heuristic=heuristic, pop_size=300, generations=3000)
    visualizer = ConsoleVisualizer(delay_s=0.1)
    limits = None
    
    runner = Runner(game_file=game_file,
                    solver=solver,
                    visulizer=visualizer,
                    limits=limits)
    runner.run_solver()

    