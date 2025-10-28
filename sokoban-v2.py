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
    Genetic algorithm ported from the provided script.
    - Moves are strings: "up","down","left","right"
    - Selection: keep top half (by points), mutate them, then crossover to refill population
    - Fitness is negated sum of Manhattan distances (lower distance -> higher fitness)
    """

    name = "genetic"

    INT_TO_MOVE = {0: "up", 1: "down", 2: "left", 3: "right"}
    MOVE_TO_INT = {v: k for k, v in INT_TO_MOVE.items()}

    class Chromosome:
        def __init__(self, rng: Optional[np.random.Generator] = None, seq: Optional[List[Move]] = None, max_length: int = 50):
            self.rng = rng if rng is not None else np.random.default_rng()
            self._sequence: List[Move] = list(seq) if seq is not None else []
            self.max_length = max_length
            self.points = float("inf")  # lower is better (sum distances)
            self.fitness = -float("inf")  # higher is better (=-points)
            self.valid_prefix_len = 0

        def init_random(self, initial_length: int):
            L = max(1, int(self.rng.integers(1, initial_length + 1)))
            self._sequence = [Genetic.INT_TO_MOVE[int(self.rng.integers(0, 4))] for _ in range(L)]

        def copy(self) -> "Genetic.Chromosome":
            c = Genetic.Chromosome(self.rng, list(self._sequence), self.max_length)
            c.points = self.points
            c.fitness = self.fitness
            c.valid_prefix_len = self.valid_prefix_len
            return c

        def evaluate(self, game: "SokobanGame"):
            # simulate legal prefix
            state = game.initial_state
            seq = self._sequence
            valid = 0
            for mv in seq:
                ns = game.apply_move(state, mv)
                if ns is None:
                    break
                state = ns
                valid += 1
            self.valid_prefix_len = valid

            # compute points = sum of min manhattan distances box -> goal
            boxes = list(state.boxes)
            goals = list(game.board.goals)
            total = 0
            for b in boxes:
                if goals:
                    total += min(abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in goals)
                else:
                    total += 0
            self.points = total
            # fitness: higher better => negative points
            self.fitness = -float(self.points)
            return self.fitness

    # ---------- GA top-level ----------
    def __init__(
        self,
        heuristic: Optional["Heuristic"] = None,
        pop_size: int = 100,
        generations: int = 1000,
        initial_length: int = 10,
        max_length: int = 50,
        change_rate: float = 0.05,
        mix_length: int = 3,
        add_step: int = 10,
        elitism_rate: float = 0.5,  # fraction to keep (script kept top50)
    ) -> None:
        super().__init__()
        self.heuristic = heuristic or ManhattanHeuristic()
        self.pop_size = max(4, int(pop_size))
        self.generations = int(generations)
        self.initial_length = int(initial_length)
        self.max_length = int(max_length)
        self.change_rate = float(change_rate)
        self.mix_length = int(mix_length)
        self.add_step = int(add_step)
        self.elitism_rate = float(elitism_rate)
        self.rng = np.random.default_rng()
        self.population: List[Genetic.Chromosome] = []
        self.best_ever: Optional[Genetic.Chromosome] = None

    # ---------- helpers ----------
    def _make_initial_population(self, game: "SokobanGame"):
        self.population = []
        for _ in range(self.pop_size):
            c = Genetic.Chromosome(self.rng, max_length=self.max_length)
            c.init_random(self.initial_length)
            c.evaluate(game)
            self.population.append(c)
        # sort ascending by points (lower points better)
        self.population.sort(key=lambda x: x.points)

    def _mutate_population(self, survivors: List[Chromosome]):
        # variation: mutate in place (like script)
        for chrom in survivors:
            if chrom.points != 0:
                seq = chrom._sequence
                i = 0
                # iterate carefully as we may insert/delete
                while i < len(seq):
                    r = self.rng.random()
                    if r < self.change_rate:
                        # replace gene
                        seq[i] = Genetic.INT_TO_MOVE[int(self.rng.integers(0, 4))]
                        i += 1
                    elif r < 2 * self.change_rate and len(seq) < self.max_length:
                        # insert random gene
                        seq.insert(i, Genetic.INT_TO_MOVE[int(self.rng.integers(0, 4))])
                        i += 1
                    else:
                        i += 1
                # occasionally append random moves if too short (script had add_step)
                if len(seq) < 2:
                    for _ in range(min(self.add_step, self.max_length - len(seq))):
                        seq.append(Genetic.INT_TO_MOVE[int(self.rng.integers(0, 4))])

    def _hybridize(self, parent_a: Chromosome, parent_b: Chromosome) -> Chromosome:
        a = [Genetic.MOVE_TO_INT[m] for m in parent_a._sequence]
        b = [Genetic.MOVE_TO_INT[m] for m in parent_b._sequence]
        child_ints: List[int] = []
        i = 0
        # iterate in chunks of mix_length as script
        while i < len(a) - self.mix_length and i < len(b) - self.mix_length:
            flag = int(self.rng.integers(0, 2))
            if flag == 0:
                for j in range(self.mix_length):
                    child_ints.append(a[i + j])
            else:
                for j in range(self.mix_length):
                    child_ints.append(b[i + j])
            i += self.mix_length
        # if child too short, append random steps
        while len(child_ints) < min(self.max_length, max(len(a), len(b))):
            child_ints.append(int(self.rng.integers(0, 4)))
            if len(child_ints) >= self.max_length:
                break
        # if still short and script appends add_step, append that many
        if len(child_ints) < self.max_length:
            for _ in range(min(self.add_step, self.max_length - len(child_ints))):
                child_ints.append(int(self.rng.integers(0, 4)))
        # build sequence as moves
        child_moves = [Genetic.INT_TO_MOVE[i] for i in child_ints][: self.max_length]
        child = Genetic.Chromosome(self.rng, seq=child_moves, max_length=self.max_length)
        return child

    # ---------- main search ----------
    def _search(self, game: "SokobanGame", limits: SolverLimits) -> tuple[bool, List[Move]]:
        # build initial population
        self._make_initial_population(game)
        # trackers
        best_overall = min(self.population, key=lambda c: c.points)
        self.best_ever = best_overall.copy()

        best_points_history: List[float] = []
        stagnation_count = 0

        # stagnation parameters (tunable)
        recent_window = 40
        old_window = 40
        stagnation_threshold = 1.0  # change in points smaller than this considered "no improvement"
        report_every = 10

        with tqdm(total=self.generations, desc="GA Progress", unit="gen") as pbar:
            for gen in range(self.generations):
                # record complexity
                self.tracker.record_node()
                self.tracker.track_frontier(len(self.population))

                # sort by points ascending (lower better)
                self.population.sort(key=lambda x: x.points)
                top = self.population[0]

                # update history and stagnation detection
                best_points_history.append(top.points)
                if len(best_points_history) > (recent_window + old_window):
                    recent_best = min(best_points_history[-recent_window:])
                    old_best = min(best_points_history[-(recent_window + old_window):-recent_window])
                    # since lower points is better, improvement = old_best - recent_best
                    improvement = old_best - recent_best
                    if improvement < stagnation_threshold:
                        stagnation_count += 1
                    else:
                        stagnation_count = 0

                # prepare postfix values
                fit_display = f"{-top.points:.0f}"   # show "fitness" (higher better) as -points for visual consistency
                mv_display = len(top._sequence)
                stg_display = stagnation_count
                mut_display = f"{self.change_rate:.3f}"

                pbar.set_postfix({"fit": fit_display, "mv": mv_display, "stg": stg_display, "mut": mut_display})
                pbar.update(1)

                # print top5 occasionally (same style as you showed)
                if gen % report_every == 0 or gen < 20:
                    top5 = [int(-c.fitness) if hasattr(c, "fitness") and c.fitness != -float("inf") else int(c.points)
                            for c in self.population[:5]]
                    # normalize display: we want to show fitness-like numbers; our fitness = -points
                    display_top5 = [str(t) for t in top5]
                    tqdm.write(f"[Gen {gen}] top5: {', '.join(display_top5)}")

                # check solved (points == 0)
                if top.points == 0:
                    prefix = top._sequence[: top.valid_prefix_len]
                    tqdm.write(f"[SUCCESS] Exact solution found at gen {gen} (len={len(prefix)})")
                    return True, prefix

                # selection + variation + reproduction step (existing logic)
                retain_n = max(2, int(self.pop_size * self.elitism_rate))
                survivors = [c.copy() for c in self.population[:retain_n]]

                # mutate survivors (variation)
                self._mutate_population(survivors)
                for s in survivors:
                    s.evaluate(game)

                # produce new population
                new_pop: List[Genetic.Chromosome] = []
                new_pop.extend(survivors)
                while len(new_pop) < self.pop_size:
                    idxs = self.rng.choice(len(survivors), size=2, replace=False)
                    a = survivors[int(idxs[0])]
                    b = survivors[int(idxs[1])]
                    child = self._hybridize(a, b)
                    # occasional small random append
                    if self.rng.random() < 0.02 and len(child._sequence) < self.max_length:
                        for _ in range(min(self.add_step, self.max_length - len(child._sequence))):
                            child._sequence.append(Genetic.INT_TO_MOVE[int(self.rng.integers(0, 4))])
                    child.evaluate(game)
                    new_pop.append(child)

                self.population = new_pop

                # update best ever
                cur_best = min(self.population, key=lambda c: c.points)
                if cur_best.points < self.best_ever.points:
                    self.best_ever = cur_best.copy()

                # optional: if stagnation_count large, increase mutation/introduce immigrants
                if stagnation_count and stagnation_count % 50 == 0:
                    tqdm.write(f"[STUCK] gen {gen}: stagnation_count={stagnation_count}, injecting random immigrants")
                    # inject a few random individuals into population
                    n_imm = max(1, int(self.pop_size * 0.05))
                    for i in range(n_imm):
                        im = Genetic.Chromosome(self.rng, max_length=self.max_length)
                        im.init_random(self.initial_length)
                        im.evaluate(game)
                        # replace from the end
                        self.population[-1 - i] = im
                    # reset some history to give fresh running room
                    best_points_history = best_points_history[-recent_window:]

            # end generations

        # final fallback -- return best-ever legal prefix
        if self.best_ever:
            prefix = self.best_ever._sequence[: self.best_ever.valid_prefix_len]
            tqdm.write("[INFO] No exact solution found; returning best-ever prefix (approx).")
            return False, prefix
        else:
            best = min(self.population, key=lambda c: c.points)
            return False, best._sequence[: best.valid_prefix_len]


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
    game_file = r"D:\1. REFERENCES\1. AI - Machine Learning\10. GenAI\GAs\BTL-NMAI-251\test\game\pico_6.txt"
    heuristic = ManhattanHeuristic()
    solver = Genetic(heuristic=heuristic, pop_size=200, generations=5000, initial_length=20, max_length=100, change_rate=0.05, mix_length=3, add_step=10, elitism_rate=0.5)
    visualizer = ConsoleVisualizer(delay_s=0.1)
    limits = None
    
    runner = Runner(game_file=game_file,
                    solver=solver,
                    visulizer=visualizer,
                    limits=limits)
    runner.run_solver()

    