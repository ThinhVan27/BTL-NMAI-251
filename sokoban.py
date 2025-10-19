"""High-level Sokoban skeleton with game rules, solver stubs, and visualization hooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from abc import ABC, abstractmethod
import time
import tracemalloc
import pyautogui
import collections
import itertools
import heapq
import os
import argparse

counter = itertools.count()
GAME_ROOT = Path("test/game")
SOLUTION_ROOT = Path("test/solution")

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
    time_limit_s: float = None
    node_limit: int = None


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
        # limits = limits or SolverLimits()
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

        initial_state = game.initial_state

        # Hàng đợi (queue) chứa các tuple (trạng thái, danh_sách_nước_đi)
        queue = collections.deque([(initial_state, [])])

        # Tập hợp các trạng thái đã duyệt để tránh lặp
        visited_states = {initial_state}

        self.tracker.track_frontier(len(queue))

        while queue:
            # Lấy trạng thái hiện tại và đường đi ra khỏi hàng đợi
            current_state, current_moves = queue.popleft()
            self.tracker.record_node() # Ghi nhận đã mở rộng 1 node

            # Kiểm tra giới hạn thời gian và số node
            if limits and (time.perf_counter() - self.tracker._start_time) > limits.time_limit_s:
                print("[INFO] Time limit exceeded")
                return False, []
            if limits and self.tracker.nodes_expanded > limits.node_limit:
                print("[INFO] Node limit exceeded")
                return False, []

            # 2. Kiểm tra xem có phải trạng thái đích không
            if game.is_goal(current_state):
                return True, current_moves  # Trả về kết quả và chuỗi hành động

            # 3. Lấy các trạng thái kế tiếp hợp lệ
            for move, next_state in game.successors(current_state):
                # 4. Nếu trạng thái kế tiếp chưa được duyệt
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    new_moves = current_moves + [move]
                    queue.append((next_state, new_moves))

            # Cập nhật kích thước lớn nhất của hàng đợi
            self.tracker.track_frontier(len(queue))

        # Nếu hàng đợi rỗng mà không tìm thấy giải pháp
        return False, []


class AStarSolver(Solver):
    name = "astar"

    def __init__(self, heuristic: Optional["Heuristic"] = None) -> None:
        super().__init__()
        self.heuristic = heuristic
        
    def _init_cost_functions(self, game: SokobanGame) -> None:
        
        init_state = game.initial_state
        
        self.f = {} # Cost function = g(n) + h(n)
        self.g = {} # Cost funxtion from init state to n
        self.h = {} # Heuristic function from n to goal
        
        self.g[init_state] = 0.0
        self.h[init_state] = self.heuristic.estimate(game.initial_state, game)
        self.f[init_state] = self.g[init_state] + self.h[init_state]
    
    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        self._init_cost_functions(game)
        
        openSet = set()
        openHeap = []
        init_state = game.initial_state
        openSet.add(init_state)
        heapq.heappush(openHeap, (self.f.get(init_state, float('inf')), next(counter), init_state))
        
        cameFrom = dict()
        self.tracker.record_node()
        while len(openSet) > 0:
            self.tracker.track_frontier(len(openSet))
            if limits and (time.perf_counter() - self.tracker._start_time) > limits.time_limit_s:
                print("[INFO] Time limit exceeded")
                return False, []
            if limits and self.tracker.nodes_expanded > limits.node_limit:
                print("[INFO] Node limit exceeded")
                return False, []
            _, _, current_state = heapq.heappop(openHeap)
            
            if game.is_goal(current_state):
                return (True, self._reconstruct_path(cameFrom, current_state))
            
            openSet.remove(current_state)
            
            for move, succ_state in game.successors(current_state):
                path_cost = 1.0
                
                temp_gScore_succ = self.g.get(current_state, float('inf')) + path_cost
                if temp_gScore_succ < self.g.get(succ_state, float('inf')):                    
                    cameFrom[succ_state] = (current_state, move)
                    self.g[succ_state] = temp_gScore_succ
                    self.f[succ_state] = temp_gScore_succ + self.heuristic.estimate(succ_state, game)
                
                    if succ_state not in openSet:                        
                        self.tracker.record_node()
                        openSet.add(succ_state)
                        heapq.heappush(openHeap, (self.f.get(succ_state, float('inf')), next(counter), succ_state))
        
        return (False, [])
    
    def _reconstruct_path(self, cameFrom: Dict[str, str], current_state: str):
        moves = []
        while current_state in cameFrom.keys():
            previous_state, move = cameFrom[current_state]
            moves.append(move)
            current_state = previous_state
        
        return list(reversed(moves))
        

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
        
        heuristic_dist_of_boxes2goals = 0.0
        for box in boxes:
            heuristic_dist_of_boxes2goals += min([self._mahattan_dist(box, goal) for goal in goals])
        h = min_dist_player2box + heuristic_dist_of_boxes2goals
        
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

class Runner:
    def __init__(self, game_file: str, 
                 solver: Solver, 
                 visulizer: SolutionVisualizer, 
                 limits: Optional[SolverLimits]):
        self.game_file = game_file
        self.solver = solver
        self.visualizer = visulizer
        self.limits = limits
        self.game = load_game_from_path(GAME_ROOT/Path(self.game_file))
        
    def run_solver(self, replace=True) -> None:
        if replace or not os.path.exists(SOLUTION_ROOT/Path(self.game_file)):
            result = self.solver.solve(self.game, self.limits)
            self._save_solution(result)
        else:
            result = self._load_solution()
        self._visulize(result)
    
    def _save_solution(self, result: SolverResult) -> None:
        game_path = SOLUTION_ROOT/Path(self.game_file)
        moves = ", ".join(result.move_sequence)
        with open(game_path, 'w', encoding='utf-8') as f:
            f.write(moves)
    
    def _load_solution(self) -> SolverResult:
        solution_path = SOLUTION_ROOT/Path(self.game_file)
        with open(solution_path, 'r', encoding='utf-8') as f:
            moves = f.read()
        
        return SolverResult(solved=True,
                            move_sequence=moves.split(", "),
                            elapsed_time_s=-1,
                            nodes_expanded=-1,
                            max_frontier=-1,
                            peak_memory_bytes=-1)

    def _visulize(self, result: SolverResult) -> None:
        time.sleep(3) # Wait for playing
        self.visualizer.display(self.game, result.move_sequence)
        print(f"[INFO] Time elapsed: {result.elapsed_time_s}")
        print(f"[INFO] Peak memory: {result.peak_memory_bytes}")
        print(f"[INFO] Nodes: {result.nodes_expanded}")
        print(f"[INFO] Max frontier: {result.max_frontier}")
        
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sokoban solver runner")
    parser.add_argument("-s", "--solver", choices=["astar", "bfs"], default="bfs",
                        help="Solver to use")
    parser.add_argument("-v", "--visualizer", choices=["console", "playright"], default="console",
                        help="Visualizer to use ('playright' maps to PlaywrightVisualizer)")
    parser.add_argument("-g", "--game-file", default="mini_1.txt", help="Relative game file under test/game")
    parser.add_argument("-r", "--replace", action="store_true", help="Run solver and save new solution if it has already existed")
    args = parser.parse_args()
    
    # Select solver
    if args.solver == "astar":
        heuristic = ManhattanHeuristic()
        solver = AStarSolver(heuristic)
    else:
        solver = BFSSolver()
    # Select visualizer (note: 'playright' per CLI maps to PlaywrightVisualizer)
    if args.visualizer == "console":
        visualizer = ConsoleVisualizer(delay_s=0.05)
    else:
        visualizer = PlaywrightVisualizer(delay_s=0.05)
    
    limits = None
    
    runner = Runner(game_file=args.game_file,
                    solver=solver,
                    visulizer=visualizer,
                    limits=limits)
    runner.run_solver(replace=args.replace)