from sokoban import *
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from pathlib import Path

GAME_ROOT = Path("test/game")
SOLUTION_ROOT = Path("test/solution")

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
        game_path = SOLUTION_ROOT/Path(self.game_file)
        moves = game_path.write_text(", ".join(moves))
    
    def _load_solution(self) -> List[str]:
        solution_path = SOLUTION_ROOT/Path(self.game_file[:-4]+"_solution.txt")
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
    game_file = "mini_25.txt"
    heuristic = ManhattanHeuristic()
    solver = AStarSolver(heuristic)
    visualizer = ConsoleVisualizer(delay_s=0.1)
    limits = None
    
    runner = Runner(game_file=game_file,
                    solver=solver,
                    visulizer=visualizer,
                    limits=limits)
    runner.run_solver()