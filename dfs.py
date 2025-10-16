from sokoban import *
from pathlib import Path



class DFSSolver(Solver):
    name = "dfs"

    def _search(self, game: SokobanGame, limits: SolverLimits) -> tuple[bool, List[Move]]:
        
        stack = [(game.initial_state, [])]  
        visited = set()  
        
        while stack:
            state, moves = stack.pop()  
            
            
            if game.is_goal(state):
                return True, moves  
            
            
            visited.add(state)
            
            
            for move in game.legal_moves(state):
                next_state = game.apply_move(state, move)
                
                
                if next_state is not None and next_state not in visited:
                    stack.append((next_state, moves + [move]))
        
        return False, []  



path = Path('C:/Users/ADMIN/Desktop/251/NMAI/BTL-NMAI-251/test/game/micro_6.txt')  


game = load_game_from_path(path)


solver = DFSSolver()
result = solver.solve(game)

print(f"Solved: {result.solved}")
print(f"Move sequence: {result.move_sequence}")
print(f"Elapsed time: {result.elapsed_time_s} seconds")
print(f"Nodes expanded: {result.nodes_expanded}")
print(f"Max frontier size: {result.max_frontier}")
print(f"Peak memory used: {result.peak_memory_bytes} bytes")
