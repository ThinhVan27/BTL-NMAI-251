# Chess Agent – ML‑Guided Search (Policy + Minimax) Plan

This document describes how to build the **final assignment agent** using a combination of:

* A **neural policy network** (ML component), and
* A **classical minimax / negamax search** (existing strong engine),

such that the final agent reliably beats a **random agent ≥ 60%** of the time.

Codex should use this as a **design + implementation checklist** to integrate the hybrid agent into the existing codebase.

---

## 1. High‑Level Idea

We already have:

* A **minimax agent** that beats random ≈ 90% → strong teacher.
* A **policy network** trained via imitation learning from minimax (behavior cloning), but as a standalone policy it only achieves ≈ 23% winrate vs random.

Instead of using the neural policy alone, we will:

> Use the **policy net as a move prior** to **guide** minimax search: at each node, the policy ranks legal moves, and the search only explores the **top‑k** moves.

This is similar in spirit to AlphaZero:

* **Neural network** → suggests which moves are promising.
* **Tree search (minimax/negamax)** → does deeper lookahead using a static evaluation function.

This hybrid agent remains **strong** (close to the original minimax) but:

* Uses **ML** in a critical way (move ordering and pruning),
* Is explainable and easy to present in the report.

---

## 2. Existing Components to Reuse

### 2.1. Environment / Representation

From `chess_env_v2.py`:

* `ChessEnv` with:

  * `board` (chess.Board)
  * `get_state()` → `(13, 8, 8)` tensor
  * `encode_action(move)` → `int` in `[0, 4095]`
  * `decode_action(idx)` → `chess.Move`
  * `get_legal_actions()` → list of legal action indices

### 2.2. Agents

From existing modules:

* `RandomAgent` (random_agent.py):

  * Picks a random legal move from the current board.

* `MinimaxAgent` (or equivalent):

  * Exposes something like: `get_best_move(board: chess.Board) -> chess.Move`
  * Uses **classical minimax/negamax + static evaluation**.
  * This is currently ≈ 90% winrate vs random.

Codex must ensure there is a **clean, reusable MinimaxAgent** class in a file like `minimax_agent.py`.

### 2.3. Policy Network (Student)

From the distillation work:

* `PolicyNet` (e.g. in `policy_net.py`):

  * Input: `(13, 8, 8)` board state.
  * Output: `(4096,)` logits over all possible moves.
  * Trained via cross-entropy on `(state, teacher_action_idx)` pairs from minimax vs random games.

Weights are stored as `models/policy_supervised.pth`.

Codex should be able to:

```python
policy = PolicyNet(input_shape=(13, 8, 8), num_actions=4096)
policy.load_state_dict(torch.load("models/policy_supervised.pth", map_location=device))
policy.to(device).eval()
```

---

## 3. New Component: ML‑Guided Minimax Agent

Create a new agent class, e.g. in `ml_guided_minimax_agent.py`:

```python
from agent import Agent
from chess_env_v2 import ChessEnv
from policy_net import PolicyNet
from minimax_agent import MinimaxEvaluator  # factor out evaluation if needed

class MLGuidedMinimaxAgent(Agent):
    def __init__(self, policy, depth=3, top_k=6, device="cuda"):
        ...

    def get_action(self, game_state: chess.Board) -> chess.Move:
        ...
```

### 3.1. Design

**Core idea:**

1. At the root (and optionally at deeper nodes), use the **policy net** to score all legal moves.
2. Select the **top‑k** moves according to these scores.
3. Run **minimax / negamax search** only over these top‑k moves up to a fixed depth `D`.
4. Return the move with the best minimax value.

This gives us:

* **Lower branching factor** due to policy pruning.
* **Stronger play** than pure policy because we still look ahead.
* **Practical speed** because `k` is small (e.g. 4–8), and depth can be modest (e.g. 2–3).

### 3.2. Policy Scoring Function

Implement a helper to convert a `chess.Board` into policy scores for moves:

```python
def policy_scores_for_board(policy, env: ChessEnv, board: chess.Board, device) -> dict[Move, float]:
    env.board = board
    state = env.get_state()
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = policy(state_t)[0]  # (4096,)

    legal_idxs = env.get_legal_actions()
    # mask illegal moves
    mask = torch.full((policy.num_actions,), float('-inf'), device=device)
    mask[legal_idxs] = 0.0
    masked_logits = logits + mask

    # convert back to per-move dictionary
    scores = {}
    for idx in legal_idxs:
        move = env.decode_action(idx)
        scores[move] = masked_logits[idx].item()
    return scores
```

### 3.3. Minimax / Negamax with Policy‑Guided Move Ordering

Refactor the existing minimax implementation so that:

* There is a **pure evaluator** for leaf nodes (e.g. `evaluate(board) -> float`, based on material + heuristics).
* The search itself can accept a move ordering strategy.

Then, in `MLGuidedMinimaxAgent`, implement something like:

```python
def _negamax(self, board, depth, alpha, beta, color_sign):
    if depth == 0 or board.is_game_over(claim_draw=True):
        return color_sign * self.evaluator.evaluate(board)

    # get policy scores and sort moves by descending score
    scores = policy_scores_for_board(self.policy, self.env, board, self.device)
    ordered_moves = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)

    # restrict to top-k moves
    ordered_moves = ordered_moves[: self.top_k]

    best_value = -1e9
    for move in ordered_moves:
        board.push(move)
        val = -self._negamax(board, depth-1, -beta, -alpha, -color_sign)
        board.pop()

        if val > best_value:
            best_value = val
        if best_value > alpha:
            alpha = best_value
        if alpha >= beta:
            break

    return best_value
```

At the root in `get_action`:

```python
def get_action(self, game_state: chess.Board) -> chess.Move:
    board = game_state.copy()
    color_sign = 1 if board.turn == chess.WHITE else -1

    # compute policy scores once at root
    scores = policy_scores_for_board(self.policy, self.env, board, self.device)
    ordered_moves = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
    ordered_moves = ordered_moves[: self.top_k]

    best_move = None
    best_value = -1e9
    alpha, beta = -1e9, 1e9

    for move in ordered_moves:
        board.push(move)
        val = -self._negamax(board, self.depth-1, -beta, -alpha, -color_sign)
        board.pop()

        if val > best_value:
            best_value = val
            best_move = move
        if val > alpha:
            alpha = val

    # fallback in degenerate cases
    if best_move is None:
        legal_moves = list(game_state.legal_moves)
        if not legal_moves:
            return None
        best_move = random.choice(legal_moves)

    return best_move
```

**Key points:**

* `self.evaluator.evaluate(board)` can use the existing classical evaluation from the current minimax implementation.
* ML is used to **order and prune moves**; evaluation remains deterministic and fast.

---

## 4. Evaluation Script: ML‑Guided vs Random

Create `eval_ml_guided_vs_random.py` to compare:

1. `RandomAgent` vs `RandomAgent` (sanity) → ~50%.
2. `MinimaxAgent` vs `RandomAgent` → baseline (≈ 0.9 winrate).
3. `MLGuidedMinimaxAgent` vs `RandomAgent` → **target metric**.

Pseudocode:

```python
def play_match(agent_white, agent_black, max_moves=200) -> int:
    board = chess.Board()
    env = ChessEnv()
    env.board = board

    moves = 0
    while not board.is_game_over(claim_draw=True) and moves < max_moves:
        if board.turn == chess.WHITE:
            move = agent_white.get_action(board)
        else:
            move = agent_black.get_action(board)
        if move is None or move not in board.legal_moves:
            break
        board.push(move)
        moves += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0   # draw
    return 1 if outcome.winner == chess.WHITE else -1
```

Then:

```python
def eval_agent_vs_random(agent_factory, n_games=100, max_moves=200):
    wins = draws = losses = 0
    for i in range(n_games):
        random_agent = RandomAgent()
        if i % 2 == 0:
            # agent as White
            agent = agent_factory()
            res = play_match(agent, random_agent, max_moves)
        else:
            # agent as Black
            agent = agent_factory()
            res = play_match(random_agent, agent, max_moves)
            res *= -1  # flip perspective so +1 means agent win

        if res > 0:
            wins += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1

    winrate = wins / n_games
    print(f"[RESULT] Agent vs random over {n_games} games: winrate={winrate:.3f} (W/D/L={wins}/{draws}/{losses})")
    return winrate
```

Usage for ML‑guided agent:

```python
def make_ml_guided_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet(input_shape=(13, 8, 8), num_actions=4096)
    policy.load_state_dict(torch.load("models/policy_supervised.pth", map_location=device))
    policy.to(device).eval()
    return MLGuidedMinimaxAgent(policy=policy, depth=3, top_k=6, device=device)

winrate = eval_agent_vs_random(make_ml_guided_agent, n_games=100)
```

The assignment target is **winrate ≥ 0.60**.

---

## 5. Reporting / Write‑up Guidance

In the final report, the method can be described as:

1. **Teacher agent:**

   * Deterministic minimax search with classical heuristic evaluation, achieving ≈ 90% winrate vs random.

2. **Student policy network (ML):**

   * CNN with residual blocks, input `(13, 8, 8)` representation, output 4096 actions.
   * Trained via supervised learning (behavior cloning) on minimax self-play positions.

3. **Hybrid agent – ML‑Guided Minimax:**

   * At each decision point, use the neural policy to produce a **move prior** over legal moves.
   * Restrict minimax search to the **top‑k** moves according to the network.
   * Run depth‑`D` negamax with alpha‑beta pruning using the classical evaluator.

4. **Results:**

   * Show comparison of:

     * Random vs random (~50% winrate baseline).
     * Minimax vs random (high, ~90%).
     * ML‑Guided Minimax vs random (expected ≥ 60%).
   * Emphasize that the **ML component is crucial** for search efficiency and move selection.

This clearly satisfies the requirement: *“use an ML method to build an agent that beats a random agent ≥ 60% of the time”*.

---

## 6. Tasks for Codex (Checklist)

1. **Ensure `MinimaxAgent` is a clean module**

   * File: `minimax_agent.py`.
   * API: `get_best_move(board: chess.Board) -> chess.Move`.
   * Separate evaluation function if needed: `MinimaxEvaluator.evaluate(board) -> float`.

2. **Ensure `PolicyNet` is defined and loadable**

   * File: `policy_net.py`.
   * Verify compatibility with existing `teacher_minimax.npz` dataset and `policy_supervised.pth` checkpoint.

3. **Implement `policy_scores_for_board(...)`**

   * Uses `ChessEnv`, `get_state()`, and `PolicyNet` to produce a `dict[Move, score]` for legal moves.

4. **Implement `MLGuidedMinimaxAgent`**

   * File: `ml_guided_minimax_agent.py`.
   * Constructor arguments: `policy`, `depth`, `top_k`, `device`.
   * Methods:

     * `_negamax(board, depth, alpha, beta, color_sign)`
     * `get_action(game_state: chess.Board) -> chess.Move`.
   * Integrate policy-based move ordering and top‑k pruning.

5. **Implement evaluation script**

   * File: `eval_ml_guided_vs_random.py`.
   * Implement `play_match` and `eval_agent_vs_random` as described.
   * Evaluate `MLGuidedMinimaxAgent` vs `RandomAgent` for at least 100 games.

6. **(Optional) CLI / Main entry point**

   * Add a main script that can run different agents:

     * `RandomAgent`
     * `MinimaxAgent`
     * `PolicyNet`-only agent
     * `MLGuidedMinimaxAgent`
   * Useful for demos and quick experiments.

With this plan, Codex can implement a robust **ML‑guided search agent** that leverages the existing minimax engine and the distilled policy network, and is suitable for the assignment’s winrate requirement and ML focus.
