# Monte Carlo Value Network Chess Agent — Implementation Plan (Markdown Spec)

This document defines the full implementation plan to upgrade your existing chess RL codebase into a **Monte Carlo value-learning agent with 1-ply lookahead**, capable of defeating a random agent with >60% win rate on limited computation (Colab/Kaggle).

It replaces the old DQN-style agent with a simpler and more stable training loop based on final game outcomes (Monte Carlo returns). The following files will be modified or extended:

* `agent.py`
* `chess_env_v2.py` (unchanged)
* `random_agent.py` (unchanged)
* `rl_agent_big.py` (new agent + value network)
* `train_local.ipynb` (new training loop)

---

# 1. Overview of New Architecture

We introduce a new agent that uses:

### **1. Value Network `V(s)`**

A CNN that takes a state tensor `[13, 8, 8]` and outputs a scalar in `[-1, 1]` representing win probability from the side-to-move’s perspective.

### **2. 1-Ply Lookahead Move Selection**

For each legal move:

1. Apply the move → next state `s'`
2. Compute `V(s')`
3. Select the move with highest predicted value (`argmax V`), with optional ε-greedy exploration.

### **3. Training via Monte Carlo Returns**

For each self-play game vs the RandomAgent:

* At the end of game, compute outcome:

  * Win → `z = +1`
  * Draw or max-step → `z = 0`
  * Loss → `z = -1`
* For **every state where our agent moved**, store `(state_tensor_13, z)` into the replay buffer.
* Train the value network using **MSE(V(s), z)**.

This avoids Q-learning instability and drastically reduces complexity.

---

# 2. Board Encoding (New Function)

Add a new helper in `rl_agent_big.py`:

### **`board_to_tensor_13(board)`**

Returns a `float32` numpy array of shape `[13, 8, 8]`.

* Channels 0–5: white pieces (P, N, B, R, Q, K)
* Channels 6–11: black pieces (P, N, B, R, Q, K)
* Channel 12: side-to-move plane

  * all ones if White to move
  * all zeros if Black to move

This function replaces usage of the old `_board_to_tensor()` from `RLAgent`.

---

# 3. Value Network (New Model)

Add this to `rl_agent_big.py`, below the existing models:

```python
class ValueNet(nn.Module):
    def __init__(self, in_channels=13, channels=64):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        v = self.value_head(x)
        return v.squeeze(-1)
```

---

# 4. New Agent Class: MonteCarloValueAgent

Add this in `rl_agent_big.py`:

### **Class Responsibilities**

* Manage replay buffer
* Compute epsilon schedule
* Choose moves via 1-ply lookahead
* Run gradient updates on the value network
* Save/load model

---

### **4.1. Constructor**

```python
class MonteCarloValueAgent(Agent):
    def __init__(self,
                 epsilon_start=0.3,
                 epsilon_end=0.05,
                 epsilon_decay_episodes=1500,
                 lr=1e-3,
                 replay_size=50000,
                 batch_size=256):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ValueNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.batch_size = batch_size

        self.replay_states  = []
        self.replay_targets = []
        self.replay_size    = replay_size
```

### **4.2. Epsilon schedule**

```python
    def epsilon_for_episode(self, episode):
        t = min(1.0, episode / self.epsilon_decay_episodes)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * t
```

---

### **4.3. Move Selection (1-Ply Lookahead)**

```python
    def choose_move(self, board, epsilon=0.1):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        if random.random() < epsilon:
            return random.choice(legal_moves)

        next_states = []
        for move in legal_moves:
            b = board.copy()
            b.push(move)
            next_states.append(board_to_tensor_13(b))

        with torch.no_grad():
            batch = torch.FloatTensor(np.stack(next_states)).to(self.device)
            values = self.model(batch).cpu().numpy()

        best_idx = int(np.argmax(values))
        return legal_moves[best_idx]
```

---

### **4.4. Replay Buffer Add**

```python
    def add_episode_samples(self, states_list, z):
        for s in states_list:
            self.replay_states.append(s)
            self.replay_targets.append(z)

        if len(self.replay_states) > self.replay_size:
            self.replay_states  = self.replay_states[-self.replay_size:]
            self.replay_targets = self.replay_targets[-self.replay_size:]
```

---

### **4.5. Train Step**

```python
    def train_step(self):
        if len(self.replay_states) < self.batch_size:
            return None

        idxs = np.random.choice(len(self.replay_states), self.batch_size, replace=False)
        batch_states  = np.stack([self.replay_states[i]  for i in idxs])
        batch_targets = np.array([self.replay_targets[i] for i in idxs], dtype=np.float32)

        states_tensor  = torch.FloatTensor(batch_states).to(self.device)
        targets_tensor = torch.FloatTensor(batch_targets).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(states_tensor)
        loss = self.loss_fn(preds, targets_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
```

---

### **4.6. Save/Load**

```python
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device).eval()
```

---

# 5. Game Runner Function (Self-Play vs RandomAgent)

Add this in `rl_agent_big.py` or a new utils file:

```python
def play_game_value_vs_random(value_agent,
                              random_agent,
                              max_moves=100,
                              epsilon=0.1,
                              agent_plays_white=True):

    board = chess.Board()
    states_list = []
    moves = 0
    agent_color = chess.WHITE if agent_plays_white else chess.BLACK

    while not board.is_game_over(claim_draw=True) and moves < max_moves:
        if board.turn == agent_color:
            states_list.append(board_to_tensor_13(board))
            move = value_agent.choose_move(board, epsilon=epsilon)
        else:
            move = random_agent.get_action(board)

        if move is None:
            break
        board.push(move)
        moves += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        z = 0.0
    else:
        if outcome.winner is None:
            z = 0.0
        elif outcome.winner == agent_color:
            z = 1.0
        else:
            z = -1.0

    if moves >= max_moves and outcome is None:
        z = 0.0

    return states_list, z
```

---

# 6. Modify `train_local.ipynb` — New Training Loop

### **6.1. Import and instantiate**

```python
from rl_agent_big import MonteCarloValueAgent, play_game_value_vs_random
from random_agent import RandomAgent

value_agent = MonteCarloValueAgent()
random_agent = RandomAgent()
```

---

### **6.2. Main Training Loop**

```python
num_episodes = 3000
eval_every = 200
eval_games = 200
win_history = []

for episode in range(1, num_episodes+1):
    epsilon = value_agent.epsilon_for_episode(episode)
    agent_white = (episode % 2 == 1)

    states, z = play_game_value_vs_random(
        value_agent,
        random_agent,
        max_moves=100,
        epsilon=epsilon,
        agent_plays_white=agent_white
    )

    if states:
        value_agent.add_episode_samples(states, z)

    loss = value_agent.train_step()

    if episode % eval_every == 0:
        wins = draws = losses = 0
        for _ in range(eval_games):
            s_eval, z_eval = play_game_value_vs_random(
                value_agent,
                random_agent,
                max_moves=100,
                epsilon=0.0,
                agent_plays_white=True
            )
            if z_eval > 0: wins += 1
            elif z_eval < 0: losses += 1
            else: draws += 1

        win_rate = wins / eval_games
        win_history.append(win_rate)
        print(f"Episode {episode} — WinRate={win_rate:.3f}  W/D/L={wins}/{draws}/{losses}")

        if win_rate >= 0.60:
            value_agent.save("value_agent_60pct.pth")
```

---

# 7. Optional: Plot Learning Curve

```python
plt.plot(np.arange(len(win_history)) * eval_every, win_history)
plt.xlabel("Episode")
plt.ylabel("Win Rate vs Random")
plt.grid(True)
plt.show()
```

---

# 8. Notes & Recommendations

* Start with small networks (64 channels) for speed.
* Use `max_moves=100` to avoid long random games.
* Value-based Monte Carlo learning converges faster than Q-learning for this environment.
* Expect >60% win rate in the 1500–3000 episode range on GPU.

---

# 9. Summary

This spec defines:

1. New board encoding (13×8×8)
2. ValueNet model
3. MonteCarloValueAgent class with replay buffer
4. Game-runner for self-play vs RandomAgent
5. Entire new training loop for `train_local.ipynb`
6. Evaluation and model saving strategy

It is fully ready to be implemented by an automated code assistant (e.g., Codex).
