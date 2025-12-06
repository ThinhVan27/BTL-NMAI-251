import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch
from torch.amp import autocast, GradScaler
from agent import Agent

def board_to_tensor_13(board: chess.Board) -> np.ndarray:
    state = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            state[channel, rank, file] = 1.0

    # Side to move plane
    state[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return state

def evaluate_material(board: chess.Board, color: chess.Color) -> float:
    """Return material advantage (our - opponent) using simple piece values."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    our_score = 0
    opp_score = 0

    for piece_type, value in piece_values.items():
        our_score += len(board.pieces(piece_type, color)) * value
        opp_score += len(board.pieces(piece_type, not color)) * value

    return float(our_score - opp_score)

def normalized_material_value(board: chess.Board, color: chess.Color) -> float:
    max_material = 39.0
    raw = evaluate_material(board, color)
    return max(min(raw / max_material, 1.0), -1.0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return torch.relu(x)

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.hidden_channels = 64

        # conv trunk
        self.conv1 = nn.Conv2d(input_shape[0], self.hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(self.hidden_channels) for _ in range(3)]
        )

        flat_size = self.hidden_channels * 8 * 8

        # value stream
        self.value_fc = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

        # advantage stream
        self.advantage_fc = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions),
        )

    def forward(self, x):
        # shared trunk
        x = torch.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)

        value = self.value_fc(x)           # (B,1)
        advantage = self.advantage_fc(x)   # (B,A)

        # dueling aggregation
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def evaluate_state(self, x):
        """
        Forward pass that returns only V(s), reusing the same trunk
        as forward(). Used for evaluation pretraining and greedy-eval tests.
        """
        x = torch.relu(self.bn1(self.conv1(x)))
        for block in self.res_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        value = self.value_fc(x)  # (B,1)
        return value


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

class RLAgent(Agent):
    def __init__(self, state_shape=(12, 8, 8), action_size=4096):
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DuelingDQN(state_shape, action_size).to(self.device)
        self.target_net = DuelingDQN(state_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Lower LR for stability with ResNet
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100000) # Increased memory
        
        self.batch_size = 512 # Increased batch size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        # Faster decay initially to exploit learned behavior sooner
        self.epsilon_decay = 0.9995 
        self.scaler = torch.GradScaler()
        
    def get_action(self, game_state, legal_moves_indices=None):
        is_inference = False
        board_for_inference = None

        if isinstance(game_state, chess.Board):
            is_inference = True
            board_for_inference = game_state.copy()
            state_tensor = self._board_to_tensor(game_state)
            legal_moves_indices = self._get_legal_actions(game_state)
        else:
            state_tensor = game_state
        
        if not is_inference and random.random() < self.epsilon:
            if legal_moves_indices:
                return random.choice(legal_moves_indices)
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            if legal_moves_indices:
                # Mask illegal moves with negative infinity
                mask = torch.full((1, self.action_size), -float('inf')).to(self.device)
                mask[0, legal_moves_indices] = 0
                q_values += mask
                
            action_idx = q_values.argmax().item()
        
        if is_inference:
            return self._decode_action(action_idx, board_for_inference)
            
        return action_idx

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
          # Double DQN Logic
          with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            expected_q_values = reward + (1 - done) * self.gamma * next_q_values
          q_values = self.policy_net(state).gather(1, action)
          loss = nn.MSELoss()(q_values, expected_q_values)
        
        # self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer) # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path, training=False):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = 0.25 if training else 0.0
        if training: self.policy_net.train()
        else: self.policy_net.eval()

    def _board_to_tensor(self, board):
        state = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                channel = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6
                state[channel, rank, file] = 1
        return state

    def _get_legal_actions(self, board):
        legal_moves = []
        for move in board.legal_moves:
            legal_moves.append(move.from_square * 64 + move.to_square)
        return legal_moves

    def _decode_action(self, action_idx: int, board: chess.Board = None) -> chess.Move:
        from_square = action_idx // 64
        to_square = action_idx % 64
        move = chess.Move(from_square, to_square)
        
        # Logic to auto-promote during inference/testing
        if board:
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(to_square)
                if (piece.color == chess.WHITE and rank == 7) or \
                   (piece.color == chess.BLACK and rank == 0):
                    move.promotion = chess.QUEEN
        
        return move

class MonteCarloValueAgent(Agent):
    def __init__(self,
                 epsilon_start=0.2,
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
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.batch_size = batch_size

        self.replay_states = []
        self.replay_targets = []
        self.replay_size = replay_size

    def epsilon_for_episode(self, episode):
        t = min(1.0, episode / self.epsilon_decay_episodes)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * t

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
            self.model.eval()
            values = self.model(batch).cpu().numpy()

        best_idx = int(np.argmax(values))
        return legal_moves[best_idx]

    def add_episode_samples(self, states_list, z):
        for s in states_list:
            self.replay_states.append(s)
            self.replay_targets.append(z)

        if len(self.replay_states) > self.replay_size:
            self.replay_states = self.replay_states[-self.replay_size:]
            self.replay_targets = self.replay_targets[-self.replay_size:]

    def train_step(self):
        if len(self.replay_states) < self.batch_size:
            return None

        idxs = np.random.choice(len(self.replay_states), self.batch_size, replace=False)
        batch_states = np.stack([self.replay_states[i] for i in idxs])
        batch_targets = np.array([self.replay_targets[i] for i in idxs], dtype=np.float32)

        states_tensor = torch.FloatTensor(batch_states).to(self.device)
        targets_tensor = torch.FloatTensor(batch_targets).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(states_tensor)
        loss = self.loss_fn(preds, targets_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device).eval()

def play_game_value_vs_random(value_agent,
                              random_agent,
                              max_moves=150,
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
        z_result = 0.0
    else:
        if outcome.winner is None:
            z_result = 0.0
        elif outcome.winner == agent_color:
            z_result = 1.0
        else:
            z_result = -1.0

    if moves >= max_moves and outcome is None:
        z_result = 0.0

    z_material = normalized_material_value(board, agent_color)
    alpha = 0.7
    z = alpha * z_material + (1.0 - alpha) * z_result

    return states_list, z
