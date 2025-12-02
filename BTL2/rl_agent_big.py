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
        
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(input_shape[0], self.hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_channels) for _ in range(6)
        ])
        
        flat_size = self.hidden_channels * 8 * 8
        
        # Value Stream (Evaluates the board state)
        self.value_fc = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
        # Advantage Stream (Evaluates each specific action)
        self.advantage_fc = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)
        )
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))

        for block in self.res_blocks:
          x = block(x)
        
        x = x.view(x.size(0), -1)
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Dueling Network Aggregation
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

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