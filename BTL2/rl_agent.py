import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
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
        
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Towers (AlphaZero style, but smaller)
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        
        flat_size = 64 * 8 * 8
        
        # Value Stream (Evaluates the board state)
        self.value_fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage Stream (Evaluates each specific action)
        self.advantage_fc = nn.Sequential(
            nn.Linear(flat_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions)
        )
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
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
        self.memory = deque(maxlen=20000) # Increased memory
        
        self.batch_size = 64 # Increased batch size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        # Faster decay initially to exploit learned behavior sooner
        self.epsilon_decay = 0.995 
        
    def get_action(self, game_state, legal_moves_indices=None):
        is_inference = False
        if isinstance(game_state, chess.Board):
            is_inference = True
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
            return self._decode_action(action_idx)
            
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
        
        # Double DQN Logic
        # 1. Select best action using Policy Net
        next_actions = self.policy_net(next_state).argmax(1, keepdim=True)
        # 2. Evaluate that action using Target Net
        next_q_values = self.target_net(next_state).gather(1, next_actions)
        
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values
        q_values = self.policy_net(state).gather(1, action)
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        

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
        self.epsilon = 0.05 if training else 0.0
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

    def _decode_action(self, action_idx: int) -> chess.Move:
        return chess.Move(action_idx // 64, action_idx % 64)