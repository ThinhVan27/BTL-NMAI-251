import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from agent import Agent

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Increased filters
        
        flat_size = 128 * 8 * 8
        
        # Value Stream
        self.value_fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage Stream
        self.advantage_fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Combine: Q = V + (A - mean(A))
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
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999 # Slower decay (was 0.995)
        
    def get_action(self, game_state, legal_moves_indices=None):
        # game_state is expected to be the tensor representation if called internally,
        # but the base Agent.get_action expects a chess.Board.
        
        is_inference = False
        if isinstance(game_state, chess.Board):
            is_inference = True
            # Convert board to tensor
            state_tensor = self._board_to_tensor(game_state)
            # Also get legal moves to mask
            legal_moves_indices = self._get_legal_actions(game_state)
        else:
            state_tensor = game_state
        
        # For inference, we might want to disable exploration or use a very small epsilon
        # But let's stick to self.epsilon for now, or maybe 0.05 if inference?
        # The user passes epsilon in train.py, but here it uses self.epsilon which is 1.0 initially
        # but decays. If we load a model, we might want to set epsilon to low.
        # But let's just fix the return type first.
        
        action_idx = 0
        if random.random() < self.epsilon:
            if legal_moves_indices:
                action_idx = random.choice(legal_moves_indices)
            else:
                action_idx = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                if legal_moves_indices:
                    # Mask illegal moves
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
        
        q_values = self.policy_net(state).gather(1, action)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1)
            expected_q_values = reward + (1 - done) * self.gamma * next_q_values
            
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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
        # When loading for inference, we usually want to lower epsilon
        self.epsilon = 0.1 if training else 0.0
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def _board_to_tensor(self, board):
        # Helper to convert board to tensor (duplicated from ChessEnv for standalone usage)
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
        # Helper to get legal action indices
        legal_moves = []
        for move in board.legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            legal_moves.append(from_square * 64 + to_square)
        return legal_moves

    def _decode_action(self, action_idx: int) -> chess.Move:
        """Decodes an integer 0-4095 into a chess move."""
        from_square = action_idx // 64
        to_square = action_idx % 64
        return chess.Move(from_square, to_square)
