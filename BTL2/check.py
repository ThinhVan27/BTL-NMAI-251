from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import torch
from policy_net import PolicyNet

data = np.load("teacher_minimax.npz")
states = data["states"]
actions = data["actions"]

dataset = TensorDataset(torch.from_numpy(states), torch.from_numpy(actions))
loader = DataLoader(dataset, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PolicyNet().to(device)
policy.load_state_dict(torch.load("models/policy_supervised.pth"))
policy.eval()
correct = 0
total = 0

with torch.no_grad():
    for s_batch, a_batch in loader:
        s_batch = s_batch.to(device)
        a_batch = a_batch.to(device)
        logits = policy(s_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == a_batch).sum().item()
        total += a_batch.size(0)

print("Train accuracy:", correct / total)
