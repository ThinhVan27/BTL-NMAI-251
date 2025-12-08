import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from policy_net import PolicyNet


def train_policy(
    data_path: str = "teacher_minimax.npz",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    checkpoint_path: str = "policy_supervised.pth",
):
    data = np.load(data_path)
    states = torch.from_numpy(data["states"]).float()
    actions = torch.from_numpy(data["actions"]).long()

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            logits = model(batch_states)
            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_states.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[INFO] Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Saved trained policy to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Supervised training for policy network.")
    parser.add_argument("--data", type=str, default="teacher_minimax.npz", help="Path to npz dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--out", type=str, default="policy_supervised.pth", help="Checkpoint output path.")
    args = parser.parse_args()

    train_policy(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.out,
    )


if __name__ == "__main__":
    main()
