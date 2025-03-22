import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WalksDataset
from proxy_network import RnnWalkNet


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.
    total_correct = 0
    total_samples = 0

    for walks, labels, _ in tqdm(dataloader, desc="Training", leave=False):
        B, N, T, D = walks.shape
        walks = walks.view(B * N, T, D).to(device)
        labels = labels.repeat_interleave(N).to(device)  # (B * N,)

        logits = model(walks)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * walks.size(0)
        total_samples += walks.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_samples = 0

    for walks, labels, _ in tqdm(dataloader, desc="Validation", leave=False):
        B, N, T, D = walks.shape
        walks = walks.view(B * N, T, D).to(device)
        labels = labels.repeat_interleave(N).to(device)

        logits = model(walks)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * walks.size(0)
        total_samples += walks.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/proxy_params.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_dataset = WalksDataset(cfg["train_walk_npz_path"])
    val_dataset = WalksDataset(cfg["test_walk_npz_path"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    # Init model
    model = RnnWalkNet(cfg, cfg["num_classes"], net_input_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(cfg["logdir"], exist_ok=True)

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        model.save_weights(cfg["logdir"], step=epoch, keep=True)


if __name__ == "__main__":
    main()
