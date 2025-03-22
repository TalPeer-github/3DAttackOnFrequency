import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import WalksDataset
from proxy_network import RnnWalkNet


def kl_self_loss(logits_ref, logits_pred):
    probs_ref = F.softmax(logits_ref.detach(), dim=1)         # Target distribution (frozen)
    log_probs_pred = F.log_softmax(logits_pred, dim=1)        # Current model output
    return F.kl_div(log_probs_pred, probs_ref, reduction="batchmean")


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.
    total_samples = 0

    for walks, _, _ in tqdm(dataloader, desc="Training", leave=False):
        B, N, T, D = walks.shape
        walks = walks.view(B * N, T, D).to(device)

        # 1. Reference output (clean prediction)
        with torch.no_grad():
            logits_ref = model(walks)

        # 2. Forward pass again on (same or slightly transformed walk)
        logits_pred = model(walks)

        # 3. Minimize KL divergence
        loss = kl_self_loss(logits_ref, logits_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * walks.size(0)
        total_samples += walks.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.
    total_samples = 0

    for walks, _, _ in tqdm(dataloader, desc="Validation", leave=False):
        B, N, T, D = walks.shape
        walks = walks.view(B * N, T, D).to(device)

        logits_ref = model(walks)
        logits_pred = model(walks)

        loss = kl_self_loss(logits_ref, logits_pred)

        total_loss += loss.item() * walks.size(0)
        total_samples += walks.size(0)

    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="proxy_params.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_set = WalksDataset(cfg["train_walk_npz_path"])
    test_set = WalksDataset(cfg["test_walk_npz_path"])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    # Initialize CloudWalker (proxy model)
    model = RnnWalkNet(cfg, cfg["num_classes"], net_input_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    os.makedirs(cfg["logdir"], exist_ok=True)

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, test_loader, device)

        print(f"[Epoch {epoch:02d}] Train KL: {train_loss:.4f} | Val KL: {val_loss:.4f}")
        model.save_weights(cfg["logdir"], step=epoch, keep=True)


if __name__ == "__main__":
    main()
