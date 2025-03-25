import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler
from dataset import WalksDataset
from proxy_network import RnnWalkNet
from types import SimpleNamespace
from tqdm import tqdm


def labels_to_onehot(labels_tensor, num_classes):
    return torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes)


def lr_scaling(x, args):
    x_th = 500e3 / args.scheduler_step_size
    return 1.0 if x < x_th else 0.5

def custom_collate_fn(batch):
    walks, labels, ids = zip(*batch)  # list of Tensors and strings

    walks_tensor = torch.stack(walks)   # Assumes all walks have same shape
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return walks_tensor, labels_tensor, list(ids)

def start_train(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WalksDataset(cfg.train_walk_npz_path)
    val_dataset = WalksDataset(cfg.test_walk_npz_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    proxy_model = RnnWalkNet(cfg, cfg.num_classes, net_input_dim=3).to(device)
    optimizer = torch.optim.Adam(proxy_model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        scale_fn=lambda x: lr_scaling(x, args),
        step_size_up=args.scheduler_step_size,
        scale_mode='cycle'
    )

    val_criterion = torch.nn.CrossEntropyLoss(reduction=args.loss_reduction)
    proxy_train_criterion = torch.nn.KLDivLoss(reduction="sum", log_target=False)

    def _train(epoch):
        correct = 0
        total_loss = 0.
        proxy_model.train()
        pbar = tqdm(train_loader, desc=f"[Train Epoch {epoch}]")

        for batch_idx, (walk_features, labels_, _) in enumerate(pbar):
            walk_features, labels_ = walk_features.to(device), labels_.to(device)
            shape = walk_features.shape  # [B, num_walks, walk_len, 3]
            batch_size, num_walks, walk_len = shape[:3]

            walk_features = walk_features.view(-1, walk_len, 3)
            labels = labels_.repeat_interleave(num_walks)

            optimizer.zero_grad()
            _, logits = proxy_model(walk_features)
            probabilities = F.softmax(logits, dim=1)

            labels_onehot = labels_to_onehot(labels, num_classes=cfg.num_classes).float()
            loss = proxy_train_criterion(F.log_softmax(probabilities, dim=1), labels_onehot)

            total_loss += loss.item()
            predictions = probabilities.argmax(dim=1)
            correct += int((predictions == labels).sum())

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / (len(train_loader.dataset) * num_walks)
        return avg_loss, accuracy

    def _val(epoch):
        proxy_model.eval()
        total_loss = 0.
        correct = 0
        pbar = tqdm(val_loader, desc=f"[Val Epoch {epoch}]")

        with torch.no_grad():
            for walk_features, labels_, _ in pbar:
                walk_features, labels_ = walk_features.to(device), labels_.to(device)
                shape = walk_features.shape
                batch_size, num_walks, walk_len = shape[:3]

                walk_features = walk_features.view(-1, walk_len, 3)
                labels = labels_.repeat_interleave(num_walks)

                _, logits = proxy_model(walk_features)
                probabilities = F.softmax(logits, dim=1)

                labels_onehot = labels_to_onehot(labels, num_classes=cfg.num_classes).float()
                loss = proxy_train_criterion(F.log_softmax(probabilities, dim=1), labels_onehot)

                total_loss += loss.item()
                predictions = probabilities.argmax(dim=1)
                correct += int((predictions == labels).sum())

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / (len(val_loader.dataset) * num_walks)
        return avg_loss, accuracy

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n[INFO] Starting epoch {epoch}/{args.num_epochs}...")
        train_loss, train_accuracy = _train(epoch)
        val_loss, val_accuracy = _val(epoch)

        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if epoch % 1 == 0: # after testing - change 1 to 5 so we save every 5 epochs. 
            proxy_model.save_weights(cfg.logdir, step=epoch, keep=True, optimizer=optimizer)

    return train_losses, val_losses, train_accuracies, val_accuracies


def main():
    print("[INFO] Loading configuration and starting training...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/proxy_params.json")
    cli_args = parser.parse_args()

    with open(cli_args.config, "r") as f:
        cfg_dict = json.load(f)

    cfg = SimpleNamespace(**cfg_dict)
    cfg.args = SimpleNamespace(**cfg.args)

    print("[INFO] Training initialized. Running epochs...")
    start_train(cfg, cfg.args)


if __name__ == "__main__":
    main()
