import os
import csv

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from base.cloud_walker.evaluate_classification import *
import config
import imitation_train

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train(model, train_loader, optimizer, criterion):
    """
     Iterate in batches over the training dataset.
     Perform a single forward pass
     Compute the loss.
     Derive gradients
     Update parameters based on gradients
     Clear gradients
    """
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, loader, criterion):
    """
    Iterate in batches over the training/test dataset.
    Use the class with the highest probability.
    Check against ground-truth labels.
    Derive ratio of correct predictions.
    :param loader:
    :return:
    """
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits, data.y)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += int((pred == data.y).sum())

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


@torch.no_grad()
def predict_test(model, loader, output_file="predictions.txt", include_score=True):
    # TODO - match attack logic
    args = config.args

    model.eval()
    preds = []
    confidences = []
    y_true = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            conf = prob.max(dim=1).values

            preds.append(pred.cpu().numpy())
            confidences.append(conf.cpu().numpy())
            y_true.extend(data.y)

    if args.run_mode == 'eval':
        evaluate_metrics(y_true=y_true, y_pred=preds, scores=confidences, exp_args=args)


def run_experiment():
    args = config.args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    follow_training = False  # TODO - set later
    create_pseudo_test = False  # TODO - set later
    train_losses, val_losses, train_accuracies, val_accuracies = imitation_train.train_proxy(device)

    if follow_training:
        plot_learning_curve(train_accuracies, val_accuracies, train_losses, val_losses)

    if args.run_mode == 'eval' and create_pseudo_test:  # TODO - align with evaluate_classification.py
        test_mean_acc, test_mean_acc_per_class, _ = calc_accuracy_test()
        return train_losses, val_losses, train_accuracies, val_accuracies, test_mean_acc, test_mean_acc_per_class
    return train_losses, val_losses, train_accuracies, val_accuracies, 0, 0
