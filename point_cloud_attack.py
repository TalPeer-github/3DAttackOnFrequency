import os
from time import time
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch_geomtric
from torch_geometric.data import Data
import torch_geometric.transforms as T

import scipy

from utils.config import modelnet10_classes, args, label_to_class, class_to_label
from utils.attack_utils import ClipPointsLinf, UntargetedLogitsAdvLoss, L2Dist, CrossEntropyAdvLoss, ClipPointsL2

from utils.visalizations import plot_pc, visualize_point_cloud_spectral, plot_attack


def knn(x, k):
    """
    x:(B, 3, N)
    """
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)

        vec = x.transpose(2, 1).unsqueeze(2) - x.transpose(2, 1).unsqueeze(1)
        dist = -torch.sum(vec ** 2, dim=-1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_Laplace_from_pc(ori_pc, data):
    """
    ori_pc:(B, 3, N) - shape (B, 3, N) where B is batch size, 3 is coordinates, and N is number of points
    """

    pc = ori_pc.detach().clone()

    with torch.no_grad():
        idx = knn(pc, 30)
        pc = pc.transpose(2, 1).contiguous()  # (B, N, 3)
        point_mat = pc.unsqueeze(2) - pc.unsqueeze(1)  # (B, N, N, 3)
        A = torch.exp(-torch.sum(point_mat.square(), dim=3))  # (B, N, N)
        mask = torch.zeros_like(A)
        B, N, k = idx.shape
        idx = idx.reshape(B, N, k, 1)
        mask = mask.unsqueeze(-1)
        mask.scatter_(2, idx, 1)

        mask = mask.squeeze(-1)
        mask = mask + mask.transpose(2, 1)
        mask[mask > 1] = 1

        A = A * mask
        D = torch.diag_embed(torch.sum(A, dim=2))
        L = D - A
        e, v = torch.linalg.eigh(L)
    return e.to(ori_pc), v.to(ori_pc)


class CWAOF:
    def __init__(self, model, device='cuda', eps=0.2, steps=10, lr=1e-3,
                 clip_mode='linf', budget=0.18, low_pass=100, gamma=0.5, max_iter=150, kappa=30):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.eps = eps
        self.steps = steps
        self.lr = lr
        self.gamma = gamma
        self.clip_mode = clip_mode
        self.low_pass = low_pass

        self.model = model
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam
        self.max_iter = max_iter
        self.clip_func = ClipPointsLinf(budget=budget) if self.clip_mode == 'linf' else ClipPointsL2(budget=budget)
        self.adv_func = UntargetedLogitsAdvLoss(kappa=kappa)
        self.dist_func = L2Dist()
        self.model.eval()
        print(f"Attack Module initialized:\n"
              f"\t low pass = {self.low_pass}\n\t clip mode = {self.clip_mode}\n\t kappa = {kappa}")

    def attack(self, data, specific_lowpass=False):
        if specific_lowpass:
            print(f"Attack starts:\n\t low pass = {self.low_pass}")

        self.model.eval()

        data = data.to(self.device)

        unsqueeze_pos = data.pos.unsqueeze(0).float().cuda().detach()
        unsqueeze_pos = unsqueeze_pos.transpose(2, 1).contiguous()

        B, _, N = unsqueeze_pos.shape

        ori_data = unsqueeze_pos.clone().detach().to(self.device)
        ori_data.requires_grad = False
        target = data.y.clone().detach().to(self.device)
        label_val = target.detach().cpu().numpy()

        attack_successes_list = []

        best_dist = 1e10
        best_attack = None
        reached = False
        for step in tqdm(range(self.steps)):
            adv_data = ori_data.clone().detach() + torch.randn_like(ori_data) * 1e-7
            adv_data.requires_grad = False

            Evs, V = get_Laplace_from_pc(adv_data, data)
            projs = torch.bmm(adv_data, V)
            lfc = torch.bmm(projs[..., :self.low_pass], V[..., :self.low_pass].transpose(2, 1))
            hfc = torch.bmm(projs[..., self.low_pass:], V[..., self.low_pass:].transpose(2, 1))
            lfc = lfc.detach().clone()
            hfc = hfc.detach().clone()
            lfc.requires_grad_()
            hfc.requires_grad_(False)
            ori_lfc = lfc.detach().clone()
            ori_lfc.requires_grad_(False)
            ori_hfc = hfc.detach().clone()
            ori_hfc.requires_grad_(False)

            optimizer = optim.Adam([lfc], lr=self.lr, betas=(0.9, 0.999), weight_decay=0.)
            adv_loss = torch.tensor(0.).cuda()
            if step == 0:
                with torch.no_grad():
                    _data = ori_lfc + ori_hfc
                    _data = _data.transpose(1, 2).squeeze(0)
                    orig_data = Data(pos=_data, x=data.x, batch=data.batch, y=data.y)
                    clean_output = self.model(orig_data)
                    clean_pred = torch.argmax(clean_output, dim=-1)
                    # if target == clean_pred.detach().cpu().numpy():
                    target = clean_pred.clone().detach().to(self.device)

            for iteration in range(self.max_iter):
                adv_data = lfc + hfc
                adv_data = adv_data.transpose(1, 2).squeeze(0)
                adversarial_data = Data(pos=adv_data, x=data.x, batch=data.batch, y=data.y)
                logits = self.model(adversarial_data)
                adv_loss = (1 - self.gamma) * self.adv_func(logits, target).mean()
                optimizer.zero_grad()
                adv_loss.backward()

                adversarial_lfc = Data(pos=lfc.transpose(1, 2).squeeze(0), x=data.x, batch=data.batch, y=data.y)
                lfc_logits = self.model(adversarial_lfc)
                lfc_adv_loss = self.gamma * self.adv_func(lfc_logits, target).mean()
                lfc_adv_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    adv_data = lfc + hfc
                    adv_data.data = self.clip_func(adv_data.detach().clone(), ori_data)
                    coeff = torch.bmm(adv_data, V)
                    hfc.data = torch.bmm(coeff[..., self.low_pass:],
                                         V[..., self.low_pass:].transpose(2, 1))  # (B, 3, N)
                    lfc.data = torch.bmm(coeff[..., :self.low_pass], V[..., :self.low_pass].transpose(2, 1))

                with torch.no_grad():
                    adv_logits = self.model(adversarial_data)
                    adv_pred = torch.argmax(adv_logits, dim=1)

                    lfc_adv_logits = self.model(adversarial_lfc)
                    lfc_pred = torch.argmax(lfc_adv_logits, dim=1)

                attack_successes_count = ((adv_pred != target) * (lfc_pred != target)).sum().item()
                attack_successes_list.append(attack_successes_count)

                dist_current_val = torch.sqrt(torch.sum((adv_data - ori_data) ** 2, dim=[1, 2])).detach().cpu().numpy()

                attack_success_cond = attack_successes_count > 0
                dist_cond = dist_current_val.item(0) <= best_dist

                if (dist_cond and attack_success_cond):
                    best_dist = dist_current_val.item(0)
                    best_attack = adv_data.detach().clone()

        if best_attack is None:
            best_attack = adv_data.detach().clone()
        adv_pc = self.clip_func(best_attack, ori_data)
        adv_pc = adv_pc.detach().transpose(2, 1)

        adv_pc = adv_pc.squeeze(0).float()
        adversarial_data.pos = adv_pc
        adversarial_data.to(self.device)
        with torch.no_grad():
            adv_logits = self.model(adversarial_data)
            adv_pred = torch.argmax(adv_logits, dim=-1)

        plot_pc(pos_orig=data.pos, pos_adv=adv_pc, true_class=target, pred_class=clean_pred, adv_class=adv_pred)
        return adv_pc, clean_pred, adv_pred

    def batch_attack(self, dataloader):
        """
        Generate adversarial examples for a batch of point clouds
        
        Args:
            dataloader: DataLoader containing point clouds to attack
            
        Returns:
            adversarial_examples: List of adversarial point clouds
            original_preds: List of original predictions
            adversarial_preds: List of predictions on adversarial examples
        """
        adversarial_examples = []
        clean_preds = []
        adversarial_preds = []
        true_labels = []
        i = 1
        for data in tqdm(dataloader, desc="Spectral Adversarial Attack progress"):

            adv_data, clean_pred, adv_pred = self.attack(data)
            adversarial_examples.append(adv_data)
            true_labels.extend(data.y.cpu().numpy().tolist())
            clean_preds.extend(clean_pred.cpu().numpy().tolist())
            adversarial_preds.extend(adv_pred.cpu().numpy().tolist())

            plot_attack(orig_pos=data.pos.cpu().numpy(), adv_pos=adv_data.cpu().numpy(),
                        clean_pred=clean_pred.cpu().item(), adv_pred=adv_pred.cpu().item(),
                        true_label=data.y.cpu().item(), linf=self.clip_mode == 'linf')

            try:
                if i % 5 == 0:
                    clean_acc = (clean_preds == true_labels).sum().item() / len(clean_preds)
                    adv_acc = ((clean_preds != adversarial_preds) * (true_labels == clean_preds)).sum().item() / len(
                        clean_preds)
                    print(f"{i} Samples:")
                    print(f"\tClean accuracy: {clean_acc * 100}%")
                    print(f"\tAttack success rate: {adv_acc * 100}%")
                i += 1
            except Exception as e:
                continue
        return adversarial_examples, clean_preds, adversarial_preds, true_labels
