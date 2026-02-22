import os
import json
import math
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from src.torch_model import PlantClassifier
from src.torch_dataset import PlantDataset, load_labels, load_splits

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.softmax(logits, dim=1)[torch.arange(logits.size(0), device=logits.device), targets]
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

def prepare_loaders(batch_size=32, num_workers=2):
    labels = load_labels("data/labels.json")
    splits = load_splits("data/splits.json")
    inv = {k: v for k, v in labels.items()}
    train_ds = PlantDataset(splits["train"], labels, split="train")
    val_ds = PlantDataset(splits["val"], labels, split="val")
    test_ds = PlantDataset(splits["test"], labels, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, len(labels)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(p.tolist())
    acc = accuracy_score(ys, ps)
    mf1 = f1_score(ys, ps, average="macro", zero_division=0)
    cm = confusion_matrix(ys, ps)
    return acc, mf1, cm

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, num_classes = prepare_loaders(args.batch_size, args.num_workers)
    model = PlantClassifier(num_classes=num_classes, backbone=args.backbone, dropout=args.dropout).to(device)
    model.freeze_backbone()
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head, weight_decay=args.weight_decay)
    sched = CosineAnnealingWarmRestarts(opt, T_anneal := args.t0, T_mult=args.t_mult)
    criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_f1 = 0.0
    for epoch in range(args.head_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step(epoch + 1)
        acc, mf1, _ = evaluate(model, val_loader, device)
        if mf1 > best_f1:
            best_f1 = mf1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("models", "best_head.pth"))
    model.unfreeze_last_fraction(args.unfreeze_frac)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_ft, weight_decay=args.weight_decay)
    sched = CosineAnnealingWarmRestarts(opt, T_anneal, T_mult=args.t_mult)
    swa_model = AveragedModel(model)
    swa_start = max(0, args.ft_epochs - args.swa_epochs)
    swa_sched = SWALR(opt, anneal_strategy="cos", anneal_epochs=1, swa_lr=args.swa_lr)
    for epoch in range(args.ft_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step(epoch + 1)
        acc, mf1, _ = evaluate(model, val_loader, device)
        if mf1 > best_f1:
            best_f1 = mf1
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("models", "best_finetune.pth"))
    update_bn(train_loader, swa_model, device=device)
    os.makedirs("models", exist_ok=True)
    torch.save(swa_model.state_dict(), os.path.join("models", "swa.pth"))
    acc_val, mf1_val, cm_val = evaluate(swa_model, val_loader, device)
    acc_test, mf1_test, cm_test = evaluate(swa_model, test_loader, device)
    with open("outputs/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": {"acc": acc_val, "macro_f1": mf1_val}, "test": {"acc": acc_test, "macro_f1": mf1_test}}, f, indent=2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="efficientnetv2_rw_s")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--t0", type=int, default=5)
    p.add_argument("--t_mult", type=int, default=2)
    p.add_argument("--head_epochs", type=int, default=5)
    p.add_argument("--ft_epochs", type=int, default=10)
    p.add_argument("--swa_epochs", type=int, default=3)
    p.add_argument("--swa_lr", type=float, default=5e-6)
    p.add_argument("--unfreeze_frac", type=float, default=0.3)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
