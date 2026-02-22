import os
import json
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from src.torch_model import PlantClassifier
from src.torch_dataset import PlantDataset, load_labels, load_splits

def prepare_loader(split="test", batch_size=64, num_workers=2):
    labels = load_labels("data/labels.json")
    splits = load_splits("data/splits.json")
    rows = splits[split]
    ds = PlantDataset(rows, labels, split="val")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True), len(labels)

def metrics(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(p.tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro", zero_division=0), confusion_matrix(ys, ps).tolist()

def latency_cpu(model, loader, warmup=5, iters=50):
    dev = torch.device("cpu")
    model_cpu = model.to(dev)
    model_cpu.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        for _ in range(warmup):
            model_cpu(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model_cpu(x)
        dt = time.perf_counter() - t0
    n = x.shape[0]
    return (dt / iters) * 1000.0, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="models/swa.pth")
    ap.add_argument("--backbone", type=str, default="efficientnetv2_rw_s")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, num_classes = prepare_loader("test", args.batch_size, args.num_workers)
    model = PlantClassifier(num_classes=num_classes, backbone=args.backbone).to(device)
    if os.path.isfile(args.weights):
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd, strict=False)
    acc, mf1, cm = metrics(model, loader, device)
    ms, n = latency_cpu(model, loader)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval.json", "w", encoding="utf-8") as f:
        json.dump({"acc": acc, "macro_f1": mf1, "confusion_matrix": cm, "latency_ms_per_batch": ms, "batch_size": n}, f, indent=2)
    print("outputs/eval.json")

if __name__ == "__main__":
    main()
