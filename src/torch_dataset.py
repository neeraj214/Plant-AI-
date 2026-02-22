import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_stats(path="data/stats.json"):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            o = json.load(f)
            return o.get("mean", [0.485,0.456,0.406]), o.get("std", [0.229,0.224,0.225])
    return [0.485,0.456,0.406], [0.229,0.224,0.225]

def load_splits(path="data/splits.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_labels(path="data/labels.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_transforms(split="train", mean=None, std=None):
    if mean is None or std is None:
        mean, std = load_stats()
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(224, 224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomSunFlare(flare_roi=(0,0,1,0.5), angle_lower=0.5, p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

class PlantDataset(Dataset):
    def __init__(self, rows, label_to_idx, split="train"):
        self.rows = rows
        self.label_to_idx = label_to_idx
        mean, std = load_stats()
        self.tf = build_transforms(split, mean, std)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = cv2.imread(r["path"])
        if img is None:
            img = np.zeros((224,224,3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.tf(image=img)["image"]
        y = self.label_to_idx[r["label"]]
        return out.float(), torch.tensor(y, dtype=torch.long)
