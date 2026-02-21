import os
import re
import json
import math
import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import cv2
import numpy as np

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def scan_images(root: str = "data/raw") -> List[Dict]:
    rows = []
    if not os.path.isdir(root):
        return rows
    for source in sorted(os.listdir(root)):
        sp = os.path.join(root, source)
        if not os.path.isdir(sp):
            continue
        for label in sorted(os.listdir(sp)):
            lp = os.path.join(sp, label)
            if not os.path.isdir(lp):
                continue
            for fn in os.listdir(lp):
                ext = os.path.splitext(fn)[1].lower()
                if ext in VALID_EXTS:
                    path = os.path.join(lp, fn).replace("\\", "/")
                    rows.append({"path": path, "source": source, "label_raw": label, "label_norm": _norm(label)})
    return rows

def build_label_mapping(rows: List[Dict], aliases: Dict[str, str] = None) -> Dict[str, int]:
    mp = {}
    for r in rows:
        k = r["label_norm"]
        if aliases and k in aliases:
            k = _norm(aliases[k])
        mp[k] = 0
    labels = sorted(list(mp.keys()))
    return {lbl: i for i, lbl in enumerate(labels)}

def apply_alias(label_norm: str, aliases: Dict[str, str]) -> str:
    if aliases and label_norm in aliases:
        return _norm(aliases[label_norm])
    return label_norm

def rows_with_final_labels(rows: List[Dict], aliases: Dict[str, str] = None) -> List[Dict]:
    out = []
    for r in rows:
        lbl = apply_alias(r["label_norm"], aliases)
        out.append({"path": r["path"], "source": r["source"], "label": lbl})
    return out

def stratified_split(rows: List[Dict], ratios=(0.7, 0.15, 0.15), seed: int = 42) -> Dict[str, List[Dict]]:
    random.seed(seed)
    by_lbl = defaultdict(list)
    for r in rows:
        by_lbl[r["label"]].append(r)
    train, val, test = [], [], []
    for lbl, items in by_lbl.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
        n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train - 1)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return {"train": train, "val": val, "test": test}

def compute_mean_std(paths: List[str], max_images: int = 2000) -> Tuple[List[float], List[float]]:
    xs = []
    for i, p in enumerate(paths):
        if i >= max_images:
            break
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        arr = img.astype(np.float32) / 255.0
        xs.append(arr.reshape(-1, 3))
    if not xs:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    arr = np.concatenate(xs, axis=0)
    mean = arr.mean(axis=0).tolist()
    std = arr.std(axis=0).tolist()
    return mean, std

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_aliases(path: str = "data/aliases.json") -> Dict[str, str]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def class_counts(rows: List[Dict]) -> Dict[str, int]:
    c = Counter([r["label"] for r in rows])
    return dict(sorted(c.items(), key=lambda x: x[0]))
