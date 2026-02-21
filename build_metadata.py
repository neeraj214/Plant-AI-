import os
import sys
import json
from src.data_utils import scan_images, rows_with_final_labels, build_label_mapping, stratified_split, compute_mean_std, save_json, load_aliases

def main():
    aliases = load_aliases()
    rows = scan_images("data/raw")
    rows = rows_with_final_labels(rows, aliases)
    labels = build_label_mapping(rows, aliases)
    splits = stratified_split(rows, (0.7, 0.15, 0.15))
    mean, std = compute_mean_std([r["path"] for r in rows])
    os.makedirs("data", exist_ok=True)
    save_json(rows, "data/metadata.json")
    save_json(labels, "data/labels.json")
    save_json(splits, "data/splits.json")
    save_json({"mean": mean, "std": std}, "data/stats.json")
    with open("data/metadata.csv", "w", encoding="utf-8") as f:
        f.write("path,label,source\n")
        for r in rows:
            f.write(f"{r['path']},{r['label']},{r['source']}\n")
    print("metadata.json, labels.json, splits.json, stats.json, metadata.csv")

if __name__ == "__main__":
    main()
