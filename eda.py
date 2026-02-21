import os
from collections import Counter
import json
import matplotlib.pyplot as plt
from src.data_utils import scan_images, rows_with_final_labels, build_label_mapping, load_aliases, class_counts, save_json

def ensure_metadata():
    if os.path.isfile("data/metadata.json"):
        with open("data/metadata.json", "r", encoding="utf-8") as f:
            rows = json.load(f)
        return rows
    aliases = load_aliases()
    rows = scan_images("data/raw")
    rows = rows_with_final_labels(rows, aliases)
    save_json(rows, "data/metadata.json")
    return rows

def plot_counts(counts):
    os.makedirs("outputs/eda", exist_ok=True)
    labels = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize=(max(8, len(labels) * 0.4), 6))
    plt.bar(labels, values)
    plt.xticks(rotation=90)
    plt.ylabel("images")
    plt.tight_layout()
    out = "outputs/eda/class_distribution.png"
    plt.savefig(out, dpi=150)
    print(out)

def main():
    rows = ensure_metadata()
    if not rows:
        print("no_images_found")
        return
    counts = class_counts(rows)
    plot_counts(counts)

if __name__ == "__main__":
    main()
