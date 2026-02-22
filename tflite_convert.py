import os
import json
import argparse
import cv2
import numpy as np
import tensorflow as tf
from src.data_utils import load_aliases, scan_images, rows_with_final_labels, save_json

def rep_images(paths, target=(224,224), max_images=500):
    imgs = []
    for i, p in enumerate(paths):
        if i >= max_images:
            break
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target)
        imgs.append(img.astype(np.float32)/255.0)
    return imgs

def rep_dataset_gen(images):
    for img in images:
        yield [np.expand_dims(img, 0)]

def convert(saved_model_dir, out_path, rep_paths):
    imgs = rep_images(rep_paths)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: rep_dataset_gen(imgs)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model", type=str, required=True)
    ap.add_argument("--out", type=str, default="models/model_quantized.tflite")
    ap.add_argument("--rep_root", type=str, default="data/raw")
    args = ap.parse_args()
    rows = rows_with_final_labels(scan_images(args.rep_root), load_aliases())
    paths = [r["path"] for r in rows]
    out = convert(args.saved_model, args.out, paths)
    print(out)

if __name__ == "__main__":
    main()
