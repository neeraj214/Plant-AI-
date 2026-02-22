import os
import json
import time
import argparse
import numpy as np
import cv2
import torch
import tensorflow as tf
from src.torch_model import PlantClassifier
from src.torch_dataset import load_labels, load_splits

def load_images(rows, k=100):
    xs, ys = [], []
    for r in rows[:k]:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (224,224)).astype(np.float32)/255.0
        xs.append(x)
        ys.append(r["label"])
    return np.stack(xs, axis=0), ys

def torch_predict_batch(model, xs, device):
    with torch.no_grad():
        t = torch.from_numpy(xs.transpose(0,3,1,2)).to(device)
        logits = model(t)
        p = torch.argmax(logits, dim=1).cpu().numpy()
    return p

def tflite_predict_batch(interpreter, xs):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zp = inp["quantization"]
    if inp["dtype"].__name__ == "int8":
        q = np.clip(np.round(xs/scale + zp), -128, 127).astype(np.int8)
    else:
        q = xs.astype(inp["dtype"])
    interpreter.allocate_tensors()
    preds = []
    for i in range(xs.shape[0]):
        interpreter.set_tensor(inp["index"], np.expand_dims(q[i], 0))
        interpreter.invoke()
        y = interpreter.get_tensor(out["index"]).squeeze()
        if out["dtype"].__name__ == "int8":
            s, z = out["quantization"]
            y = s * (y.astype(np.float32) - z)
        preds.append(int(np.argmax(y)))
    return np.array(preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", type=str, default="models/swa.pth")
    ap.add_argument("--int8", type=str, default="models/model_quantized.tflite")
    args = ap.parse_args()
    labels = load_labels("data/labels.json")
    splits = load_splits("data/splits.json")
    xs, ys_lbl = load_images(splits["test"], k=200)
    idx = [labels[y] for y in ys_lbl]
    res = {}
    if os.path.isfile(args.fp32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PlantClassifier(num_classes=len(labels)).to(device)
        sd = torch.load(args.fp32, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        t0 = time.perf_counter()
        p = torch_predict_batch(model, xs, device)
        dt = (time.perf_counter() - t0) * 1000.0
        acc = float((p == np.array(idx)).mean())
        size_mb = os.path.getsize(args.fp32) / (1024*1024)
        res["fp32"] = {"model_size_mb": size_mb, "inference_ms_batch": dt, "top1_acc": acc}
    if os.path.isfile(args.int8):
        interpreter = tf.lite.Interpreter(model_path=args.int8)
        t0 = time.perf_counter()
        p = tflite_predict_batch(interpreter, xs)
        dt = (time.perf_counter() - t0) * 1000.0
        acc = float((p == np.array(idx)).mean())
        size_mb = os.path.getsize(args.int8) / (1024*1024)
        res["int8"] = {"model_size_mb": size_mb, "inference_ms_batch": dt, "top1_acc": acc}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/benchmark.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("outputs/benchmark.json")

if __name__ == "__main__":
    main()
