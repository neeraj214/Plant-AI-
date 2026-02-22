import os
import json
import argparse
import numpy as np
import cv2
import torch
from src.torch_model import PlantClassifier
from src.torch_dataset import load_labels, load_splits, load_stats
from src.torch_cam import GradCAM, cam_to_numpy, overlay_heatmap

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None, None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (224,224))
    mean, std = load_stats()
    t = (x.astype(np.float32)/255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    t = torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0).float()
    return img, x, t

def occlusion_map(model, x_rgb, pred_idx, patch=32, stride=16, device="cpu"):
    H, W = 224, 224
    base = torch.softmax(model(torch.from_numpy((x_rgb/255.0).transpose(2,0,1)).unsqueeze(0).float().to(device)), dim=1)[0, pred_idx].item()
    heat = np.zeros((H, W), dtype=np.float32)
    gray = np.full((patch, patch, 3), 127, dtype=np.uint8)
    for y in range(0, H - patch + 1, stride):
        for x0 in range(0, W - patch + 1, stride):
            x_mod = x_rgb.copy()
            x_mod[y:y+patch, x0:x0+patch] = gray
            t = torch.from_numpy((x_mod/255.0).transpose(2,0,1)).unsqueeze(0).float().to(device)
            p = torch.softmax(model(t), dim=1)[0, pred_idx].item()
            drop = max(0.0, base - p)
            heat[y:y+patch, x0:x0+patch] = np.maximum(heat[y:y+patch, x0:x0+patch], drop)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-6)
    return heat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="models/swa.pth")
    ap.add_argument("--backbone", type=str, default="efficientnetv2_rw_s")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--outdir", type=str, default="outputs/failures")
    args = ap.parse_args()
    labels = load_labels("data/labels.json")
    splits = load_splits("data/splits.json")
    inv = {v: k for k, v in labels.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantClassifier(num_classes=len(labels), backbone=args.backbone).to(device)
    sd = torch.load(args.weights, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    os.makedirs(args.outdir, exist_ok=True)
    report = []
    count = 0
    for r in splits[args.split]:
        img_bgr, x_rgb, t = preprocess(r["path"])
        if t is None:
            continue
        with torch.no_grad():
            logits = model(t.to(device))
            pred = int(torch.argmax(logits, dim=1).item())
            conf = float(torch.softmax(logits, dim=1)[0, pred].item())
        true = labels[r["label"]]
        if pred != true:
            cam = GradCAM(model)
            heat_t = cam(t.to(device), class_idx=torch.tensor([pred], device=device))
            heat = cam_to_numpy(heat_t, (224,224))[0]
            overlay_cam = overlay_heatmap(cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR), heat, 0.4)[:,:,::-1]
            occ = occlusion_map(model, x_rgb, pred, device=device)
            overlay_occ = overlay_heatmap(cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR), occ, 0.4)[:,:,::-1]
            grid = np.concatenate([x_rgb, (cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]), overlay_cam, overlay_occ], axis=1)
            name = f"fail_{count}_true_{inv[true]}_pred_{inv[pred]}.png"
            cv2.imwrite(os.path.join(args.outdir, name), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            report.append({"path": r["path"], "true": inv[true], "pred": inv[pred], "confidence": conf, "out": name})
            count += 1
            if count >= args.limit:
                break
    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(os.path.join(args.outdir, "report.json"))

if __name__ == "__main__":
    main()
