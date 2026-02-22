import os
import argparse
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from src.torch_model import PlantClassifier
from src.torch_dataset import PlantDataset, load_labels, load_splits
from src.torch_cam import GradCAM, cam_to_numpy, overlay_heatmap

def load_model(weights, backbone, num_classes, device):
    model = PlantClassifier(num_classes=num_classes, backbone=backbone)
    if os.path.isfile(weights):
        sd = torch.load(weights, map_location=device)
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

def sample_rows(split="test", k=8):
    labels = load_labels("data/labels.json")
    splits = load_splits("data/splits.json")
    rows = splits[split]
    if len(rows) <= k:
        return rows, labels
    return random.sample(rows, k), labels

def run_gradcam(weights, backbone, split, k, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows, labels = sample_rows(split, k)
    inv = {v: k for k, v in labels.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, backbone, len(labels), device)
    cam = GradCAM(model)
    for i, r in enumerate(rows):
        img = cv2.imread(r["path"])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (224,224))
        x_t = torch.from_numpy(x.transpose(2,0,1)).float().unsqueeze(0)/255.0
        with torch.no_grad():
            logits = model(x_t.to(device))
            pred = int(torch.argmax(logits, dim=1).item())
        heat = cam(x_t.to(device), class_idx=torch.tensor([pred], device=device))
        heat = cam_to_numpy(heat, (x.shape[1], x.shape[0]))[0]
        overlay = overlay_heatmap(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), heat, 0.4)
        grid = np.concatenate([x, cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1], overlay], axis=1)
        cv2.imwrite(os.path.join(outdir, f"cam_{i}_pred_{inv[pred]}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

def run_sanity(weights, backbone, split, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows, labels = sample_rows(split, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_t = load_model(weights, backbone, len(labels), device)
    model_r = PlantClassifier(num_classes=len(labels), backbone=backbone).to(device).eval()
    cam_t = GradCAM(model_t)
    cam_r = GradCAM(model_r)
    r = rows[0]
    img = cv2.imread(r["path"])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (224,224))
    x_t = torch.from_numpy(x.transpose(2,0,1)).float().unsqueeze(0)/255.0
    with torch.no_grad():
        logits_t = model_t(x_t.to(device))
        pred_t = int(torch.argmax(logits_t, dim=1).item())
        logits_r = model_r(x_t.to(device))
        pred_r = int(torch.argmax(logits_r, dim=1).item())
    ht = cam_to_numpy(cam_t(x_t.to(device), class_idx=torch.tensor([pred_t], device=device)), (x.shape[1], x.shape[0]))[0]
    hr = cam_to_numpy(cam_r(x_t.to(device), class_idx=torch.tensor([pred_r], device=device)), (x.shape[1], x.shape[0]))[0]
    t = ht.flatten()
    s = hr.flatten()
    t = (t - t.mean()) / (t.std() + 1e-6)
    s = (s - s.mean()) / (s.std() + 1e-6)
    corr = float(np.clip(np.corrcoef(t, s)[0,1], -1, 1))
    grid = np.concatenate([x, cv2.applyColorMap((ht*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1], cv2.applyColorMap((hr*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]], axis=1)
    cv2.imwrite(os.path.join(outdir, "sanity_randomized.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    with open(os.path.join(outdir, "san MutCorr.json").replace(" ", ""), "w") as f:
        json.dump({"pearson_correlation": corr}, f, indent=2)

def run_occlusion(weights, backbone, split, outdir, patch=32, stride=16):
    os.makedirs(outdir, exist_ok=True)
    rows, labels = sample_rows(split, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, backbone, len(labels), device)
    r = rows[0]
    img = cv2.imread(r["path"])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (224,224))
    x_t = torch.from_numpy(x.transpose(2,0,1)).float().unsqueeze(0)/255.0
    with torch.no_grad():
        logits = model(x_t.to(device))
        pred = int(torch.argmax(logits, dim=1).item())
        base = torch.softmax(logits, dim=1)[0, pred].item()
    H, W = 224, 224
    heat = np.zeros((H, W), dtype=np.float32)
    gray = np.full((patch, patch, 3), 127, dtype=np.uint8)
    for y in range(0, H - patch + 1, stride):
        for x0 in range(0, W - patch + 1, stride):
            x_mod = x.copy()
            x_mod[y:y+patch, x0:x0+patch] = gray
            x_m = torch.from_numpy(x_mod.transpose(2,0,1)).float().unsqueeze(0)/255.0
            with torch.no_grad():
                p = torch.softmax(model(x_m.to(device)), dim=1)[0, pred].item()
            drop = max(0.0, base - p)
            heat[y:y+patch, x0:x0+patch] = np.maximum(heat[y:y+patch, x0:x0+patch], drop)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-6)
    overlay = overlay_heatmap(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), heat, 0.4)
    grid = np.concatenate([x, cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1], overlay], axis=1)
    cv2.imwrite(os.path.join(outdir, "occlusion.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

def run_lime(weights, backbone, split, outdir):
    try:
        from lime import lime_image
        from skimage.segmentation import quickshift
    except Exception:
        return
    os.makedirs(outdir, exist_ok=True)
    rows, labels = sample_rows(split, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, backbone, len(labels), device)
    def predict_fn(xs):
        xs = torch.from_numpy(xs.transpose(0,3,1,2)).float()/255.0
        with torch.no_grad():
            logits = model(xs.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
        return p
    r = rows[0]
    img = cv2.imread(r["path"])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (224,224))
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    top = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top, positive_only=True, num_features=10, hide_rest=False)
    cv2.imwrite(os.path.join(outdir, "lime.png"), cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="gradcam")
    ap.add_argument("--weights", type=str, default="models/swa.pth")
    ap.add_argument("--backbone", type=str, default="efficientnetv2_rw_s")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="outputs/xai")
    args = ap.parse_args()
    if args.mode == "gradcam":
        run_gradcam(args.weights, args.backbone, args.split, args.k, os.path.join(args.outdir, "gradcam"))
    elif args.mode == "sanity":
        run_sanity(args.weights, args.backbone, args.split, os.path.join(args.outdir, "sanity"))
    elif args.mode == "occlusion":
        run_occlusion(args.weights, args.backbone, args.split, os.path.join(args.outdir, "occlusion"))
    elif args.mode == "lime":
        run_lime(args.weights, args.backbone, args.split, os.path.join(args.outdir, "lime"))

if __name__ == "__main__":
    main()
