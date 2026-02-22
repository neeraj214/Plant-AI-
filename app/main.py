import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import cv2
import torch
from src.torch_model import PlantClassifier
from src.torch_cam import GradCAM, cam_to_numpy, overlay_heatmap
from src.torch_dataset import load_stats

st.set_page_config(page_title="Am I Healthy?", page_icon="🌿", layout="centered")
st.title("Am I Healthy? Plant Disease Identifier")

labels_path = os.path.join("data", "labels.json")
class_names = None
if os.path.isfile(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        lbls = json.load(f)
        inv = {v: k for k, v in lbls.items()}
        class_names = [inv[i] for i in range(len(inv))]

weights_path = os.path.join("models", "swa.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
if os.path.isfile(weights_path):
    try:
        n_classes = len(class_names) if class_names else 38
        model = PlantClassifier(num_classes=n_classes).to(device)
        sd = torch.load(weights_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
    except Exception:
        model = None

alpha = st.slider("Grad-CAM overlay strength", 0.1, 0.9, 0.4, 0.05)
uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_column_width=True)
    img_np = np.array(image)
    x = cv2.resize(img_np, (224, 224))
    mean, std = load_stats()
    t = (x.astype(np.float32)/255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    t = torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0)
    if model is None:
        st.warning("PyTorch weights not found at models/swa.pth")
    else:
        with torch.no_grad():
            logits = model(t.to(device))
            pred = int(torch.argmax(logits, dim=1).item())
            conf = float(torch.softmax(logits, dim=1)[0, pred].item())
        name = class_names[pred] if class_names and 0 <= pred < len(class_names) else str(pred)
        st.subheader(f"Prediction: {name}  •  Confidence: {conf:.2f}")
        try:
            cam = GradCAM(model)
            heat_t = cam(t.to(device), class_idx=torch.tensor([pred], device=device))
            heat = cam_to_numpy(heat_t, (224, 224))[0]
            overlay = overlay_heatmap(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), heat, alpha)
            col1, col2 = st.columns(2)
            with col1:
                st.image(x, caption="Preprocessed", use_column_width=True)
            with col2:
                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
        except Exception as e:
            st.info("Grad-CAM unavailable for this model.")
