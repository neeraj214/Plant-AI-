import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import numpy as np
from PIL import Image
import json
import os
TORCH_AVAILABLE = True
CV2_AVAILABLE = True
API_BASE = os.getenv("STREAMLIT_API_BASE", "http://localhost:8000")
try:
    import requests
except Exception:
    requests = None
try:
    import torch
except Exception:
    TORCH_AVAILABLE = False
try:
    import cv2
except Exception:
    CV2_AVAILABLE = False
from src.torch_dataset import load_stats

st.set_page_config(page_title="Am I Healthy?", page_icon="🌿", layout="centered")
st.title("Am I Healthy? Plant Disease Identifier")
st.markdown(
    "<style>.css-1v0mbdj {max-width: 1100px;margin: auto;} .stButton>button{border-radius:8px;padding:0.6rem 1rem;} .uploadedFile{border-radius:12px;} .block-container{padding-top:1rem;}</style>",
    unsafe_allow_html=True,
)

labels_path = os.path.join("data", "labels.json")
class_names = None
if os.path.isfile(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        lbls = json.load(f)
        inv = {v: k for k, v in lbls.items()}
        class_names = [inv[i] for i in range(len(inv))]

model = None
device = None
if TORCH_AVAILABLE:
    from src.torch_model import PlantClassifier
    from src.torch_cam import GradCAM, cam_to_numpy, overlay_heatmap
    weights_path = os.path.join("models", "swa.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isfile(weights_path):
        try:
            n_classes = len(class_names) if class_names else 38
            model = PlantClassifier(num_classes=n_classes, pretrained=False)
            sd = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(sd, strict=False)
            model = model.to(device).eval()
        except Exception:
            model = None

with st.sidebar:
    st.header("Controls")
    alpha = st.slider("Grad-CAM overlay strength", 0.1, 0.9, 0.45, 0.05)
    show_topk = st.checkbox("Show top-3 predictions", value=True)
    st.divider()
    st.caption("Backend")
    st.write("PyTorch" + (" ✅" if TORCH_AVAILABLE else " ❌"))
    st.write(("Weights: models/swa.pth ✅" if model is not None else "Weights missing or failed to load"))

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)
    img_np = np.array(image)
    if CV2_AVAILABLE:
        x = cv2.resize(img_np, (224, 224))
    else:
        x = np.array(Image.fromarray(img_np).resize((224,224)))
    mean, std = load_stats()
    t = (x.astype(np.float32)/255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    if TORCH_AVAILABLE:
        t_torch = torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0)
    else:
        t_torch = None
    if model is None or not TORCH_AVAILABLE:
        if requests is not None:
            try:
                b = uploaded.getvalue()
                r = requests.post(f"{API_BASE}/predict", files={"file": ("upload.png", b, "image/png")}, timeout=30)
                if r.ok:
                    data = r.json()
                    if "error" in data:
                        st.warning(f"API error: {data['error']}")
                    else:
                        st.subheader(f"Prediction: {data.get('class_name', data.get('class_index'))}  •  Confidence: {float(data.get('confidence', 0.0)):.2f}")
                        p = data.get("gradcam_overlay_path")
                        if isinstance(p, str):
                            url = p if p.startswith("http") else f"{API_BASE}{p}"
                            st.image(url, caption="Grad-CAM Overlay", use_container_width=True)
                else:
                    st.warning("API call failed")
            except Exception:
                st.warning("Model backend unavailable. Provide models/swa.pth and ensure PyTorch is installed, or run the API.")
        else:
            st.warning("Model backend unavailable. Provide models/swa.pth and ensure PyTorch is installed.")
    else:
        with torch.no_grad():
            logits = model(t_torch.to(device))
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
        name = class_names[pred] if class_names and 0 <= pred < len(class_names) else str(pred)
        st.subheader(f"Prediction: {name}  •  Confidence: {conf:.2f}")
        if show_topk and class_names is not None:
            topk = torch.topk(probs, k=min(3, len(class_names)))
            labels_k = [class_names[i] for i in topk.indices.tolist()]
            vals = [float(v) for v in topk.values.tolist()]
            st.write({a: round(b, 4) for a, b in zip(labels_k, vals)})
        try:
            cam = GradCAM(model)
            heat_t = cam(t_torch.to(device), class_idx=torch.tensor([pred], device=device))
            heat = None
            if 'cam_to_numpy' in globals():
                heat = cam_to_numpy(heat_t, (224, 224))[0]
            else:
                h = heat_t[0].detach().cpu().numpy()
                h = (h - h.min()) / (h.max() - h.min() + 1e-6)
                heat = h
            if CV2_AVAILABLE:
                overlay = overlay_heatmap(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), heat, alpha)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(x, caption="Preprocessed", use_container_width=True)
                with col2:
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            else:
                st.image((heat*255).astype(np.uint8), caption="Grad-CAM", use_container_width=True)
        except Exception:
            st.info("Grad-CAM unavailable for this model.")
