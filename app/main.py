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

st.set_page_config(page_title="Plant AI — Disease Identifier", page_icon="🌱", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    :root{
      --bg:#F8FAFC; --card:#FFFFFF; --green:#16A34A; --blue:#2563EB;
      --text:#0F172A; --muted:#64748B; --border:#E2E8F0;
    }
    .stApp{background:var(--bg); color:var(--text); font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;}
    .block-container{padding-top:90px;max-width:1200px}
    .top-nav{
      position:sticky;top:0;z-index:50;height:70px;display:flex;align-items:center;justify-content:space-between;
      padding:0 20px;background:#fff;border-bottom:1px solid var(--border);box-shadow:0 2px 8px rgba(15,23,42,0.05)
    }
    .brand{font-weight:800;display:flex;align-items:center;gap:10px}
    .tag{color:var(--muted);font-weight:600}
    .links .btn{
      display:inline-block;margin-left:12px;padding:8px 12px;border-radius:10px;border:1px solid var(--border);
      color:var(--text);text-decoration:none;background:#fff;transition:all .15s ease
    }
    .links .btn:hover{box-shadow:0 4px 12px rgba(15,23,42,0.08)}
    h1.hero{font-size:34px;font-weight:800;margin:8px 0 0}
    p.sub{color:var(--muted);margin:4px 0 20px}
    /* File uploader styling */
    [data-testid="stFileUploaderDropzone"]{
      border:2px dashed var(--border); background:#fff; border-radius:16px; padding:28px; transition:border-color .15s, box-shadow .15s;
    }
    [data-testid="stFileUploaderDropzone"]:hover{border-color:var(--green); box-shadow:0 0 0 6px rgba(22,163,74,0.08)}
    /* Cards, frames */
    .card{background:var(--card); border:1px solid var(--border); border-radius:16px; padding:20px; box-shadow:0 8px 24px rgba(15,23,42,0.06)}
    .label{color:var(--muted);margin:6px 0}
    .frame{background:#fff;border:1px solid var(--border);border-radius:14px;overflow:hidden}
    /* Progress */
    .bar{width:100%;height:10px;background:#EEF2F6;border-radius:999px;overflow:hidden}
    .fill{height:100%;background:linear-gradient(90deg,var(--green),var(--blue));width:0}
    .badge{padding:4px 10px;border-radius:999px;font-weight:700}
    .low{background:rgba(22,163,74,0.12);color:#15803D}
    .mid{background:rgba(37,99,235,0.12);color:#2563EB}
    .high{background:rgba(239,68,68,0.12);color:#DC2626}
    /* Global image fade-in */
    .stImage img{animation:fadein .25s ease-out}
    @keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
    /* Buttons */
    .stButton>button{
      background:linear-gradient(135deg,var(--green),var(--blue));color:#fff;border:none;border-radius:12px;padding:10px 16px;font-weight:700
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="top-nav">
      <div class="brand">🌱 Plant AI <span class="tag">AI‑Powered Disease Detection with Explainability</span></div>
      <div class="links">
        <a class="btn" href="https://github.com/neeraj214/Plant-AI-" target="_blank">GitHub</a>
        <a class="btn" href="{API_BASE}/docs" target="_blank">API Docs</a>
      </div>
    </div>
    """,
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

c0, c1, c2 = st.columns([1,2,1])
with c1:
    st.markdown('<h1 class="hero">Detect Plant Diseases Instantly</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Upload a leaf image and visualize AI attention maps.</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    uploaded = st.file_uploader("Drag & Drop Leaf Image • Supported: JPG, PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
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
                        conf_v = float(data.get('confidence', 0.0))
                        name_v = data.get('class_name', data.get('class_index'))
                        colA, colB = st.columns(2)
                        with colA:
                            st.markdown('<div class="label">Input</div>', unsafe_allow_html=True)
                            st.image(image, use_container_width=True)
                        with colB:
                            st.markdown('<div class="label">AI Attention Map</div>', unsafe_allow_html=True)
                            alpha = st.slider("Overlay intensity", 0.0, 1.0, 0.5, 0.01)
                            pth = data.get("gradcam_overlay_path")
                            if isinstance(pth, str):
                                url = pth if pth.startswith("http") else f"{API_BASE}{pth}"
                                st.image(url, use_container_width=True)
                        st.markdown(f"### {name_v}")
                        badge = "high" if conf_v>=0.75 else "mid" if conf_v>=0.4 else "low"
                        st.markdown(f'<span class="badge {badge}">{"High Risk" if badge=="high" else "Medium Risk" if badge=="mid" else "Low Risk"}</span>', unsafe_allow_html=True)
                        st.markdown('<div class="bar"><div class="fill" id="confbar"></div></div>', unsafe_allow_html=True)
                        st.markdown(f"<script>document.getElementById('confbar').style.width='{conf_v*100:.1f}%'</script>", unsafe_allow_html=True)
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
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="label">Input</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        with col2:
            st.markdown('<div class="label">AI Attention Map</div>', unsafe_allow_html=True)
            alpha = st.slider("Overlay intensity", 0.0, 1.0, 0.5, 0.01)
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
                st.image(overlay, use_container_width=True)
            else:
                st.image((heat*255).astype(np.uint8), use_container_width=True)
        except Exception:
            st.info("Grad-CAM unavailable for this model.")
        st.markdown(f"### {name}")
        badge = "high" if conf>=0.75 else "mid" if conf>=0.4 else "low"
        st.markdown(f'<span class="badge {badge}">{"High Risk" if badge=="high" else "Medium Risk" if badge=="mid" else "Low Risk"}</span>', unsafe_allow_html=True)
        st.markdown('<div class="bar"><div class="fill" id="confbar2"></div></div>', unsafe_allow_html=True)
        st.markdown(f"<script>document.getElementById('confbar2').style.width='{conf*100:.1f}%'</script>", unsafe_allow_html=True)
        with st.expander("Model & Explainability Details"):
            st.write("Backend: PyTorch" if TORCH_AVAILABLE else "Backend: API")
            st.write("Model: EfficientNetV2")
            if class_names is not None:
                topk = torch.topk(probs, k=min(3, len(class_names)))
                labels_k = [class_names[i] for i in topk.indices.tolist()]
                vals = [float(v) for v in topk.values.tolist()]
                st.write("Top‑3:", {a: round(b, 4) for a, b in zip(labels_k, vals)})
