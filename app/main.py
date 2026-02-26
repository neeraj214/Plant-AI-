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

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'last_upload' not in st.session_state:
    st.session_state.last_upload = None

# Reset analyzed state if a new file is uploaded
def on_upload_change():
    st.session_state.analyzed = False

# --- CUSTOM CSS (GLASSMORPHISM DARK THEME) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
      --bg-gradient: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
      --accent: #22C55E;
      --accent-glow: rgba(34, 197, 94, 0.4);
      --card-bg: rgba(255, 255, 255, 0.05);
      --card-border: rgba(255, 255, 255, 0.1);
      --text-primary: #FFFFFF;
      --text-secondary: #94A3B8;
      --glass-blur: blur(12px);
    }

    /* Global Overrides */
    .stApp {
      background: var(--bg-gradient) fixed;
      color: var(--text-primary);
      font-family: 'Inter', sans-serif;
    }
    
    .block-container {
      padding-top: 2rem !important;
      max-width: 1400px;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
      background: rgba(15, 32, 39, 0.7) !important;
      backdrop-filter: var(--glass-blur);
      border-right: 1px solid var(--card-border);
    }
    
    [data-testid="stSidebar"] h2 {
      color: var(--text-primary);
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    /* Glass Cards */
    .glass-card {
      background: var(--card-bg);
      backdrop-filter: var(--glass-blur);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 24px;
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
      transition: transform 0.3s ease, box-shadow 0.3s ease;
      margin-bottom: 20px;
    }
    
    .glass-card:hover {
      box-shadow: 0 8px 32px 0 rgba(34, 197, 94, 0.15);
      border-color: rgba(34, 197, 94, 0.3);
    }

    /* Header Styling */
    .logo-container {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }
    .logo-text {
      font-size: 28px;
      font-weight: 800;
      background: linear-gradient(to right, #FFFFFF, #22C55E);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .subtitle {
      color: var(--text-secondary);
      font-size: 14px;
      font-weight: 500;
      margin-bottom: 32px;
      letter-spacing: 0.05em;
    }

    /* File Uploader Customization */
    [data-testid="stFileUploader"] {
      padding: 0;
    }
    [data-testid="stFileUploaderDropzone"] {
      background: rgba(255, 255, 255, 0.02) !important;
      border: 2px dashed var(--card-border) !important;
      border-radius: 16px !important;
      padding: 40px !important;
      transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color: var(--accent) !important;
      background: rgba(34, 197, 94, 0.05) !important;
      box-shadow: 0 0 20px var(--accent-glow) !important;
    }
    
    /* Buttons */
    .stButton > button {
      width: 100%;
      background: linear-gradient(135deg, #22C55E, #16A34A) !important;
      color: white !important;
      border: none !important;
      border-radius: 12px !important;
      padding: 12px 24px !important;
      font-weight: 700 !important;
      font-size: 16px !important;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      transition: all 0.3s ease !important;
      box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3) !important;
    }
    .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 25px var(--accent-glow) !important;
    }
    
    .secondary-btn > button {
      background: rgba(255, 255, 255, 0.05) !important;
      border: 1px solid var(--card-border) !important;
      box-shadow: none !important;
    }
    .secondary-btn > button:hover {
      background: rgba(255, 255, 255, 0.1) !important;
      border-color: var(--text-secondary) !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
      background-color: var(--accent) !important;
    }
    
    /* Labels and Titles */
    .card-title {
      color: var(--text-primary);
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .diagnosis-name {
      font-size: 38px;
      font-weight: 800;
      color: var(--text-primary);
      margin: 4px 0;
      letter-spacing: -0.02em;
      text-shadow: 0 0 20px rgba(255,255,255,0.1);
    }
    
    .confidence-text {
      color: var(--accent);
      font-size: 42px;
      font-weight: 800;
      letter-spacing: -0.02em;
      text-shadow: 0 0 20px var(--accent-glow);
    }
    
    .severity-badge {
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: inline-block;
      margin-bottom: 12px;
    }
    .severity-low { background: rgba(234, 179, 8, 0.2); color: #EAB308; border: 1px solid rgba(234, 179, 8, 0.3); }
    .severity-mid { background: rgba(249, 115, 22, 0.2); color: #F97316; border: 1px solid rgba(249, 115, 22, 0.3); }
    .severity-high { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    .img-container {
      aspect-ratio: 1 / 1;
      width: 100%;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--card-border);
      background: rgba(0,0,0,0.2);
    }
    .img-container img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    
    .info-text {
      color: var(--text-secondary);
      font-size: 14px;
      line-height: 1.6;
    }

    /* History Items */
    .history-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.03);
      margin-bottom: 8px;
      border: 1px solid transparent;
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .history-item:hover {
      background: rgba(255, 255, 255, 0.07);
      border-color: var(--card-border);
      transform: translateX(4px);
    }
    .history-thumb {
      width: 40px;
      height: 40px;
      border-radius: 8px;
      object-fit: cover;
    }
    .history-info {
      flex: 1;
      overflow: hidden;
    }
    .history-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .history-time {
      font-size: 11px;
      color: var(--text-secondary);
    }

    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
      animation: fadeInUp 0.6s ease-out forwards;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR (HISTORY) ---
with st.sidebar:
    st.markdown("## 🕒 Analysis History")
    if not st.session_state.history:
        st.markdown('<p class="info-text">No previous scans found. Start by uploading an image.</p>', unsafe_allow_html=True)
    else:
        for idx, item in enumerate(reversed(st.session_state.history[-10:])):
            st.markdown(f"""
                <div class="history-item">
                    <img src="{item['thumb']}" class="history-thumb">
                    <div class="history-info">
                        <div class="history-label">{item['label']}</div>
                        <div class="history-time">{item['time']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
# Header
st.markdown("""
    <div class="animate-in">
        <div class="logo-container">
            <span style="font-size: 32px;">🌿</span>
            <span class="logo-text">Plant AI</span>
        </div>
        <div class="subtitle">EFFICIENTNETV2 • GRAD-CAM ENABLED • REAL-TIME DIAGNOSIS</div>
    </div>
""", unsafe_allow_html=True)

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

# Two-Card Dashboard Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
        <div class="glass-card animate-in">
            <div class="card-title">📤 Upload Leaf Photo</div>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", on_change=on_upload_change)
    
    if uploaded:
        # Check if it's a new upload to reset state
        if st.session_state.last_upload != uploaded.name:
            st.session_state.last_upload = uploaded.name
            st.session_state.analyzed = False
            
        st.markdown('<div class="stButton">', unsafe_allow_html=True)
        if st.button("IDENTIFY DISEASE"):
            st.session_state.analyzed = True
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="info-text" style="text-align:center;">Select a clear image of the affected plant leaf for the most accurate AI diagnosis.</p>', unsafe_allow_html=True)

with col_right:
    st.markdown("""
        <div class="glass-card animate-in">
            <div class="card-title">🔍 Diagnostic Results</div>
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.analyzed or uploaded is None:
        st.markdown(f"""
            <div style="height: 300px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,255,255,0.02); border-radius: 16px; border: 1px dashed var(--card-border);">
                <span style="font-size: 48px; opacity: 0.2;">{"📈" if uploaded else "📊"}</span>
                <p style="color: var(--text-secondary); margin-top: 16px;">{"Click Identify to see results" if uploaded else "Results will appear here after upload"}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # (This part stays the same but is now inside the if analyzed)
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)
        
        # Preprocessing
        if CV2_AVAILABLE:
            x = cv2.resize(img_np, (224, 224))
        else:
            x = np.array(Image.fromarray(img_np).resize((224,224)))
        
        mean, std = load_stats()
        t = (x.astype(np.float32)/255.0 - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        
        # Prediction Logic
        name_v = "Analyzing..."
        conf_v = 0.0
        top3_data = None
        grad_overlay = None
        
        if model is not None and TORCH_AVAILABLE:
            t_torch = torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(t_torch)
                probs = torch.softmax(logits, dim=1)[0]
                pred = int(torch.argmax(probs).item())
                conf_v = float(probs[pred].item())
                name_v = class_names[pred] if class_names else str(pred)
                
                # Top-3
                if class_names:
                    pk = torch.topk(probs, k=min(3, len(class_names)))
                    top3_data = {class_names[i]: float(v) for i, v in zip(pk.indices.tolist(), pk.values.tolist())}
            
            # Grad-CAM
            try:
                cam = GradCAM(model)
                heat_t = cam(t_torch, class_idx=torch.tensor([pred], device=device))
                heat = cam_to_numpy(heat_t, (224, 224))[0] if 'cam_to_numpy' in globals() else None
                if heat is not None and CV2_AVAILABLE:
                    grad_overlay = overlay_heatmap(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), heat, 0.5)
            except: pass
        elif requests is not None:
            try:
                r = requests.post(f"{API_BASE}/predict", files={"file": uploaded.getvalue()}, timeout=10)
                if r.ok:
                    data = r.json()
                    name_v = data.get('class_name', "Unknown")
                    conf_v = float(data.get('confidence', 0.0))
                    pth = data.get("gradcam_overlay_path")
                    if pth:
                        grad_overlay = pth if pth.startswith("http") else f"{API_BASE}{pth}"
            except: pass

        # Display Results Side-by-Side
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown('<p class="info-text" style="text-align:center; margin-bottom:8px;">ORIGINAL INPUT</p>', unsafe_allow_html=True)
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with res_col2:
            st.markdown('<p class="info-text" style="text-align:center; margin-bottom:8px;">AI ATTENTION MAP</p>', unsafe_allow_html=True)
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            if grad_overlay is not None:
                st.image(grad_overlay, use_container_width=True)
            else:
                st.markdown('<div style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,0.05);">No Map Available</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<p class="info-text" style="margin-top: 16px; text-align: center; font-size: 12px; opacity: 0.8;">The attention map highlights leaf regions prioritized by the EfficientNetV2 model for classification.</p>', unsafe_allow_html=True)

# Diagnosis Section (Below the cards)
if uploaded is not None and st.session_state.analyzed:
    # Determine severity
    sev_class = "severity-high" if conf_v > 0.8 else "severity-mid" if conf_v > 0.5 else "severity-low"
    sev_label = "High Confidence" if conf_v > 0.8 else "Moderate Risk" if conf_v > 0.5 else "Uncertain"

    st.markdown('<div class="glass-card animate-in" style="margin-top: 24px;">', unsafe_allow_html=True)
    st.markdown('<div style="border-bottom: 1px solid var(--card-border); margin-bottom: 24px; padding-bottom: 12px;"><p class="info-text" style="margin:0; font-weight:700; letter-spacing:0.1em; color:var(--accent);">AI SYSTEM DIAGNOSIS</p></div>', unsafe_allow_html=True)
    
    diag_col1, diag_col2 = st.columns([1.5, 1])
    
    with diag_col1:
        st.markdown(f'<div class="severity-badge {sev_class}">{sev_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="diagnosis-name">{name_v}</div>', unsafe_allow_html=True)
        st.markdown('<p class="info-text" style="margin-top: 8px; max-width: 90%;">Our deep learning model has identified markers consistent with the disease listed above. Please consult an agronomist for treatment protocols.</p>', unsafe_allow_html=True)
        
        # Add to history if not already there
        from datetime import datetime
        history_entry = {
            "label": name_v,
            "time": datetime.now().strftime("%H:%M"),
            "thumb": "https://api.dicebear.com/7.x/identicon/svg?seed=" + name_v
        }
        if not st.session_state.history or st.session_state.history[-1]["label"] != name_v:
             st.session_state.history.append(history_entry)
             
    with diag_col2:
        st.markdown('<div style="background: rgba(255,255,255,0.02); padding: 20px; border-radius: 16px; border: 1px solid var(--card-border);">', unsafe_allow_html=True)
        st.markdown('<p class="info-text" style="margin-bottom:0; font-weight:600;">DETECTION CONFIDENCE</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-text">{conf_v*100:.2f}%</div>', unsafe_allow_html=True)
        st.progress(conf_v)
        st.markdown('</div>', unsafe_allow_html=True)

    # Top-3 Visualization
    if top3_data:
        st.markdown('<div style="margin-top: 32px;">', unsafe_allow_html=True)
        st.markdown('<p class="info-text" style="margin-bottom: 16px; font-weight:700;">PROBABILITY DISTRIBUTION (TOP 3)</p>', unsafe_allow_html=True)
        
        t3_col1, t3_col2 = st.columns(2)
        
        items = list(top3_data.items())
        for i, (label, prob) in enumerate(items):
            target_col = t3_col1 if i < 2 else t3_col2
            with target_col:
                st.markdown(f"""
                    <div style="margin-bottom: 16px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span style="font-size:13px; font-weight:600; color:var(--text-primary);">{i+1}. {label}</span>
                            <span style="font-size:13px; color:var(--accent); font-weight:700;">{prob*100:.2f}%</span>
                        </div>
                        <div style="height:6px; width:100%; background:rgba(255,255,255,0.05); border-radius:3px;">
                            <div style="height:100%; width:{prob*100}%; background:var(--accent); border-radius:3px; box-shadow: 0 0 10px var(--accent-glow);"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="display: flex; gap: 16px; margin-top: 32px; border-top: 1px solid var(--card-border); padding-top: 24px;">', unsafe_allow_html=True)
    btn_col1, btn_col2, _ = st.columns([1, 1, 1.5])
    with btn_col1:
        st.button("📄 GENERATE PDF REPORT")
    with btn_col2:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        st.button("🔗 EXPORT DIAGNOSIS")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

