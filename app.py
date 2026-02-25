import os
import io
import json
import uuid
import asyncio
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.torch_model import PlantClassifier
from src.torch_cam import GradCAM, cam_to_numpy, overlay_heatmap

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built React app if present
if os.path.isdir(os.path.join("web", "dist")):
    app.mount("/ui", StaticFiles(directory=os.path.join("web", "dist"), html=True), name="ui")

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Plant AI API",
        "links": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict",
            "ui": "/ui" if os.path.isdir(os.path.join("web", "dist")) else None
        },
        "backend": BACKEND
    }

class_names = None
labels_path = "data/labels.json"
if os.path.isfile(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        lbls = json.load(f)
        inv = {v: k for k, v in lbls.items()}
        class_names = [inv[i] for i in range(len(inv))]

BACKEND = os.getenv("MODEL_BACKEND", "torch")
TFLITE_PATH = os.getenv("TFLITE_PATH", "models/model_quantized.tflite")
TORCH_PATH = os.getenv("TORCH_PATH", "models/swa.pth")
BACKBONE = os.getenv("BACKBONE", "efficientnetv2_rw_s")

interpreter = None
input_details = None
output_details = None

torch_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tflite():
    global interpreter, input_details, output_details
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def load_torch():
    global torch_model
    n = len(class_names) if class_names else 38
    m = PlantClassifier(num_classes=n, backbone=BACKBONE, pretrained=False)
    if os.path.isfile(TORCH_PATH):
        sd = torch.load(TORCH_PATH, map_location=device)
        m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    torch_model = m

if BACKEND == "tflite" and os.path.isfile(TFLITE_PATH):
    load_tflite()
else:
    BACKEND = "torch"
    load_torch()

def preprocess_image(b):
    img = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (224,224)).astype(np.float32)/255.0
    return img, x

def predict_tflite(x):
    import tensorflow as tf
    inp = input_details[0]
    out = output_details[0]
    scale, zp = inp["quantization"]
    if scale and zp is not None and inp["dtype"] == np.int8:
        q = np.clip(np.round(x/scale + zp), -128, 127).astype(np.int8)
    else:
        q = x.astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], np.expand_dims(q, 0))
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])
    if out["dtype"] == np.int8:
        s, z = out["quantization"]
        y = s * (y.astype(np.float32) - z)
    y = y.squeeze()
    p = int(np.argmax(y))
    conf = float(np.max(tf.nn.softmax(y)))
    return p, conf

def predict_torch(x):
    with torch.no_grad():
        t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(device)
        logits = torch_model(t)
        p = int(torch.argmax(logits, dim=1).item())
        conf = float(torch.softmax(logits, dim=1)[0, p].item())
    return p, conf

def gradcam_overlay_for_torch(x, class_idx):
    # x: HWC float32 0..1, RGB resized 224x224
    t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(device)
    cam = GradCAM(torch_model)
    with torch.no_grad():
        pass
    heat_t = cam(t, class_idx=torch.tensor([class_idx], device=device))
    heat = cam_to_numpy(heat_t, (224,224))[0]
    bgr = cv2.cvtColor((x*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(bgr, heat, alpha=0.45)
    os.makedirs("temp", exist_ok=True)
    name = f"{uuid.uuid4().hex}.png"
    path = os.path.join("temp", name)
    cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return f"/overlay/{name}"

@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    b = await file.read()
    img, x = preprocess_image(b)
    if x is None:
        return {"error": "invalid_image"}
    if BACKEND == "tflite" and interpreter is not None:
        p, conf = predict_tflite(x)
    elif torch_model is not None:
        p, conf = predict_torch(x)
    else:
        return {"error": "no_model"}
    name = class_names[p] if class_names and 0 <= p < len(class_names) else str(p)
    resp = {"class_index": p, "class_name": name, "confidence": conf}
    if BACKEND == "torch" and torch_model is not None:
        try:
            overlay_path = gradcam_overlay_for_torch(x, p)
            resp["gradcam_overlay_path"] = overlay_path
        except Exception:
            pass
    return resp

@app.get("/overlay/{name}")
async def overlay(name: str):
    p = os.path.join("temp", name)
    if not os.path.isfile(p):
        return {"error": "not_found"}
    return FileResponse(p, media_type="image/png")
