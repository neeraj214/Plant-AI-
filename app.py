import os
import io
import json
import asyncio
import numpy as np
import cv2
import tensorflow as tf
import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from src.data_utils import load_aliases
from src.torch_model import PlantClassifier

app = FastAPI()

class_names = None
labels_path = "data/labels.json"
if os.path.isfile(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        lbls = json.load(f)
        inv = {v: k for k, v in lbls.items()}
        class_names = [inv[i] for i in range(len(inv))]

BACKEND = os.getenv("MODEL_BACKEND", "tflite")
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
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def load_torch():
    global torch_model
    if not os.path.isfile(TORCH_PATH):
        return
    n = len(class_names) if class_names else 38
    m = PlantClassifier(num_classes=n, backbone=BACKBONE)
    sd = torch.load(TORCH_PATH, map_location=device)
    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    torch_model = m

if BACKEND == "tflite" and os.path.isfile(TFLITE_PATH):
    load_tflite()
elif os.path.isfile(TORCH_PATH):
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
    return {"class_index": p, "class_name": name, "confidence": conf}
