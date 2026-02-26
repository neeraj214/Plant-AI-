# Plant AI — Disease Identifier

Plant AI is an end‑to‑end system to classify plant diseases from leaf images and explain predictions with Grad‑CAM. It includes:
- Data engineering utilities and Albumentations pipeline
- PyTorch model (EfficientNetV2) with Grad‑CAM
- FastAPI backend with optional TFLite/torch mode
- Streamlit UI with live overlays
- Optional React web UI (Vite)
- Benchmarking and CI

## Quick Start (Torch Backend)

Recommended Python: 3.10–3.12

1) Install core dependencies (Torch backend only)

```bash
python -m pip install --upgrade pip
python -m pip install streamlit opencv-python numpy Pillow pandas scikit-learn torch torchvision timm fastapi uvicorn[standard] python-multipart aiofiles requests lime scikit-image
```

2) Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
# Health: http://localhost:8000/health
# Docs:   http://localhost:8000/docs
```

3) Start the Streamlit UI

```bash
streamlit run app/main.py
# UI:     http://localhost:8501/
```

4) Add your model
- Place a trained checkpoint at `models/swa.pth` for meaningful predictions.
- Without it, the API loads an untrained fallback model and returns dummy predictions.

## Optional: TFLite Backend

TensorFlow is currently not available for Python 3.14+. If you need TFLite:
- Use Python 3.10–3.12, then:

```bash
python -m pip install "tensorflow>=2.15,<2.16"
python tflite_convert.py --model-path path/to/saved_model --rep-dataset-path path/to/rep_ds --out models/model_quantized.tflite
$env:MODEL_BACKEND="tflite"; uvicorn app:app --host 0.0.0.0 --port 8000
```

## Optional: React Frontend

Requires Node.js ≥ 18 (local machine).

```bash
cd web
npm install
npm run dev      # http://localhost:5173/
# Production build served by API at /ui
npm run build    # then open http://localhost:8000/ui/
```

## API

- `POST /predict` — multipart form upload `file` (image)
  - Response: `{ class_index, class_name, confidence, gradcam_overlay_path? }`
- `GET /overlay/{name}` — returns saved Grad‑CAM overlay image
- `GET /health` — health and selected backend

## Troubleshooting

- Streamlit warns about `use_column_width`: fixed in code; use `use_container_width`.
- API returns `no_model`: add `models/swa.pth` or run in TFLite mode with `models/model_quantized.tflite`.
- Albumentations missing: handled via lazy import in `src/torch_dataset.py`. Install with `pip install albumentations` if you train locally.
- TensorFlow install fails on Python 3.14+: use the Torch backend or Python 3.10–3.12 for TFLite.

## Repo Map (key files)

- `app.py` — FastAPI backend (CORS, Grad‑CAM overlay endpoint, serves `/ui` if built)
- `app/main.py` — Streamlit UI with live Grad‑CAM
- `web/` — React single‑page app (Vite)
- `src/` — PyTorch model, dataset, and Grad‑CAM utilities
- `benchmark.py` — Compare FP32 vs INT8 size/time/accuracy
- `report_failures.py` — Misclassification grids (Grad‑CAM + occlusion)
