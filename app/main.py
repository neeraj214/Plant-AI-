import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from src.model import build_model
from src.interpret import gradcam_heatmap
import os

st.set_page_config(page_title="Am I Healthy?", page_icon="🌿", layout="centered")
st.title("Am I Healthy? Plant Disease Identifier")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_column_width=True)
    arr = np.array(image.resize((224, 224)), dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    x = np.expand_dims(arr, axis=0)
    model_path = os.path.join("models", "model.h5")
    if os.path.exists(model_path):
        model, last_conv = build_model()
        model = tf.keras.models.load_model(model_path)
        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        st.write(f"Predicted class index: {idx} (confidence {conf:.2f})")
        try:
            heat = gradcam_heatmap(x, model, last_conv)
            st.image(heat, caption="Grad-CAM", use_column_width=True)
        except Exception as e:
            st.write("Grad-CAM unavailable")
    else:
        st.write("Model not found")
