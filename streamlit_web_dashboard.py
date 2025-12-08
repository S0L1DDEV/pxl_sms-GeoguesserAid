"""
Streamlit Web Dashboard for Dual-Model Image Classification

Run: streamlit run streamlit_web_dashboard.py

Features
- Load both models (SIGLIP checkpoint and GeoGuessr-55)
- Live local screenshot mode (requires running locally with a desktop)
- Upload image mode
- Display prediction lists + bar charts side-by-side
- Simple caching so models only load once

Requirements
- streamlit
- torch
- transformers
- pillow
- pyautogui (for local screenshot mode)

Install example:
pip install streamlit torch transformers pillow pyautogui

Note: If you run on a machine without GPU, models will fall back to CPU automatically.
"""

import streamlit as st
from PIL import Image
import io
import time
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, SiglipForImageClassification
import pyautogui
import numpy as np

# -------------------------
# Config / constants
# -------------------------
BASE_MODEL = "google/siglip-base-patch16-224"
MODEL_A_PATH = "./checkpoints/checkpoint-28130"  # adapt path if needed
MODEL_B_NAME = "prithivMLmods/GeoGuessr-55"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Mapping for model B labels (GeoGuessr-55)
id2label_B = {
    "0": "Argentina","1": "Australia","2": "Austria","3": "Bangladesh","4": "Belgium",
    "5": "Bolivia","6": "Botswana","7": "Brazil","8": "Bulgaria","9": "Cambodia",
    "10": "Canada","11": "Chile","12": "Colombia","13": "Croatia","14": "Czechia",
    "15": "Denmark","16": "Finland","17": "France","18": "Germany","19": "Ghana",
    "20": "Greece","21": "Hungary","22": "India","23": "Indonesia","24": "Ireland",
    "25": "Israel","26": "Italy","27": "Japan","28": "Kenya","29": "Latvia",
    "30": "Lithuania","31": "Malaysia","32": "Mexico","33": "Netherlands",
    "34": "New Zealand","35": "Nigeria","36": "Norway","37": "Peru","38": "Philippines",
    "39": "Poland","40": "Portugal","41": "Romania","42": "Russia","43": "Singapore",
    "44": "Slovakia","45": "South Africa","46": "South Korea","47": "Spain","48": "Sweden",
    "49": "Switzerland","50": "Taiwan","51": "Thailand","52": "Turkey","53": "Ukraine",
    "54": "United Kingdom"
}

# -------------------------
# Helpers: load models once
# -------------------------
@st.cache_resource
def load_model_A():
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageClassification.from_pretrained(MODEL_A_PATH).to(device)
    return processor, model

@st.cache_resource
def load_model_B():
    processor = AutoImageProcessor.from_pretrained(MODEL_B_NAME)
    model = SiglipForImageClassification.from_pretrained(MODEL_B_NAME).to(device)
    return processor, model

processor_A, model_A = load_model_A()
processor_B, model_B = load_model_B()

# -------------------------
# Prediction functions
# -------------------------

def predict_A(img: Image.Image, topk=5):
    inputs = processor_A(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model_A(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    topk_idx = probs.argsort()[-topk:][::-1]
    return [(model_A.config.id2label[int(i)], float(probs[int(i)] * 100.0)) for i in topk_idx]


def predict_B(img: Image.Image, topk=5):
    inputs = processor_B(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model_B(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    topk_idx = probs.argsort()[-topk:][::-1]
    return [(id2label_B[str(int(i))], float(probs[int(i)] * 100.0)) for i in topk_idx]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Dual Model Image Dashboard", layout="wide")

st.title("Dual-Model Image Classification Dashboard")
st.write("Live demo of your SIGLIP checkpoint (Model A) and GeoGuessr-55 (Model B).")

# Sidebar controls
mode = st.sidebar.selectbox("Input mode", ["Upload image", "Live screenshot (local)", "Example image URL"])

if mode == "Live screenshot (local)":
    st.sidebar.info("This captures a screenshot from your local machine using pyautogui. Run locally.")
    interval = st.sidebar.slider("Screenshot interval (s)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
else:
    interval = None

crop_area = st.sidebar.checkbox("Crop center region (useful for screenshots)", value=False)

# Main area: two columns for models
col_image, col_models = st.columns([1, 1.2])

with col_image:
    st.subheader("Input Image")

    if mode == "Upload image":
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        img = None
        if uploaded is not None:
            img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    elif mode == "Example image URL":
        url = st.text_input("Image URL (http/https)")
        img = None
        if url:
            try:
                import requests
                resp = requests.get(url, timeout=10)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                st.sidebar.error(f"Error fetching image: {e}")
    else:  # Live screenshot
        st.info("Click \"Start live\" to begin taking local screenshots.")
        start = st.button("Start live")
        stop = st.button("Stop live")
        img = None

    # Placeholder for image display
    img_display = st.empty()

with col_models:
    st.subheader("Predictions")
    cols = st.columns(2)
    # Create empty containers that will be reused
    placeholder_A = cols[0].empty()
    placeholder_B = cols[1].empty()

# Function to render predictions

def render_predictions(img: Image.Image):
    # optional crop
    display_img = img
    if crop_area:
        w, h = img.size
        cw, ch = int(w * 0.6), int(h * 0.6)
        left = (w - cw) // 2
        top = (h - ch) // 2
        display_img = img.crop((left, top, left + cw, top + ch))

    # show image - FIXED: use_container_width instead of use_column_width
    img_display.image(display_img, use_container_width=True)

    # get predictions
    predsA = predict_A(display_img, topk=5)
    predsB = predict_B(display_img, topk=5)

    # Render Model A - FIXED: Use .container() to replace content instead of appending
    with placeholder_A.container():
        st.markdown("**SIGLIP Checkpoint (Model A)**")
        for label, score in predsA:
            st.write(f"{label} — {score:.1f}%")
        # bar chart
        scoresA = np.array([s for _, s in predsA])
        labelsA = [l for l, _ in predsA]
        st.bar_chart(data={'score': scoresA}, height=180)

    # Render Model B - FIXED: Use .container() to replace content instead of appending
    with placeholder_B.container():
        st.markdown("**GeoGuessr-55 (Model B)**")
        for label, score in predsB:
            st.write(f"{label} — {score:.1f}%")
        scoresB = np.array([s for _, s in predsB])
        labelsB = [l for l, _ in predsB]
        st.bar_chart(data={'score': scoresB}, height=180)

# Run loop for live mode
if mode == "Live screenshot (local)":
    live_running = False
    # Simple loop: user clicks start and we run until they press Stop or interrupt
    if start:
        live_running = True
        st.sidebar.success("Live capture started — close the Streamlit app or press Stop to end.")

    # We'll use a while loop but with Streamlit we must be careful: we'll iterate a few times and
    # provide a Stop button. For an indefinite run you'd normally use session state or threads.
    if live_running:
        try:
            while True:
                screenshot = pyautogui.screenshot()
                img_curr = screenshot.convert("RGB")
                render_predictions(img_curr)
                time.sleep(interval)
                # If user clicked Stop, break. Streamlit can't detect button clicks inside loop easily,
                # so provide a keyboard interrupt (Ctrl+C) or stop the app. To keep this demo simple,
                # break if the Streamlit app is stopped externally.
        except KeyboardInterrupt:
            st.sidebar.warning("Live capture stopped by user.")

# Non-live mode: render once when an image is provided
else:
    if img is not None:
        render_predictions(img)
    else:
        st.info("Provide an image (upload or URL) to see predictions.")

# Footer
st.markdown("---")
st.caption(f"Device used for inference: {device}")