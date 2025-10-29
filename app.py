import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
from PIL import Image
import warnings
import cv2
import os

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("Arch-Ai-Tex")
with col2:
    st.image("QR.png", width=110)
    st.caption("Scan the QR to view the full project or GitHub repository")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_layout_pipeline():
    try:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe
    except Exception as e:
        return None

LAYOUT_PIPE = load_layout_pipeline()

def generate_layout_image(area, bedrooms):
    area_sq_feet = area * 10.764
    if LAYOUT_PIPE:
        prompt = f"architectural layout of a {bedrooms}-bedroom house in {area_sq_feet:.0f} square feet area, clear lines, black and white floor plan"
        image = LAYOUT_PIPE(prompt=prompt, guidance_scale=7.5).images[0]
        return image
    else:
        img = np.ones((256, 256, 3), np.uint8) * 255
        text = f"{bedrooms} BR | {area:.0f} sq. m"
        cv2.putText(img, text, (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return Image.fromarray(img)

st.subheader("Generative Layout Design")

area = st.number_input("Enter available area (in sq. meters):", min_value=10.0, max_value=1000.0, value=150.0)
bedrooms = st.slider("Select number of bedrooms:", 1, 5, 3)

if st.button("Generate Layout"):
    with st.spinner("Generating layout..."):
        layout_img = generate_layout_image(area, bedrooms)
        st.image(layout_img, caption="Generated Layout", use_container_width=True)

st.subheader("Room Predictor (Trained Model)")

@st.cache_resource
def load_room_predictor():
    model_path = "random_forest_classifier_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    return None

ROOM_MODEL = load_room_predictor()

if ROOM_MODEL:
    uploaded_file = st.file_uploader("Upload housing dataset (.csv)", type=["csv"])
    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        X = df.drop("RoomType", axis=1, errors="ignore")
        preds = ROOM_MODEL.predict(X)
        st.write("Predictions:", preds)
else:
    st.warning("Room predictor model not found. Please upload it to the repo.")

st.markdown("---")
st.markdown("© 2025 Arch-Ai-Tex | AI-powered Architectural Design Tool")
