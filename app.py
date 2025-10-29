import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import warnings

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

# ---------------------- GAN MODEL ----------------------
class DCGAN_Generator(nn.Module):
    @staticmethod
    def block(in_f, out_f):
        return nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.gen = nn.Sequential(
            DCGAN_Generator.block(512, 256),
            DCGAN_Generator.block(256, 128),
            DCGAN_Generator.block(128, 64),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.gen(out)

@st.cache_resource
def load_all_models():
    rf_model_loaded = None
    try:
        rf_model_loaded = joblib.load("random_forest_classifier_model.joblib")
    except Exception as e:
        print(f"RF model missing: {e}")

    generator = DCGAN_Generator().to(DEVICE)
    try:
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    except Exception as e:
        print(f"GAN weights missing: {e}")

    generator.eval()
    return rf_model_loaded, generator

RF_MODEL, GAN_MODEL = load_all_models()

def predict_dwelling_type(area, bedrooms, rf_model):
    if rf_model is None:
        return "Prediction Model Missing"
    features = np.array([[area, bedrooms]])
    return rf_model.predict(features)[0]

def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    dwelling_type = predict_dwelling_type(area, bedrooms, rf_model)
    images = []
    for _ in range(count):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = generator(z)
        img_np = img_tensor.squeeze().cpu().numpy()
        img_np = np.clip(((img_np + 1) * 127.5), 0, 255).astype(np.uint8)

        if CHANNELS > 1 and img_np.ndim == 3 and img_np.shape[0] == CHANNELS:
            img_np = np.transpose(img_np, (1, 2, 0))

        if denoise:
            if CHANNELS == 1:
                img_np = cv2.fastNlMeansDenoising(img_np, None, h=10)
            else:
                img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10)

        mode = 'L' if CHANNELS == 1 else 'RGB'
        images.append(Image.fromarray(img_np, mode))
    return dwelling_type, images


# ---------------------- OPTIMIZED LAYOUT GENERATOR ----------------------
def generate_semantic_layout(area, rooms, property_type, shape, width, height):
    layout = {
        "Area (sqm)": area,
        "Bedrooms": rooms,
        "Property Type": property_type,
        "Plot Shape": shape,
        "Dimensions": {"Width": width, "Height": height}
    }

    # Randomized room rectangles for visualization
    np.random.seed(42)
    room_positions = []
    for i in range(rooms):
        rw, rh = np.random.uniform(0.2, 0.4) * width, np.random.uniform(0.2, 0.4) * height
        rx, ry = np.random.uniform(0, width - rw), np.random.uniform(0, height - rh)
        room_positions.append((rx, ry, rw, rh))
    return layout, room_positions


def plot_layout(layout, width, height, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title(title)
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.3)

    # Draw plot boundary
    rect = plt.Rectangle((0, 0), width, height, fill=None, edgecolor='black', lw=2)
    ax.add_patch(rect)

    # Draw rooms
    layout_data, rooms = generate_semantic_layout(
        layout["Area (sqm)"], layout["Bedrooms"],
        layout["Property Type"], layout["Plot Shape"],
        width, height
    )
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(rooms)))
    for i, (rx, ry, rw, rh) in enumerate(rooms):
        ax.add_patch(plt.Rectangle((rx, ry), rw, rh, color=colors[i], alpha=0.7))
        ax.text(rx + rw / 2, ry + rh / 2, f"Room {i+1}",
                ha='center', va='center', fontsize=10, weight='bold')

    return fig


# ---------------------- STYLING ----------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 1.1em;
    transition: all 0.2s;
    border: none;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stImage > img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("### AI Floor Plan Generator")
with col2:
    st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
    st.image("QR.png", width=110)
    st.markdown(
        "<p style='font-size:14px; color:gray; text-align:right;'>Scan the QR to view the full project or GitHub repository.</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- MODEL SELECTION ----------------------
mode = st.radio("Select Model:", ["GAN Generator", "Optimized Layout"], horizontal=True)
st.markdown("---")

# ---------------------- GAN MODEL MODE ----------------------
if mode == "GAN Generator":
    col_len, col_wid = st.columns(2)
    with col_len:
        house_length = st.number_input("Enter House Length (m)", min_value=10.0, max_value=10000.0, value=50.0)
    with col_wid:
        house_width = st.number_input("Enter House Width (m)", min_value=10.0, max_value=10000.0, value=30.0)
    area = house_length * house_width
    st.markdown(f"**Calculated Total Area:** {area:.2f} m²")
    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=8, value=3)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

    if st.button("Generate Floorplans", type="primary", use_container_width=True):
        dwelling_type, floor_plan_images = generate_final_plans(
            GAN_MODEL, area, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
        )
        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        st.markdown("### Generated Floorplans:")
        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(floor_plan_images):
                img = floor_plan_images[i]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                col.image(img, caption=f"Floorplan {i+1}")
                col.download_button(
                    label=f"Download Plan {i+1}",
                    data=buf.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(area)}sqm_Beds{bedrooms}.png",
                    mime="image/png"
                )

# ---------------------- OPTIMIZED LAYOUT MODE ----------------------
else:
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", 30.0, 10000.0, 120.0)
    with colB:
        num_rooms = st.number_input("Enter Number of Bedrooms", 1, 10, 2)

    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Bungalow"])
    plot_shape = st.selectbox("Plot Shape", ["Square", "Rectangular"])
    colW, colH = st.columns(2)
    with colW:
        plot_w = st.number_input("Plot Width (m)", 5.0, 100.0, 10.0)
    with colH:
        plot_h = st.number_input("Plot Height (m)", 5.0, 100.0, 10.0)

    if st.button("Generate Optimized Layout", type="primary", use_container_width=True):
        with st.spinner("Generating layout..."):
            layout, _ = generate_semantic_layout(total_area, num_rooms, property_type, plot_shape, plot_w, plot_h)
            st.json(layout)
            fig = plot_layout(layout, plot_w, plot_h, "Optimized Layout")
            st.pyplot(fig)
