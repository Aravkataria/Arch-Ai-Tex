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
import math

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256


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
        rf_model_loaded = joblib.load("room_predictor.joblib")
    except Exception:
        pass

    generator = DCGAN_Generator().to(DEVICE)
    try:
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    except Exception:
        pass

    generator.eval()
    return rf_model_loaded, generator


RF_MODEL, GAN_MODEL = load_all_models()


def predict_dwelling_type(area, bedrooms, rf_model):
    if rf_model is None:
        return "Prediction Model Missing"
    try:
        features = np.array([[float(area), int(bedrooms)]])
        return rf_model.predict(features)[0]
    except Exception:
        return "Prediction Failed"


def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    dwelling_type = predict_dwelling_type(area, bedrooms, rf_model)
    images = []

    for _ in range(count):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = generator(z)
            img_np = img_tensor.squeeze().cpu().numpy()
            print("DEBUG:", torch.min(generated), torch.max(generated), generated.shape)
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


def generate_semantic_layout(total_area, num_bedrooms, property_type, plot_shape, plot_w, plot_h):
    total_area = float(total_area)
    num_bedrooms = max(0, int(num_bedrooms))
    fixed_ratios = {"living+dining": 0.28, "kitchen": 0.08, "bathroom": 0.06}
    fixed_total = sum(fixed_ratios.values())
    remaining_ratio = max(0.0, 1.0 - fixed_total)

    rooms = []
    for name, ratio in fixed_ratios.items():
        rooms.append({"name": name, "area": round(total_area * ratio, 2)})

    if num_bedrooms > 0:
        per_bed_ratio = remaining_ratio / num_bedrooms
        for i in range(num_bedrooms):
            rooms.append({"name": f"bedroom_{i+1}", "area": round(total_area * per_bed_ratio, 2)})
    else:
        rooms.append({"name": "utility/other", "area": round(total_area * remaining_ratio, 2)})

    current_sum = round(sum(r["area"] for r in rooms), 2)
    diff = round(total_area - current_sum, 2)
    if abs(diff) >= 0.01:
        rooms[0]["area"] = round(rooms[0]["area"] + diff, 2)

    return {"rooms": rooms}, ""


def plot_layout(layout, plot_w, plot_h, title="Layout"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, plot_h)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), plot_w, plot_h, fill=False, edgecolor='black', linewidth=1.2))

    rooms = layout.get("rooms", [])
    if not rooms:
        ax.set_title(title)
        return fig

    total_area = sum(r["area"] for r in rooms)
    scale = (plot_w * plot_h) / max(total_area, 1.0)
    pad = min(plot_w, plot_h) * 0.02

    x, y = pad, pad
    row_h = 0
    colors = ["#f4cccc", "#d9ead3", "#cfe2f3", "#fff2cc", "#d9d2e9", "#c2f0c2"]

    for i, r in enumerate(rooms):
        desired_area = max(0.1, r["area"])
        rect_area = desired_area * scale
        w = math.sqrt(rect_area) * 1.3
        h = rect_area / w

        if x + w + pad > plot_w:
            x = pad
            y += row_h + pad
            row_h = 0

        if y + h + pad > plot_h:
            break

        rect = plt.Rectangle((x, y), w, h,
                             facecolor=colors[i % len(colors)],
                             edgecolor='black', linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{r['name']}\n{r['area']} m²",
                ha='center', va='center', fontsize=8)

        x += w + pad
        row_h = max(row_h, h)

    ax.set_title(title)
    return fig


st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 1.05em;
    border: none;
    transition: all 0.15s;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
}
.stImage > img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("AI Floor Plan Generator")
with col2:
    st.image("QR.png", width=110)
    st.markdown("<p style='font-size:13px; color:gray; text-align:right;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)

st.markdown("---")

mode = st.radio("Select Model:", ["GAN Generator", "Optimized Layout"], horizontal=True)

if mode == "GAN Generator":
    col_len, col_wid = st.columns(2)
    with col_len:
        house_length = st.number_input("Enter House Length (m)", min_value=5.0, value=50.0, step=1.0)
    with col_wid:
        house_width = st.number_input("Enter House Width (m)", min_value=5.0, value=30.0, step=1.0)

    area = house_length * house_width
    st.markdown(f"**Calculated Total Area:** {area:.2f} m²")

    if area < 400:
        scale_factor = 400 / area
        area = area * scale_factor
        st.warning(f"Area too small, scaled up by ×{scale_factor:.2f} for better visibility (≈ {area:.2f} m²).")

    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

    pixel_scale = area / (IMG_SIZE * IMG_SIZE)
    st.markdown(f"1 pixel ≈ {pixel_scale:.4f} m² (auto-adjusted to total area {area:.2f} m²)")

    if st.button("Generate Floorplans", type="primary", use_container_width=True):
        dwelling_type, floor_plan_images = generate_final_plans(
            GAN_MODEL, area, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
        )

        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(floor_plan_images):
                img = floor_plan_images[i]
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                col.image(img, caption=f"Plan {i+1}", use_column_width=True)
                col.download_button(
                    label=f"Download Plan {i+1}",
                    data=buf.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(area)}sqm_Beds{bedrooms}.png",
                    mime="image/png",
                )

else:
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        num_rooms = st.number_input("Enter Total Number of Rooms", min_value=1, value=3,
                                    help="This count includes kitchen and bathroom.")

    st.markdown("<p style='font-size:13px; color:gray;'>Note: The total number of rooms includes the kitchen and bathroom.</p>", unsafe_allow_html=True)

    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Bungalow"])
    plot_shape = st.selectbox("Plot Shape", ["Square", "Rectangular"])

    colW, colH = st.columns(2)
    with colW:
        plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=10.0)
    with colH:
        plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=10.0)

    if st.button("Generate Optimized Layout"):
        with st.spinner("Generating layout..."):
            layout, _ = generate_semantic_layout(total_area, num_rooms, property_type, plot_shape, plot_w, plot_h)
            dwelling_type = predict_dwelling_type(total_area, num_rooms, RF_MODEL)
            st.success(f"Predicted Dwelling Type: {dwelling_type}")
            fig = plot_layout(layout, plot_w, plot_h, f"{property_type} Layout")
            st.pyplot(fig)
