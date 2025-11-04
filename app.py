import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
import warnings
import random

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

# ------------------------------
# DCGAN Generator
# ------------------------------
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

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_models():
    rf_model = None
    generator = DCGAN_Generator().to(DEVICE)
    try:
        rf_model = joblib.load("room_predictor.joblib")
    except Exception:
        pass
    try:
        state_dict = torch.load("generator_epoch100.pth", map_location=DEVICE)
        generator.load_state_dict(state_dict)
    except Exception:
        pass
    generator.eval()
    return rf_model, generator

RF_MODEL, GAN_MODEL = load_models()

# ------------------------------
# Prediction + Generation
# ------------------------------
def predict_dwelling_type(area, bedrooms, rf_model):
    if rf_model is None:
        return "Unknown Type"
    try:
        features = np.array([[float(area), int(bedrooms)]])
        return rf_model.predict(features)[0]
    except Exception:
        return "Prediction Failed"

def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    dwelling_type = predict_dwelling_type(area, bedrooms, rf_model)
    images = []
    if area < 100:
        area = 100
    pixel_area = area / (IMG_SIZE * IMG_SIZE)

    for _ in range(count):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = generator(z)
            img_np = img_tensor.squeeze().cpu().numpy()
            img_np = np.clip(((img_np + 1) * 127.5), 0, 255).astype(np.uint8)
            if CHANNELS > 1 and img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            if denoise:
                img_np = cv2.fastNlMeansDenoising(img_np, None, h=10)
            img = Image.fromarray(img_np, mode='L' if CHANNELS == 1 else 'RGB')
            images.append(img)
    return dwelling_type, images, pixel_area

# ------------------------------
# Mock segmentation model (replaceable later)
# ------------------------------
def segment_rooms(image, num_rooms):
    img = np.array(image)
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in range(num_rooms)]
    for i in range(num_rooms):
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(x1+20, w), random.randint(y1+20, h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), colors[i], -1)
    blended = cv2.addWeighted(np.array(image.convert("RGB")), 0.5, mask, 0.5, 0)
    return Image.fromarray(blended)

# ------------------------------
# Optimized Layout Function
# ------------------------------
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
    return {"rooms": rooms}, ""

def plot_layout(layout, plot_w, plot_h, title="Layout"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, plot_h)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), plot_w, plot_h, fill=False, edgecolor='black', linewidth=1.2))
    rooms = layout.get("rooms", [])
    total_area = sum(r["area"] for r in rooms)
    scale = (plot_w * plot_h) / max(total_area, 1.0)
    pad = min(plot_w, plot_h) * 0.02
    x, y, row_h = pad, pad, 0
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
        rect = plt.Rectangle((x, y), w, h, facecolor=colors[i % len(colors)], edgecolor='black', linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{r['name']}\n{r['area']} m²", ha='center', va='center', fontsize=8)
        x += w + pad
        row_h = max(row_h, h)
    ax.set_title(title)
    return fig

# ------------------------------
# Styling
# ------------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 1.05em;
    transition: all 0.15s;
    border: none;
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

# ------------------------------
# Main UI
# ------------------------------
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
        house_length = st.number_input("Enter House Length (m)", min_value=10.0, value=50.0, step=1.0)
    with col_wid:
        house_width = st.number_input("Enter House Width (m)", min_value=10.0, value=30.0, step=1.0)

    area_m2 = house_length * house_width
    area_sqft = area_m2 * 10.7639
    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

    if st.button("Generate Floorplans", type="primary", use_container_width=True):
        dwelling_type, floor_plan_images, pixel_area = generate_final_plans(
            GAN_MODEL, area_sqft, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
        )

        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        st.markdown(f"**Area to Pixel Ratio:** 1 pixel ≈ {pixel_area:.4f} m²")

        st.markdown("Generated Floorplans with Room Segmentation:")
        for i, img in enumerate(floor_plan_images):
            seg_img = segment_rooms(img, bedrooms)
            buf1, buf2 = io.BytesIO(), io.BytesIO()
            img.save(buf1, format="PNG")
            seg_img.save(buf2, format="PNG")

            colA, colB = st.columns(2)
            with colA:
                st.image(img, caption=f"Generated Plan {i+1}", use_column_width=True)
                st.download_button(f"Download Original {i+1}", data=buf1.getvalue(),
                                   file_name=f"plan_{i+1}.png", mime="image/png")
            with colB:
                st.image(seg_img, caption=f"Segmented (Rooms Highlighted)", use_column_width=True)
                st.download_button(f"Download Segmented {i+1}", data=buf2.getvalue(),
                                   file_name=f"segmented_{i+1}.png", mime="image/png")

else:
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        num_rooms = st.number_input("Enter Total Number of Rooms", min_value=1, value=3)
    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Bungalow"])
    plot_shape = st.selectbox("Plot Shape", ["Square", "Rectangular"])
    colW, colH = st.columns(2)
    with colW:
        plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=10.0)
    with colH:
        plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=10.0)

    if st.button("Generate Optimized Layout"):
        layout, _ = generate_semantic_layout(total_area, num_rooms, property_type, plot_shape, plot_w, plot_h)
        dwelling_type = predict_dwelling_type(total_area, num_rooms, RF_MODEL)
        st.success(f"Predicted Dwelling Type: **{dwelling_type}**")
        fig = plot_layout(layout, plot_w, plot_h, f"{property_type} Layout")
        st.pyplot(fig)
