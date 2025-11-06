import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.segmentation as models
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2
import math
import warnings
from PIL import Image

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
def load_models():
    rf_model = None
    generator = DCGAN_Generator().to(DEVICE)
    
    try:
        rf_model = joblib.load("room_predictor.joblib")
    except Exception:
        rf_model = None

    # ✅ Fixed GAN loading logic — checks multiple common filenames
    loaded = False
    possible_files = [
        "generator_epoch100.pth",
        "generator_epoch_100.pth",
        "generator.pth"
    ]
    for fname in possible_files:
        try:
            state_dict = torch.load(fname, map_location=DEVICE)
            generator.load_state_dict(state_dict, strict=False)
            loaded = True
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Error loading GAN generator from '{fname}': {e}")
            continue

    if not loaded:
        st.error("GAN generator weights not found in any of the expected files.")

    generator.eval()
    return rf_model, generator, None


RF_MODEL, GAN_MODEL, SEG_MODEL = load_models()

def predict_dwelling_type(area, bedrooms, rf_model):
    if rf_model is None:
        return "Unknown Type (RF model missing)"
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
    seed_base = int(area * 10 + bedrooms * 1234)

    for i in range(count):
        torch.manual_seed(seed_base + i)
        z = torch.randn(1, LATENT_DIM, 1, 1).to(DEVICE)
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
            img = Image.fromarray(img_np, mode)
            images.append(img)

    return dwelling_type, images, pixel_area


def apply_segmentation(image, num_rooms):
    if image.mode != "L":
        img_cv = np.array(image.convert("L"))
    else:
        img_cv = np.array(image)

    _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    seg_rgb = np.zeros((*img_cv.shape, 3), dtype=np.uint8)
    
    room_colors = [
        (255, 199, 107), 
        (130, 202, 157), 
        (174, 199, 232), 
        (255, 152, 150), 
        (197, 176, 213), 
        (255, 237, 111), 
        (188, 189, 34),
        (140, 86, 75),
    ]
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        color_index = (i - 1) % len(room_colors)
        seg_rgb[labels == i] = room_colors[color_index]

    seg_pil = Image.fromarray(seg_rgb).resize(image.size)
    return seg_pil


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

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("AI Floor Plan Generator")
with col2:
    st.image("https://placehold.co/110x110/38761D/ffffff?text=LOGO", width=110)
    st.markdown("<p style='font-size:13px; color:gray; text-align:right;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)

st.markdown("---")

col_len, col_wid = st.columns(2)
with col_len:
    house_length = st.number_input("Enter House Length (m)", min_value=10.0, value=50.0, step=1.0)
with col_wid:
    house_width = st.number_input("Enter House Width (m)", min_value=10.0, value=30.0, step=1.0)

area_m2 = house_length * house_width
if area_m2 < 100:
    area_m2 = 100
area_sqft = area_m2 * 10.7639

st.markdown(f"**Calculated Total Area:** {area_m2:.2f} m² (≈ {area_sqft:.0f} sq ft)**")

bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

if st.button("Generate Floorplans", type="primary", use_container_width=True):
    dwelling_type, floor_plan_images, pixel_area = generate_final_plans(
        GAN_MODEL, area_m2, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
    )
    
    st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
    st.markdown(f"**Area to Pixel Ratio:** 1 pixel ≈ {pixel_area:.4f} m²")
    st.markdown("Generated Floorplans:")

    cols = st.columns(3)
    cols_seg = st.columns(3)
    for i, img in enumerate(floor_plan_images):
        seg_img = apply_segmentation(img, bedrooms)
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        
        cols[i].image(img, caption=f"Plan {i+1}", use_column_width=True)
        cols_seg[i].image(seg_img, caption=f"Segmented Plan {i+1}", use_column_width=True)
        
        seg_buf = io.BytesIO()
        seg_img.save(seg_buf, format="PNG")
        cols_seg[i].download_button(
            label=f"Download Seg. Plan {i+1}",
            data=seg_buf.getvalue(),
            file_name=f"segmented_plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png",
            mime="image/png",
        )
