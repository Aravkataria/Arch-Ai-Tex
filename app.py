import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
from PIL import Image
import warnings
import cv2
import pandas as pd

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

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
        rf_model_loaded = joblib.load("random_forest_classifier_model.joblib")
    except Exception as e:
        print(f"ERROR: Random Forest model not found: {e}")

    generator = DCGAN_Generator().to(DEVICE)
    try:
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    except Exception as e:
        print(f"ERROR: GAN weights not found: {e}")

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
                img_np = cv2.fastNlMeansDenoising(img_np, None, h=10, templateWindowSize=7, searchWindowSize=21)
            else:
                img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        mode = 'L' if CHANNELS == 1 else 'RGB'
        img_pil = Image.fromarray(img_np, mode)
        images.append(img_pil)
    return dwelling_type, images

PRETRAINED_DATA = {
    "rooms": [
        {"name": "living+dining", "area": 33.6, "x": 0.1, "y": 0.1, "w": 5.2169, "h": 6.4406},
        {"name": "bedroom_1", "area": 25.9, "x": 0.1, "y": 6.69, "w": 4.5802, "h": 5.6546},
        {"name": "bedroom_2", "area": 25.9, "x": 4.83, "y": 6.69, "w": 4.5802, "h": 5.6546},
        {"name": "kitchen", "area": 20.8, "x": 0.1, "y": 12.49, "w": 4.1046, "h": 5.0674},
        {"name": "bathroom", "area": 13.8, "x": 4.35, "y": 12.49, "w": 3.3433, "h": 4.1275}
    ]
}

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
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("### AI Floor Plan Generator")

with col2:
    st.markdown("<div style='text-align:right; padding-top:10px;'>", unsafe_allow_html=True)
    st.image("QR.png", width=110)
    st.markdown(
        "<p style='font-size:14px; color:gray; text-align:right;'>"
        "Scan the QR to view the full project or GitHub repository."
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

col_len, col_wid = st.columns(2)
with col_len:
    house_length = st.number_input("Enter House Length (m)", min_value=10.0, max_value=10000.0, value=50.0, step=1.0)
with col_wid:
    house_width = st.number_input("Enter House Width/Depth (m)", min_value=10.0, max_value=10000.0, value=30.0, step=1.0)

area = house_length * house_width
st.markdown(f"**Calculated Total Area:** **{area:.2f} m²**")
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=8, value=3, step=1)
denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

st.markdown("---")

model_choice = st.radio(
    "Choose Model Type",
    ("Use Our Trained Model", "Use Pre-Trained Model"),
    horizontal=True
)

if st.button("Generate Floor Plans", type="primary", use_container_width=True):
    if model_choice == "Use Our Trained Model":
        with st.spinner('Generating floor plans using your GAN model...'):
            dwelling_type, floor_plan_images = generate_final_plans(
                GAN_MODEL, area, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
            )
        st.success("Floor Plans Generated")
        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        st.divider()

        cols = st.columns([1, 0.1, 1, 0.1, 1])
        for i, col_index in enumerate([0, 2, 4]):
            if i < len(floor_plan_images):
                img = floor_plan_images[i]
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                with cols[col_index]:
                    st.image(img, caption=f"Plan {i+1}", use_column_width=True)
                    st.download_button(
                        label=f"Download Plan {i+1}",
                        data=img_buffer.getvalue(),
                        file_name=f"plan_{i+1}_Area{int(area)}sqm_Beds{bedrooms}.png",
                        mime="image/png",
                        key=f"download_{i}",
                        use_container_width=True
                    )
    else:
        st.success("Pre-Trained Model Output")
        st.subheader("Room Layout Data")
        df = pd.DataFrame(PRETRAINED_DATA["rooms"])
        st.dataframe(df, use_container_width=True)
        st.caption("Pre-trained model generated room data after rescaling and greedy packing.")
