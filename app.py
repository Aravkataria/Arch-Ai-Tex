import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
from PIL import Image
import warnings
import cv2

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

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
        print(f"ERROR: Could not load Random Forest model (File 'random_forest_classifier_model.joblib' missing or corrupt): {e}")

    generator = DCGAN_Generator().to(DEVICE)
    try:
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    except Exception as e:
        print(f"ERROR: Could not load GAN weights (File 'generator_epoch100.pth' missing or corrupt): {e}")
    
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

st.title("Arch-Ai-Tex")
st.markdown("### AI Floor Plan Generator")
st.markdown("Generate floor plans optimized for your required area and number of rooms.")

col_len, col_wid = st.columns(2)
with col_len:
    house_length = st.number_input("Enter House Length (m)", min_value=10.0, max_value=10000.0, value=50.0, step=1.0)
with col_wid:
    house_width = st.number_input("Enter House Width/Depth (m)", min_value=10.0, max_value=10000.0, value=30.0, step=1.0)

area = house_length * house_width
st.markdown(f"**Calculated Total Area:** **{area:.2f} m²**")
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=8, value=3, step=1)
denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False, help="Uses a non-local means filter to smooth noise from the generated image.")

st.markdown("---")

if 'generated' not in st.session_state:
    st.session_state['generated'] = False
    st.session_state['images'] = []
    st.session_state['dwelling_type'] = None
    st.session_state['area'] = area
    st.session_state['bedrooms'] = bedrooms

if st.button("Generate Optimized Floor Plans", type="primary", use_container_width=True):
    if area > 0 and bedrooms >= 0:
        with st.spinner('AI is generating 3 floor plans...'):
            dwelling_type, floor_plan_images = generate_final_plans(
                GAN_MODEL, area, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
            )
        st.session_state['images'] = floor_plan_images
        st.session_state['generated'] = True
        st.session_state['dwelling_type'] = dwelling_type
        st.session_state['area'] = area
        st.session_state['bedrooms'] = bedrooms
        st.toast("Floor Plans Generated!")
    else:
        st.error("Please enter valid length, width, and bedroom values.")

if st.session_state.get('generated'):
    st.divider()
    st.header("Generated Floor Plans")

    if st.session_state['dwelling_type'] and st.session_state['dwelling_type'] != "Prediction Model Missing":
        st.subheader(f"Predicted Dwelling Type: {st.session_state['dwelling_type']}")

    cols = st.columns([1, 0.1, 1, 0.1, 1])
    images = st.session_state['images']

    for i, col_index in enumerate([0, 2, 4]):
        if i < len(images):
            img = images[i]
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")

            with cols[col_index]:
                st.image(img, caption=f"Plan {i+1} (256x256)", use_column_width=True)
                st.download_button(
                    label=f"Download Plan {i+1}",
                    data=img_buffer.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(st.session_state['area'])}sqm_Beds{st.session_state['bedrooms']}.png",
                    mime="image/png",
                    key=f"download_{i}",
                    use_container_width=True
                )
    st.divider()
