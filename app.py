import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
from PIL import Image
import warnings
import cv2
from transformers import T5Tokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline  # layout generation

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
    rf_model = None
    try:
        rf_model = joblib.load("random_forest_classifier_model.joblib")
    except:
        pass

    generator = DCGAN_Generator().to(DEVICE)
    try:
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    except:
        pass

    generator.eval()
    return rf_model, generator

@st.cache_resource
def load_text_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return tokenizer, model

@st.cache_resource
def load_layout_pipeline():
    # this uses a text-to-image diffusion model to generate layout-style floor plans
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

RF_MODEL, GAN_MODEL = load_all_models()
TEXT_TOKENIZER, TEXT_MODEL = load_text_model()
LAYOUT_PIPE = load_layout_pipeline()

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
            img_np = np.transpose(img_np, (1,2,0))
        if denoise:
            if CHANNELS == 1:
                img_np = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
            else:
                img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10,10,7,21)
        mode = 'L' if CHANNELS==1 else 'RGB'
        img_pil = Image.fromarray(img_np, mode)
        images.append(img_pil)
    return dwelling_type, images

def generate_text_description(area, bedrooms):
    prompt = f"Generate a detailed architectural description for a house with area {area:.2f} square meters and {bedrooms} bedrooms."
    inputs = TEXT_TOKENIZER(prompt, return_tensors="pt")
    outputs = TEXT_MODEL.generate(**inputs, max_length=150)
    return TEXT_TOKENIZER.decode(outputs[0], skip_special_tokens=True)

def generate_layout_image(area, bedrooms):
    prompt = f"floor plan with {bedrooms} bedrooms and total area {area:.2f} square meters"
    image = LAYOUT_PIPE(prompt).images[0]
    return image

# UI
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
    house_width = st.number_input("Enter House Width (m)", min_value=10.0, max_value=10000.0, value=30.0, step=1.0)

area = house_length * house_width
st.markdown(f"**Calculated Total Area:** **{area:.2f} m²**")

bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=8, value=3, step=1)
denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

model_choice = st.radio(
    "Choose AI Mode:",
    ["Custom GAN Model", "Pre-Trained Layout Generator"]
)

st.markdown("---")

if 'generated' not in st.session_state:
    st.session_state['generated'] = False
    st.session_state['images'] = []
    st.session_state['dwelling_type'] = None
    st.session_state['description'] = None
    st.session_state['layout_img'] = None

if st.button("Generate AI Output", use_container_width=True):
    if model_choice == "Custom GAN Model":
        with st.spinner("Generating floor plans..."):
            dwelling_type, floor_plan_images = generate_final_plans(
                GAN_MODEL, area, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
            )
            st.session_state['images'] = floor_plan_images
            st.session_state['dwelling_type'] = dwelling_type
            st.session_state['layout_img'] = None
    else:
        with st.spinner("Generating layout image..."):
            layout_image = generate_layout_image(area, bedrooms)
            st.session_state['layout_img'] = layout_image
            st.session_state['images'] = []
    st.session_state['generated'] = True

st.markdown("---")

if st.session_state.get('generated'):
    if model_choice == "Custom GAN Model" and st.session_state['images']:
        st.header("Generated Floor Plans")
        if st.session_state['dwelling_type']:
            st.subheader(f"Predicted Dwelling Type: {st.session_state['dwelling_type']}")
        cols = st.columns([1,0.1,1,0.1,1])
        images = st.session_state['images']
        for i, col_index in enumerate([0,2,4]):
            if i < len(images):
                img = images[i]
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                with cols[col_index]:
                    st.image(img, caption=f"Plan {i+1} (256×256)", use_column_width=True)
                    st.download_button(
                        label=f"Download Plan {i+1}",
                        data=img_buffer.getvalue(),
                        file_name=f"plan_{i+1}_Area{int(area)}_Beds{bedrooms}.png",
                        mime="image/png",
                        key=f"download_{i}"
                    )
    elif model_choice == "Pre-Trained Layout Generator" and st.session_state.get('layout_img'):
        st.header("Generated Layout")
        st.image(st.session_state['layout_img'], use_column_width=True)
