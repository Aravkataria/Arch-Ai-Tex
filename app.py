import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
from PIL import Image
import warnings
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random, json, re, math, copy, time
from transformers import pipeline

warnings.filterwarnings("ignore", message="missing ScriptRunContext")
st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

# ---------------- GAN MODEL ----------------
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
    except:
        pass
    generator = DCGAN_Generator().to(DEVICE)
    try:
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    except:
        pass
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
            if CHANNELS > 1 and img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            if denoise:
                if CHANNELS == 1:
                    img_np = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
                else:
                    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
            img_pil = Image.fromarray(img_np, 'L' if CHANNELS == 1 else 'RGB')
            images.append(img_pil)
    return dwelling_type, images

# ---------------- FLAN-T5 MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base", do_sample=False, temperature=0.0, max_new_tokens=256)

generator_t5 = load_model()

def extract_json_from_model_output(output_text):
    try:
        return json.loads(output_text)
    except:
        m = re.search(r"\{[\s\S]*\}", output_text)
        if not m: raise ValueError("No JSON block")
        js = m.group(0).replace("“", "\"").replace("”", "\"")
        return json.loads(js)

def fallback_layout(total_area, num_rooms):
    rooms = [{"name": "living+dining", "area": round(total_area * 0.3, 1)}]
    for i in range(num_rooms):
        rooms.append({"name": f"bedroom_{i+1}", "area": round(total_area * 0.15, 1)})
    rooms += [{"name": "kitchen", "area": round(total_area * 0.1, 1)},
              {"name": "bathroom", "area": round(total_area * 0.07, 1)}]
    return {"rooms": rooms}

def generate_semantic_layout(total_area, num_rooms, property_type, plot_shape, plot_w, plot_h):
    prompt = f"""Design a {property_type} layout for a {plot_shape} plot ({plot_w}x{plot_h}m), total area {total_area} sqm and {num_rooms} bedrooms. 
    Return JSON: {{ "rooms": [{{"name":"...", "area":...}}, ...] }}"""
    try:
        out = generator_t5(prompt)[0]["generated_text"]
        parsed = extract_json_from_model_output(out)
        return parsed, out
    except:
        return fallback_layout(total_area, num_rooms), ""

def plot_layout(rooms, w, h, title="Optimized Layout"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor='black', facecolor='none', linestyle='--'))
    colors = ["#c2a2d3", "#a8d8e0", "#f6c28b", "#d6eadf", "#f7c6c7", "#b5ead7"]
    x, y = 0.1, 0.1
    for r in rooms["rooms"]:
        size = math.sqrt(r["area"]) / 2
        rect = patches.Rectangle((x, y), size, size, facecolor=random.choice(colors), edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + size/2, y + size/2, f"{r['name']}\n{r['area']} sqm", ha='center', va='center')
        x += size + 0.2
        if x + size > w: x, y = 0.1, y + size + 0.3
    ax.set_title(title)
    return fig

# ---------------- UI ----------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("### AI Floor Plan Generator")
with col2:
    st.image("QR.png", width=110)
    st.markdown("<p style='font-size:14px; color:gray;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)

st.markdown("---")
mode = st.radio("Select Model:", ["GAN Generator", "Optimized Layout"], horizontal=True)

if mode == "GAN Generator":
    col_len, col_wid = st.columns(2)
    with col_len:
        house_length = st.number_input("Enter House Length (m)", 10.0, 10000.0, 50.0)
    with col_wid:
        house_width = st.number_input("Enter House Width (m)", 10.0, 10000.0, 30.0)
    area = house_length * house_width
    st.markdown(f"**Calculated Total Area:** **{area:.2f} m²**")
    bedrooms = st.number_input("Enter Number of Bedrooms", 1, 8, 3)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

    if st.button("Generate Floor Plans"):
        with st.spinner('Generating...'):
            dwelling_type, imgs = generate_final_plans(GAN_MODEL, area, bedrooms, 3, denoise_option, RF_MODEL)
            st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
            for i, img in enumerate(imgs):
                buf = io.BytesIO(); img.save(buf, format="PNG")
                st.image(img, caption=f"Plan {i+1}", use_column_width=True)
                st.download_button(f"Download Plan {i+1}", buf.getvalue(), f"plan_{i+1}.png", "image/png")

else:
    total_area = st.slider("Total area (sqm)", 30, 500, 120, 10)
    num_rooms = st.slider("Number of bedrooms", 0, 6, 2)
    property_type = st.selectbox("Property type", ["Apartment", "Villa", "Bungalow"])
    plot_shape = st.selectbox("Plot shape", ["Square", "Rectangular"])
    plot_w = st.number_input("Plot width (m)", 5.0, 50.0, 10.0)
    plot_h = st.number_input("Plot height (m)", 5.0, 50.0, 10.0)

    if st.button("Generate Optimized Layout"):
        with st.spinner("Generating layout..."):
            layout, _ = generate_semantic_layout(total_area, num_rooms, property_type, plot_shape, plot_w, plot_h)
            st.json(layout)
            fig = plot_layout(layout, plot_w, plot_h, "Optimized Layout")
            st.pyplot(fig)
