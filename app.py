import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import joblib
import warnings

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
IMG_SIZE = 256
PIXEL_TO_AREA = 0.25  # 1 pixel = 0.25 sq units (you can adjust this)

@st.cache_resource
def load_all_models():
    rf_model_loaded = None
    generator = DCGAN_Generator().to(DEVICE)

    # Load the Random Forest model first
    try:
        rf_model_loaded = joblib.load("room_predictor.joblib")
        print("✅ Room predictor loaded successfully.")
    except Exception as e:
        print(f"⚠️ Room predictor not found or failed to load: {e}")

    # Load the GAN Generator weights
    try:
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
        print("✅ Generator weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Generator weights not found or failed to load: {e}")

    generator.eval()
    return rf_model_loaded, generator

# ------------------------------------------------------------
# Generator architecture (example DCGAN-style)
# ------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# ------------------------------------------------------------
# Helper: Convert tensor to PIL
# ------------------------------------------------------------
def tensor_to_pil(tensor):
    tensor = (tensor + 1) / 2
    tensor = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


# ------------------------------------------------------------
# Generate final plans
# ------------------------------------------------------------
def generate_final_plans(num_rooms, generator, classifier=None):
    if classifier:
        dwelling_type = classifier.predict([[num_rooms]])[0]
    else:
        dwelling_type = np.random.randint(1, 5)

    generated_images = []
    for i in range(3):
        z = torch.randn(1, LATENT_DIM, 1, 1).to(DEVICE)
        with torch.no_grad():
            fake_img = generator(z).detach().cpu()

        print(f"DEBUG Plan {i+1}:", torch.min(fake_img), torch.max(fake_img), fake_img.shape)

        # Convert to image
        img = tensor_to_pil(fake_img)

        # Calculate approximate area
        area = (IMG_SIZE ** 2) * PIXEL_TO_AREA

        # If too small, increase scale
        if area < 5000:
            scale = int(np.sqrt(5000 / area))
            new_size = (IMG_SIZE * scale, IMG_SIZE * scale)
            img = img.resize(new_size, Image.NEAREST)
            area = new_size[0] * new_size[1] * PIXEL_TO_AREA

        generated_images.append((img, area))

    return dwelling_type, generated_images


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("Arch-Ai-Tex")
st.caption("AI-powered Architectural Floorplan Generator")

generator, classifier = load_models()

num_rooms = st.number_input("Enter number of rooms (including kitchen & bathroom):", min_value=1, max_value=10, value=3, step=1)

if st.button("Generate Floorplans", use_container_width=True):
    with st.spinner("Generating..."):
        dwelling_type, generated_images = generate_final_plans(num_rooms, generator, classifier)

        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        cols = st.columns(3)

        for i, (img, area) in enumerate(generated_images):
            with cols[i]:
                st.image(img, caption=f"Plan {i+1} | Approx. Area: {int(area)} sq units", use_container_width=True)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                byte_img = buf.getvalue()
                st.download_button(
                    label=f"Download Plan {i+1}",
                    data=byte_img,
                    file_name=f"floorplan_{i+1}.png",
                    mime="image/png"
                )
