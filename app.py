import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.utils import make_grid
from PIL import Image
import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
CHANNELS = 3

# ------------------- Generator Definition -------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), LATENT_DIM, 1, 1))

# ------------------- Helper Function -------------------
def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    generator.eval()
    images = []
    seed_base = int(area * 10 + bedrooms * 1234)

    for i in range(count):
        torch.manual_seed(seed_base + i)
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = generator(z)
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2.0
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if denoise:
            img_np = cv2.fastNlMeansDenoisingColored((img_np * 255).astype(np.uint8), None, 10, 10, 7, 21)
            img_np = img_np.astype(np.float32) / 255.0
        images.append(img_np)

    imgs_tensor = torch.stack([torch.tensor(img.transpose(2, 0, 1)) for img in images])
    grid = make_grid(imgs_tensor, nrow=min(count, 3))
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = np.clip(grid_np, 0, 1)
    grid_np = (grid_np * 255).astype(np.uint8)
    return grid_np, images

# ------------------- Streamlit UI -------------------
st.title("Arch-Ai-Tex: GAN Floor Plan Generator")

uploaded_model = st.file_uploader("Upload your trained generator (.pth)", type=["pth"])
area = st.number_input("Enter total area (sqft)", min_value=100, max_value=10000, value=1200)
bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=3)
count = st.slider("Number of plans to generate", 1, 5, 3)
denoise = st.checkbox("Apply denoising", value=False)

if uploaded_model is not None:
    try:
        generator = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
        torch.save(uploaded_model.read(), "temp_gen.pth")
        state_dict = torch.load("temp_gen.pth", map_location=DEVICE)
        generator.load_state_dict(state_dict)
        st.success("Generator model loaded successfully.")

        if st.button("Generate Floor Plans"):
            grid_img, imgs = generate_final_plans(generator, area, bedrooms, count, denoise)
            st.image(grid_img, caption="Generated Plans", use_container_width=True)

    except Exception as e:
        st.error(f"Error loading or generating: {e}")
else:
    st.info("Upload a trained generator .pth file to start.")
