import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.segmentation as models
import numpy as np
import io
from PIL import Image
import cv2
import warnings

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
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
        return self.model(z)

@st.cache_resource
def load_generator():
    model = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
    try:
        state_dict = torch.load("generator_epoch_100.pth", map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        st.error(f"Error loading GAN generator: {e}")
    model.eval()
    return model

@st.cache_resource
def load_segmentation_model():
    model = models.deeplabv3_resnet101(pretrained=True).to(DEVICE)
    model.eval()
    return model

def generate_floorplan(model, latent_dim, num_images=3):
    noise = torch.randn(num_images, latent_dim, 1, 1).to(DEVICE)
    with torch.no_grad():
        fake_images = model(noise).cpu()
    # Convert from [-1, 1] → [0, 255]
    fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1)
    images = []
    for img_tensor in fake_images:
        img_np = img_tensor.squeeze().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode="L")
        images.append(img)
    return images

def apply_segmentation(model, image, num_rooms):
    gray = np.array(image.convert("L"))
    edges = cv2.Canny(gray, 60, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    seg_rgb = np.full((gray.shape[0], gray.shape[1], 3), (230, 230, 230), dtype=np.uint8)
    seg_rgb[edges > 0] = (50, 100, 220)
    seg_pil = Image.fromarray(seg_rgb)
    return seg_pil

st.title("Arch-Ai-Tex")

num_rooms = st.slider("Number of rooms", 1, 10, 3)
generator_model = load_generator()
segmentation_model = load_segmentation_model()

if st.button("Generate Floorplans"):
    generated_plans = generate_floorplan(generator_model, LATENT_DIM, num_images=3)
    st.subheader("Generated Floorplans:")
    cols = st.columns(3)
    for i, img in enumerate(generated_plans):
        with cols[i]:
            st.image(img, caption=f"Plan {i+1}", use_container_width=True)

    segmented_images = []
    st.subheader("Segmented Floorplans:")
    cols2 = st.columns(3)
    for i, img in enumerate(generated_plans):
        segmented_img = apply_segmentation(segmentation_model, img, num_rooms)
        segmented_images.append(segmented_img)
        with cols2[i]:
            st.image(segmented_img, caption=f"Segmented Plan {i+1}", use_column_width=True)

    for i, img in enumerate(segmented_images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label=f"Download Plan {i+1}",
            data=byte_im,
            file_name=f"segmented_plan_{i+1}.png",
            mime="image/png"
        )
