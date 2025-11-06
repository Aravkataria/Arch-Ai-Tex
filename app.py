import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.segmentation as models
import numpy as np
import io
from PIL import Image, ImageDraw, ImageFont
import cv2
import random
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
        model.load_state_dict(state_dict)
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
        fake_images = model(noise)
    fake_images = (fake_images * 0.5 + 0.5).cpu()
    images = [T.ToPILImage()(img.squeeze(0)) for img in fake_images]
    return images

def draw_box_layout(num_rooms, overlay_image=None):
    total_rooms = num_rooms + 4
    labels = [f"Room {i+1}" for i in range(num_rooms)] + ["Kitchen", "Washroom", "Stairs", "Porch"]
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    grid_size = int(np.ceil(np.sqrt(total_rooms + 1)))
    cell_w = IMG_SIZE // grid_size
    cell_h = IMG_SIZE // grid_size
    colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200), (200, 255, 255),
              (255, 220, 180), (220, 200, 255), (255, 180, 220), (210, 255, 210), (255, 210, 255)]
    font = ImageFont.load_default()
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if idx < total_rooms:
                x0 = c * cell_w
                y0 = r * cell_h
                x1 = x0 + cell_w - 5
                y1 = y0 + cell_h - 5
                color = colors[idx % len(colors)]
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0), width=2)
                text = labels[idx]
                tw, th = draw.textsize(text, font=font)
                draw.text((x0 + (cell_w - tw) / 2, y0 + (cell_h - th) / 2), text, fill=(0, 0, 0), font=font)
                idx += 1
    if idx < grid_size * grid_size:
        draw.rectangle([0, IMG_SIZE - cell_h, IMG_SIZE, IMG_SIZE], fill=(180, 255, 180), outline=(0, 0, 0), width=2)
        draw.text((10, IMG_SIZE - cell_h + 10), "Garden", fill=(0, 0, 0), font=font)
    if overlay_image:
        overlay_resized = overlay_image.resize((IMG_SIZE, IMG_SIZE)).convert("RGBA")
        base = img.convert("RGBA")
        overlay_resized.putalpha(70)
        img = Image.alpha_composite(base, overlay_resized)
    return img.convert("RGB")

st.title("Arch-Ai-Tex")

num_rooms = st.slider("Number of rooms", 1, 10, 3)
use_overlay = st.checkbox("Use AI texture overlay")
generator_model = load_generator()
segmentation_model = load_segmentation_model()

if st.button("Generate Floorplans"):
    generated_plans = generate_floorplan(generator_model, LATENT_DIM, num_images=3)
    st.subheader("Generated Floorplans:")
    cols = st.columns(3)
    for i, img in enumerate(generated_plans):
        with cols[i]:
            st.image(img, caption=f"Plan {i+1}", use_container_width=True)
    st.subheader("Structured Floorplans:")
    structured_images = []
    cols2 = st.columns(3)
    for i, gan_img in enumerate(generated_plans):
        structured_img = draw_box_layout(num_rooms, overlay_image=gan_img if use_overlay else None)
        structured_images.append(structured_img)
        with cols2[i]:
            st.image(structured_img, caption=f"Structured Plan {i+1}", use_container_width=True)
    for i, img in enumerate(structured_images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label=f"Download Plan {i+1}", data=byte_im, file_name=f"structured_plan_{i+1}.png", mime="image/png")
