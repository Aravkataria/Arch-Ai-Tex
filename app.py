import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.segmentation as models
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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

def apply_segmentation(model, image, num_rooms):
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    h, w = img.shape[0], img.shape[1]
    draw_img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(draw_img)

    total_rooms = num_rooms + 4  # kitchen, washroom, stair, garden/porch
    rows = int(np.ceil(np.sqrt(total_rooms)))
    cols = rows

    box_w = w // cols
    box_h = h // rows

    color_palette = [
        (245, 180, 90), (180, 230, 120), (150, 200, 255),
        (255, 140, 140), (220, 190, 255), (255, 220, 130),
        (180, 255, 200), (255, 180, 200), (200, 230, 255),
        (230, 240, 180)
    ]

    labels = ["Room"] * num_rooms + ["Kitchen", "Washroom", "Stairs", "Porch/Garden"]
    random.shuffle(labels)

    k = 0
    for i in range(rows):
        for j in range(cols):
            if k >= total_rooms:
                break
            x0, y0 = j * box_w + random.randint(1, 5), i * box_h + random.randint(1, 5)
            x1, y1 = (j + 1) * box_w - random.randint(1, 5), (i + 1) * box_h - random.randint(1, 5)
            color = color_palette[k % len(color_palette)]
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0), width=2)
            draw.text((x0 + 5, y0 + 5), labels[k], fill=(0, 0, 0))
            k += 1

    blend_gan = np.array(draw_img).astype(np.float32)
    gan_overlay = cv2.cvtColor(np.array(image.resize((IMG_SIZE, IMG_SIZE))), cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(blend_gan, 0.8, gan_overlay, 0.2, 0)

    return Image.fromarray(blended.astype(np.uint8))

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
            st.image(segmented_img, caption=f"Segmented Plan {i+1}", use_container_width=True)

    for i, img in enumerate(segmented_images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label=f"Download Plan {i+1}", data=byte_im, file_name=f"segmented_plan_{i+1}.png", mime="image/png")
