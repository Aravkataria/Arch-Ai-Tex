import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

st.set_page_config(page_title="Arch-Ai-Tex", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
IMG_SIZE = 256

# ---------------------------
# Define Generator (same as before)
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(LATENT_DIM, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, IMG_SIZE, IMG_SIZE)
        return img

# ---------------------------
# Load generator model
# ---------------------------
@st.cache_resource
def load_generator():
    model = Generator().to(DEVICE)
    if os.path.exists("generator_epoch100.pth"):
        model.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    model.eval()
    return model

G = load_generator()

# ---------------------------
# Segmentation function (mock version for now)
# ---------------------------
def segment_image(img):
    img_np = np.array(img)
    h, w, _ = img_np.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Random segmentation for demonstration
    n_rooms = np.random.randint(3, 6)
    for i in range(1, n_rooms + 1):
        cv2.rectangle(mask, 
                      (np.random.randint(0, w//2), np.random.randint(0, h//2)), 
                      (np.random.randint(w//2, w), np.random.randint(h//2, h)), 
                      i, -1)
    return mask

# ---------------------------
# Apply colors to segmentation
# ---------------------------
def colorize_segmentation(mask, highlight_rooms=None):
    unique_ids = np.unique(mask)
    colors = {i: np.random.randint(0, 255, 3) for i in unique_ids if i != 0}
    seg_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for i, color in colors.items():
        if highlight_rooms and i not in highlight_rooms:
            continue
        seg_img[mask == i] = color
    return Image.fromarray(seg_img)

# ---------------------------
# Generate image
# ---------------------------
def generate_image():
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        gen_img = G(z).cpu()
    img = (gen_img.squeeze().numpy() + 1) / 2.0
    img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    return img

# ---------------------------
# UI
# ---------------------------
st.title("Arch-Ai-Tex: Floor Plan Generator + Room Segmentation")

col1, col2 = st.columns([1, 2])
with col1:
    user_rooms = st.number_input("Enter number of rooms to highlight:", min_value=1, max_value=10, value=3)
    generate_button = st.button("Generate and Segment")

if generate_button:
    with st.spinner("Generating floor plan..."):
        gen_img = generate_image()
        seg_mask = segment_image(gen_img)
        seg_img = colorize_segmentation(seg_mask)
        highlighted_img = colorize_segmentation(seg_mask, highlight_rooms=list(range(1, user_rooms + 1)))

    # Horizontal layout
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(gen_img, caption="Generated Floor Plan", use_container_width=True)
    with col_b:
        st.image(seg_img, caption="Segmented Floor Plan", use_container_width=True)
    with col_c:
        st.image(highlighted_img, caption=f"Highlighted {user_rooms} Room(s)", use_container_width=True)
