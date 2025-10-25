import streamlit as st
import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from torchvision.utils import save_image
from datetime import datetime
from PIL import Image
import warnings
import cv2

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
IMG_SIZE = 256
OUTPUT_DIR = "web_generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Random Forest model
rf_model = joblib.load("room_predictor.joblib")

def predict_dwelling_type(area, bedrooms):
    features = np.array([[area, bedrooms]])
    return rf_model.predict(features)[0]

# Stage 1 Generator (DCGAN)
class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)

        def block(in_f, out_f):
            return nn.Sequential(
                nn.BatchNorm2d(in_f),
                nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
                nn.ReLU(True)
            )

        self.gen = nn.Sequential(
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.gen(out)

# Load Stage 1 Generator
G1 = DCGAN_Generator().to(DEVICE)
G1.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
G1.eval()

# Function to generate multiple floorplans with optional denoiser
def generate_final_plans(area, bedrooms, count=3, denoise=False):
    dwelling_type = predict_dwelling_type(area, bedrooms)
    images_paths = []

    for i in range(count):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = G1(z)
        
        # Convert tensor to numpy image
        img_np = img_tensor.squeeze().cpu().numpy()  # shape: (H, W)
        img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # convert from [-1,1] to [0,255]

        # Apply OpenCV denoiser if selected
        if denoise:
            img_np = cv2.fastNlMeansDenoising(img_np, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Save image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(OUTPUT_DIR, f"floorplan_{i+1}_{timestamp}.png")
        cv2.imwrite(file_path, img_np)
        images_paths.append(file_path)
        
    return dwelling_type, images_paths

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Streamlit UI
st.title("AI Floorplan Generator")
st.write("Generate AI-based floorplans based on total area and number of bedrooms.")

area = st.number_input("Enter Total Area (sq.ft.)", min_value=0.0, step=10.0)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=0, step=1)
denoise_option = st.checkbox("Apply Denoiser (OpenCV)")

if st.button("Generate Floorplans"):
    if area > 0 and bedrooms > 0:
        with st.spinner("Generating floorplans... Please wait."):
            dwelling_type, img_paths = generate_final_plans(area, bedrooms, count=3, denoise=denoise_option)
        st.success(f"Predicted Dwelling Type: {dwelling_type}")
        st.subheader("Generated Floorplans:")
        cols = st.columns(3)
        for i, path in enumerate(img_paths):
            img = Image.open(path)
            cols[i].image(img, caption=f"Floorplan {i+1}", use_container_width=True)
    else:
        st.error("Please enter valid area and bedroom values.")
