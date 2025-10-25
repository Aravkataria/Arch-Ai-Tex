import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from datetime import datetime
import cv2
from PIL import Image
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
IMG_SIZE = 256
# Change output directory name since we are not generating 1024px images anymore
OUTPUT_DIR = "final_generated_images_256" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Random Forest model
# NOTE: Update the path if running on a different system
rf_model = joblib.load("room_predictor.joblib")

def predict_dwelling_type(area, bedrooms):
    """Predicts the dwelling type using the loaded Random Forest model."""
    features = np.array([[area, bedrooms]])
    return rf_model.predict(features)[0]

# Stage 1 Generator (Original 256px)
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
# NOTE: Update the path if running on a different system
G1 = DCGAN_Generator().to(DEVICE)
G1.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
G1.eval()

# NOTE: Removed RealESRGAN import, model loading, and the upscale_with_esrgan function.

# Function to generate floorplan (original 256x256)
def generate_final_plan(area, bedrooms, denoise=False):
    """Generates a 256x256 floorplan using the DCGAN and optionally denoises it."""
    dwelling_type = predict_dwelling_type(area, bedrooms)

    # Generate image from GAN
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        img_tensor = G1(z)
    
    # Convert tensor to numpy image (0-255 range, uint8)
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)

    # Apply OpenCV denoiser if selected
    if denoise:
        # Use cv2.fastNlMeansDenoising for grayscale images (assuming output is 1 channel)
        if len(img_np.shape) == 2:
            img_np = cv2.fastNlMeansDenoising(img_np, None, h=10, templateWindowSize=7, searchWindowSize=21)
        # If it's a 3-channel image, use cv2.fastNlMeansDenoisingColored
        elif img_np.shape[0] == 3:
            img_np = np.moveaxis(img_np, 0, -1) # Convert from C, H, W to H, W, C for cv2
            img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
            img_np = np.moveaxis(img_np, -1, 0) # Convert back to C, H, W
    
    # NOTE: Removed the upscaling block entirely. The image remains 256x256.

    # Save image
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(OUTPUT_DIR, f"floorplan_{timestamp}_256px.png")
    # Ensure image is in HxW or HxWxC format for cv2.imwrite
    if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3]: # Check if it's C, H, W
        img_to_save = np.moveaxis(img_np, 0, -1) if img_np.shape[0] == 3 else img_np.squeeze()
    else: # Assume HxW or HxWxC
        img_to_save = img_np
        
    cv2.imwrite(file_path, img_to_save)

    return dwelling_type, file_path

# Main program
def main():
    print("=== AI Floorplan Generator (256x256) ===")
    area = float(input("Enter Total Area (sq.ft.): "))
    bedrooms = int(input("Enter Number of Bedrooms: "))
    denoise_input = input("Apply Denoiser (y/n)? ").lower()
    
    # NOTE: Removed the upscale input.

    denoise = denoise_input == "y"
    # upscale is no longer a variable

    if area <= 0 or bedrooms <= 0:
        print("Error: Please enter valid area and bedroom values.")
        return

    print("Generating 256x256 floorplan... Please wait.")
    # Call the modified generation function
    dwelling_type, img_path = generate_final_plan(area, bedrooms, denoise=denoise) 
    
    print(f"Predicted Dwelling Type: {dwelling_type}")
    print(f"Generated Floorplan saved at: {img_path}")

if __name__ == "__main__":
    main()