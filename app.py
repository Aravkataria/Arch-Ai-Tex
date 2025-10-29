import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
from PIL import Image
import warnings
import cv2

# -----------------------
# Streamlit Page Config
st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

# --- CONFIGURATION ---
DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256
# Suppress the warning about missing ScriptRunContext which can appear in certain environments
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# ===== GENERATOR (DCGAN Style for 256x256) =====
class DCGAN_Generator(nn.Module):
    # Helper method for a standard ConvTranspose block
    @staticmethod
    def block(in_f, out_f):
        return nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        # Initial projection from latent vector to spatial features (16x16)
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)
        
        # Generator stack: 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.gen = nn.Sequential(
            DCGAN_Generator.block(512, 256), # Output 32x32
            DCGAN_Generator.block(256, 128), # Output 64x64
            DCGAN_Generator.block(128, 64),  # Output 128x128
            nn.ConvTranspose2d(64, channels, 4, 2, 1), # Output 256x256
            nn.Tanh() # Output activation to map values to [-1, 1]
        )

    def forward(self, z):
        # Flatten and reshape to the starting spatial size
        out = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.gen(out)


@st.cache_resource
def load_all_models():
    """Loads the Random Forest classifier and the PyTorch GAN Generator."""
    rf_model_loaded = None
    try:
        # Load the Random Forest model used for dwelling type prediction
        rf_model_loaded = joblib.load("random_forest_classifier_model.joblib")
    except Exception as e:
        print(f"ERROR: Could not load Random Forest model (File 'random_forest_classifier_model.joblib' missing or corrupt): {e}")

    generator = DCGAN_Generator().to(DEVICE)
    try:
        # Load the pre-trained weights for the GAN generator
        generator.load_state_dict(torch.load("generator_epoch100.pth", map_location=DEVICE))
    except Exception as e:
        print(f"ERROR: Could not load GAN weights (File 'generator_epoch100.pth' missing or corrupt): {e}")
    
    generator.eval()
    return rf_model_loaded, generator

# Load models once when the app starts
RF_MODEL, GAN_MODEL = load_all_models()


def predict_dwelling_type(area, bedrooms, rf_model):
    """Predicts the dwelling type (e.g., house, apartment) based on area and bedrooms."""
    if rf_model is None:
        return "Prediction Model Missing"
    # The RF model expects a 2D array of features
    features = np.array([[area, bedrooms]])
    return rf_model.predict(features)[0]


def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    """Generates floor plans using the GAN and applies post-processing."""
    dwelling_type = predict_dwelling_type(area, bedrooms, rf_model)
    images = []

    for _ in range(count):
        # Generate random noise vector (latent input)
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        
        with torch.no_grad():
            img_tensor = generator(z)

        # Convert tensor (normalized -1 to 1) to uint8 NumPy array (0 to 255)
        img_np = img_tensor.squeeze().cpu().numpy()
        # Scale and clip the values safely before converting to 8-bit integer
        img_np = np.clip(((img_np + 1) * 127.5), 0, 255).astype(np.uint8)

        # Handle channel formatting for OpenCV/PIL (PyTorch is C, H, W. OpenCV/PIL expects H, W, C)
        if CHANNELS > 1 and img_np.ndim == 3 and img_np.shape[0] == CHANNELS:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Apply denoising if selected using OpenCV
        if denoise:
            if CHANNELS == 1:
                img_np = cv2.fastNlMeansDenoising(img_np, None, h=10, templateWindowSize=7, searchWindowSize=21)
            else:
                img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        # Convert the final NumPy array to a PIL Image
        mode = 'L' if CHANNELS == 1 else 'RGB'
        img_pil = Image.fromarray(img_np, mode)
        images.append(img_pil)
            
    return dwelling_type, images

# --- Streamlit UI Setup ---
st.markdown("""
<style>
    /* Custom styling for the primary button */
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
    /* Adding styling for the images to match the desired spaced appearance */
    .stImage > img {
        border-radius: 8px; /* Rounded corners */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.8, 0.2])  # Adjust proportions if needed

with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("### AI Floor Plan Generator")

with col2:
    st.image("qr.png", caption="Scan to explore", width=100)

# Input columns
col_len, col_wid = st.columns(2)
with col_len:
    house_length = st.number_input("Enter House Length (m)", min_value=10.0, max_value=10000.0, value=50.0, step=1.0)
with col_wid:
    house_width = st.number_input("Enter House Width/Depth (m)", min_value=10.0, max_value=10000.0, value=30.0, step=1.0)

area = house_length * house_width
st.markdown(f"**Calculated Total Area:** **{area:.2f} m²**")

bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=8, value=3, step=1)
denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False, help="Uses a non-local means filter to smooth noise from the generated image.")

st.markdown("---")

# Initialize session state for persistent data
if 'generated' not in st.session_state:
    st.session_state['generated'] = False
    st.session_state['images'] = []
    st.session_state['dwelling_type'] = None
    st.session_state['area'] = area
    st.session_state['bedrooms'] = bedrooms

if st.button("Generate Optimized Floor Plans", type="primary", use_container_width=True):
    if area > 0 and bedrooms >= 0:
        with st.spinner('AI is generating 3 floor plans...'):
            dwelling_type, floor_plan_images = generate_final_plans(
                GAN_MODEL, area, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
            )
            st.session_state['images'] = floor_plan_images
            st.session_state['generated'] = True
            st.session_state['dwelling_type'] = dwelling_type
            st.session_state['area'] = area
            st.session_state['bedrooms'] = bedrooms

        st.toast("Floor Plans Generated!")
    else:
        st.error("Please enter valid length, width, and bedroom values.")

# Display generated images with custom spacing
if st.session_state.get('generated'):
    st.divider()
    st.header("Generated Floor Plans")

    # Display predicted dwelling type
    if st.session_state['dwelling_type'] and st.session_state['dwelling_type'] != "Prediction Model Missing":
        st.subheader(f"Predicted Dwelling Type: {st.session_state['dwelling_type']}")
    
    # Custom column setup: 1 (image) | 0.1 (spacer) | 1 (image) | 0.1 (spacer) | 1 (image)
    cols = st.columns([1, 0.1, 1, 0.1, 1])
    images = st.session_state['images']

    # Indices for the actual image columns are 0, 2, and 4
    for i, col_index in enumerate([0, 2, 4]):
        # Ensure we don't try to access an image index that doesn't exist
        if i < len(images):
            img = images[i]
            
            # Prepare image buffer for download button
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            
            with cols[col_index]:
                # --- EDITED LINE: Use use_column_width=True to fill the column space and ensure spacing ---
                st.image(img, caption=f"Plan {i+1} (256x256)", use_column_width=True)
                
                # Download button below the image
                st.download_button(
                    label=f"Download Plan {i+1}",
                    data=img_buffer.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(st.session_state['area'])}sqft_Beds{st.session_state['bedrooms']}.png",
                    mime="image/png",
                    key=f"download_{i}",
                    use_container_width=True # Download button can safely use container width
                )

st.divider()
