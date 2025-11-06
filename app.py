import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models.segmentation as models
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2
import math
import warnings
from PIL import Image

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

# ----------------------------
# DCGAN Generator Architecture
# ----------------------------
class DCGAN_Generator(nn.Module):
    @staticmethod
    def block(in_f, out_f):
        return nn.Sequential(
            nn.ConvTranspose2d(in_f, out_f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_f),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.gen = nn.Sequential(
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
        return self.gen(z)

# ----------------------------
# Model Loading (FIXED LOGIC)
# ----------------------------
@st.cache_resource
def load_models():
    rf_model = None
    generator = DCGAN_Generator().to(DEVICE)
    try:
        # Load the Random Forest model for dwelling type prediction
        rf_model = joblib.load("room_predictor.joblib")
    except Exception:
        # Handle missing RF model gracefully
        rf_model = None
    
    loaded = False
    # Attempt to load the generator weights from common names
    for fname in ("generator_epoch100.pth", "generator_epoch_100.pth", "generator.pth"):
        try:
            state_dict = torch.load(fname, map_location=DEVICE)
            # Load weights, non-strict loading is safer if architecture changed slightly
            generator.load_state_dict(state_dict, strict=False) 
            loaded = True
            # *** FIX: Break out of the loop immediately after a successful load ***
            break 
        except FileNotFoundError:
            # Silently ignore not found errors, try the next one
            continue
        except Exception as e:
            # Report other loading errors and continue to try the next file
            st.warning(f"Error loading generator model {fname}: {e}")
            continue

    if not loaded:
        # This error is now only displayed if *all* attempts failed.
        st.error("GAN generator weights not found or failed to load. The output will likely be noise.")

    generator.eval()
    # DeepLabV3 model is no longer necessary for the segmentation fix, 
    # but we will return None to avoid breaking the calling structure.
    return rf_model, generator, None # SEG_MODEL is now None


# Rerunning model loading with the updated function.
RF_MODEL, GAN_MODEL, SEG_MODEL = load_models()


# ----------------------------
# Dwelling Type Prediction
# ----------------------------
def predict_dwelling_type(area, bedrooms, rf_model):
    """Predicts dwelling type. Assumes area is in m² for consistency with RF training."""
    if rf_model is None:
        return "Unknown Type (RF model missing)"
    try:
        # NOTE: Assumes RF model was trained on M² area inputs
        features = np.array([[float(area), int(bedrooms)]]) 
        return rf_model.predict(features)[0]
    except Exception:
        return "Prediction Failed"


# ----------------------------
# Floorplan Generation (GAN)
# ----------------------------
def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    """
    Generates floor plans.
    The 'area' parameter MUST be in M² (square meters) for consistency with conditioning.
    """
    # Prediction uses M²
    dwelling_type = predict_dwelling_type(area, bedrooms, rf_model) 
    images = []

    # Ensure minimum area for seed calculation
    if area < 100:
        area = 100

    pixel_area = area / (IMG_SIZE * IMG_SIZE)
    # Area is used to condition the seed base (critical conditioning factor)
    seed_base = int(area * 10 + bedrooms * 1234)

    for i in range(count):
        torch.manual_seed(seed_base + i)
        # Standard DCGAN noise shape
        z = torch.randn(1, LATENT_DIM, 1, 1).to(DEVICE)
        
        with torch.no_grad():
            img_tensor = generator(z)
            img_np = img_tensor.squeeze().cpu().numpy()
            
            # Post-processing: map GAN output range [-1, 1] to pixel range [0, 255]
            img_np = np.clip(((img_np + 1) * 127.5), 0, 255).astype(np.uint8)

            if CHANNELS > 1 and img_np.ndim == 3 and img_np.shape[0] == CHANNELS:
                img_np = np.transpose(img_np, (1, 2, 0))

            if denoise:
                if CHANNELS == 1:
                    img_np = cv2.fastNlMeansDenoising(img_np, None, h=10)
                else:
                    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10)

            mode = 'L' if CHANNELS == 1 else 'RGB'
            img = Image.fromarray(img_np, mode)
            images.append(img)

    return dwelling_type, images, pixel_area


# ----------------------------
# FIX: Segmentation using Connected Components Analysis (CCA)
# ----------------------------
def apply_segmentation(image, num_rooms):
    """
    Applies Connected Components Analysis (CCA) to identify and color separate rooms 
    in the black-and-white floorplan image.
    """
    if image.mode != "L":
        # Convert to grayscale NumPy array for OpenCV processing
        img_cv = np.array(image.convert("L"))
    else:
        img_cv = np.array(image)

    # 1. Binarization: Walls are black (0), rooms are white (255)
    # We binarize to clearly separate lines (walls) from empty space (rooms).
    _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY_INV) 

    # 2. Connected Components Analysis to label each "room" (connected white area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # 3. Create the colored segmentation map
    seg_rgb = np.zeros((*img_cv.shape, 3), dtype=np.uint8)
    
    # Define a list of appealing colors for rooms
    room_colors = [
        (255, 199, 107),  # Light Orange/Peach
        (130, 202, 157),  # Light Green/Mint
        (174, 199, 232),  # Light Blue/Periwinkle
        (255, 152, 150),  # Light Red/Coral
        (197, 176, 213),  # Light Purple/Lavender
        (255, 237, 111),  # Light Yellow
        (188, 189, 34),   # Olive
        (140, 86, 75),    # Brown
    ]
    
    # Label 0 is typically the background (the largest component, often the exterior)
    for i in range(1, num_labels):
        # Optional: Skip very small components (noise)
        if stats[i, cv2.CC_STAT_AREA] < 50: 
            continue 

        # Pick a color based on component index
        color_index = (i - 1) % len(room_colors)
        color = room_colors[color_index]
        
        # Apply the color to all pixels belonging to this component (room)
        seg_rgb[labels == i] = color

    # Convert back to PIL Image
    seg_pil = Image.fromarray(seg_rgb).resize(image.size)
    return seg_pil


# ----------------------------
# Layout Generation (Optimized/Semantic)
# ----------------------------
def generate_semantic_layout(total_area, num_rooms_input, property_type, plot_shape, plot_w, plot_h):
    total_area = float(total_area)
    num_rooms_input = max(0, int(num_rooms_input))
    
    # Fixed rooms and their area ratios
    fixed_ratios = {"living+dining": 0.28, "kitchen": 0.08, "bathroom": 0.06}
    fixed_total = sum(fixed_ratios.values())
    
    # Calculate the number of bedrooms (total rooms - fixed rooms)
    num_bedrooms = max(0, num_rooms_input - len(fixed_ratios)) 
    
    remaining_ratio = max(0.0, 1.0 - fixed_total)
    rooms = []

    for name, ratio in fixed_ratios.items():
        rooms.append({"name": name, "area": round(total_area * ratio, 2)})

    if num_bedrooms > 0:
        per_bed_ratio = remaining_ratio / num_bedrooms
        for i in range(num_bedrooms):
            rooms.append({"name": f"bedroom_{i+1}", "area": round(total_area * per_bed_ratio, 2)})
    elif remaining_ratio > 0.01:
        # If no bedrooms but remaining area, assign it to utility
        rooms.append({"name": "utility/other", "area": round(total_area * remaining_ratio, 2)})

    # Small correction for floating point errors
    current_sum = round(sum(r["area"] for r in rooms), 2)
    diff = round(total_area - current_sum, 2)
    if abs(diff) >= 0.01 and rooms:
        rooms[0]["area"] = round(rooms[0]["area"] + diff, 2)

    return {"rooms": rooms, "num_bedrooms": num_bedrooms}, ""


# ----------------------------
# Layout Plotting
# ----------------------------
def plot_layout(layout, plot_w, plot_h, title="Layout"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, plot_h)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), plot_w, plot_h, fill=False, edgecolor='black', linewidth=1.2))
    
    rooms = layout.get("rooms", [])
    total_area = sum(r["area"] for r in rooms)
    
    # Scaling factor for area to plot size
    scale = (plot_w * plot_h) / max(total_area, 1.0)
    
    pad = min(plot_w, plot_h) * 0.02
    x, y = pad, pad
    row_h = 0
    colors = ["#f4cccc", "#d9ead3", "#cfe2f3", "#fff2cc", "#d9d2e9", "#c2f0c2"]
    
    for i, r in enumerate(rooms):
        desired_area = max(0.1, r["area"])
        rect_area = desired_area * scale
        
        # Simple non-optimized shape placement
        w = math.sqrt(rect_area) * 1.3 # Give it a slightly rectangular shape
        h = rect_area / w
        
        # Check if the room fits in the current row
        if x + w + pad > plot_w:
            x = pad
            y += row_h + pad
            row_h = 0
        
        # Check if the room fits vertically
        if y + h + pad > plot_h:
            break
            
        rect = plt.Rectangle((x, y), w, h, facecolor=colors[i % len(colors)], edgecolor='black', linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{r['name']}\n{r['area']} m²", ha='center', va='center', fontsize=8)
        
        x += w + pad
        row_h = max(row_h, h)
        
    ax.set_title(title)
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 1.05em;
    transition: all 0.15s;
    border: none;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
}
.stImage > img {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("AI Floor Plan Generator")
with col2:
    # NOTE: Using a placeholder image since the 'QR.png' file is not available
    st.image("https://placehold.co/110x110/38761D/ffffff?text=LOGO", width=110) 
    st.markdown("<p style='font-size:13px; color:gray; text-align:right;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)

st.markdown("---")
mode = st.radio("Select Model:", ["GAN Generator", "Optimized Layout"], horizontal=True)

# ----------------------------
# Mode 1: GAN Floorplan Generation
# ----------------------------
if mode == "GAN Generator":
    col_len, col_wid = st.columns(2)
    with col_len:
        house_length = st.number_input("Enter House Length (m)", min_value=10.0, value=50.0, step=1.0)
    with col_wid:
        house_width = st.number_input("Enter House Width (m)", min_value=10.0, value=30.0, step=1.0)
    
    area_m2 = house_length * house_width
    if area_m2 < 100:
        area_m2 = 100
        
    area_sqft = area_m2 * 10.7639
    
    st.markdown(f"**Calculated Total Area:** {area_m2:.2f} m² (≈ {area_sqft:.0f} sq ft)**")
    
    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)
    
    if st.button("Generate Floorplans", type="primary", use_container_width=True):
        
        dwelling_type, floor_plan_images, pixel_area = generate_final_plans(
            GAN_MODEL, area_m2, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
        )
        
        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        st.markdown(f"**Area to Pixel Ratio:** 1 pixel ≈ {pixel_area:.4f} m²")
        st.markdown("Generated Floorplans:")
        
        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(floor_plan_images):
                img = floor_plan_images[i]
                # Segmentation uses the CCA logic
                seg_img = apply_segmentation(img, bedrooms) 
                
                # Download button for the Original GAN image
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                
                col.image(img, caption=f"Plan {i+1}", use_column_width=True)
                col.image(seg_img, caption=f"Segmented Plan {i+1}", use_column_width=True)

                # Download button for the Segmented image
                seg_buf = io.BytesIO()
                seg_img.save(seg_buf, format="PNG")
                col.download_button(
                    label=f"Download Seg. Plan {i+1}",
                    data=seg_buf.getvalue(),
                    file_name=f"segmented_plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png", 
                    mime="image/png",
                )
                
# ----------------------------
# Mode 2: Optimized Layout
# ----------------------------
else:
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        num_rooms_input = st.number_input("Enter Total Number of Rooms", min_value=1, value=3)
        
    st.markdown("<p style='font-size:13px; color:gray;'>Note: The total number of rooms includes the kitchen and bathroom.</p>", unsafe_allow_html=True)
    
    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Bungalow"])
    plot_shape = st.selectbox("Plot Shape", ["Square", "Rectangular"])
    
    colW, colH = st.columns(2)
    with colW:
        plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=10.0)
    with colH:
        plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=10.0)
        
    if st.button("Generate Optimized Layout"):
        with st.spinner("Generating layout..."):
            
            # Generate semantic layout and extract the calculated number of bedrooms
            layout, _ = generate_semantic_layout(total_area, num_rooms_input, property_type, plot_shape, plot_w, plot_h)
            
            # Use M² and calculated bedrooms for dwelling type prediction
            dwelling_type = predict_dwelling_type(total_area, layout["num_bedrooms"], RF_MODEL) 
            
            st.success(f"Predicted Dwelling Type: **{dwelling_type}**")
            fig = plot_layout(layout, plot_w, plot_h, f"{property_type} Layout")
            st.pyplot(fig)
