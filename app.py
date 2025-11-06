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
from PIL import Image # Moved PIL import up for consistency

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

# ----------------------------
# DCGAN Generator Architecture (UNCHANGED)
# ----------------------------
class DCGAN_Generator(nn.Module):
    @staticmethod
    def block(in_f, out_f):
        return nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        # Initial projection layer
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16) 
        # Main upsampling layers
        self.gen = nn.Sequential(
            DCGAN_Generator.block(512, 256),
            DCGAN_Generator.block(256, 128),
            DCGAN_Generator.block(128, 64),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        # Reshape to start spatial generation
        out = self.fc(z).view(z.size(0), 512, 16, 16) 
        return self.gen(out)


# ----------------------------
# Model Loading (UNCHANGED)
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
# Dwelling Type Prediction (UNCHANGED)
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
# Floorplan Generation (GAN) (UNCHANGED)
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
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        
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
# FIX: Segmentation using Connected Components Analysis (CCA) (UNCHANGED)
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
    # We skip label 0 and apply colors to labels 1 through num_labels-1
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


# ---------------------------------------------------------------------
# UPDATED: Layout Generation (Optimized/Semantic) - Now includes more room types
# ---------------------------------------------------------------------
def generate_semantic_layout(total_area, num_rooms_input, property_type, plot_shape, plot_w, plot_h):
    total_area = float(total_area)
    # Total user-defined rooms: must include fixed rooms (Living, Kitchen, Bath) and Bedrooms
    # We'll treat the user's input as Bedrooms + 3 fixed rooms (Living, Kitchen, Washroom)
    
    # Base fixed rooms and their target area ratios
    # Ratios are adjusted to leave space for 'Stairs/Common', 'Porch', and 'Garden'
    fixed_ratios = {
        "Living/Dining": 0.25, 
        "Kitchen": 0.08, 
        "Washroom_1": 0.05
    }
    
    # Total fixed rooms *we* are allocating area for
    total_fixed_rooms = len(fixed_ratios)
    
    # The minimum required rooms for a functional layout
    min_functional_rooms = total_fixed_rooms + 1 # At least one bedroom

    # Calculate the number of bedrooms
    # User input rooms = (Bedrooms + fixed functional rooms) -> We'll assume the user
    # meant 'total *functional* rooms (excl. exterior space)'
    num_bedrooms = max(1, num_rooms_input - total_fixed_rooms) 
    
    # Calculate the area allocated to all functional rooms (interior)
    # A standard multiplier for interior functional space
    interior_functional_ratio = fixed_ratios["Living/Dining"] + fixed_ratios["Kitchen"] + fixed_ratios["Washroom_1"] + (num_bedrooms * 0.15)
    
    # Ensure a reasonable maximum interior ratio
    interior_functional_ratio = min(interior_functional_ratio, 0.75)
    
    # Recalculate fixed ratios based on the interior space remaining
    remaining_interior_ratio = interior_functional_ratio - sum(fixed_ratios.values())
    if num_bedrooms > 0 and remaining_interior_ratio > 0.01:
        per_bed_ratio = remaining_interior_ratio / num_bedrooms
    else:
        per_bed_ratio = 0.0

    rooms = []

    # 1. Add fixed functional rooms
    for name, ratio in fixed_ratios.items():
        rooms.append({"name": name, "area": round(total_area * ratio, 2), "type": name.split('_')[0]})

    # 2. Add bedrooms
    for i in range(num_bedrooms):
        rooms.append({"name": f"Bedroom_{i+1}", "area": round(total_area * per_bed_ratio, 2), "type": "Bedroom"})
    
    current_functional_area = sum(r["area"] for r in rooms)
    
    # 3. Add common/circulation space (Stairs/Lobby)
    stair_ratio = 0.08 if property_type != "Apartment" else 0.05
    rooms.append({"name": "Stairs/Lobby", "area": round(total_area * stair_ratio, 2), "type": "Common"})
    
    current_interior_area = current_functional_area + rooms[-1]["area"]
    
    # 4. Add exterior/utility space (Porch/Balcony, Garden)
    # The rest of the area is distributed between these "unnecessary" spaces.
    remaining_area_ratio = max(0.0, 1.0 - (current_interior_area / total_area))
    
    # Split the remaining area between Garden and Porch
    garden_ratio = remaining_area_ratio * 0.7 # Assume more garden space
    porch_ratio = remaining_area_ratio * 0.3
    
    rooms.append({"name": "Garden", "area": round(total_area * garden_ratio, 2), "type": "Garden"})
    rooms.append({"name": "Porch/Balcony", "area": round(total_area * porch_ratio, 2), "type": "Porch"})

    # Small correction for floating point errors
    current_sum = round(sum(r["area"] for r in rooms), 2)
    diff = round(total_area - current_sum, 2)
    if abs(diff) >= 0.01 and rooms:
        # Add any leftover area to the Living/Dining room (the largest)
        rooms[0]["area"] = round(rooms[0]["area"] + diff, 2)

    return {"rooms": rooms, "num_bedrooms": num_bedrooms}, ""


# ---------------------------------------------------------------------
# UPDATED: Layout Plotting - Now uses color-coded boxes for schematic layout
# ---------------------------------------------------------------------
def plot_layout(layout, plot_w, plot_h, title="Layout"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, plot_h)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw the boundary of the plot
    ax.add_patch(plt.Rectangle((0, 0), plot_w, plot_h, fill=False, edgecolor='black', linewidth=3))
    
    rooms = layout.get("rooms", [])
    total_area = sum(r["area"] for r in rooms)
    
    # --- Color Mapping for Room Types ---
    color_map = {
        "Living": "#f7d9a3",    # Light Orange/Peach
        "Kitchen": "#d3f7a3",   # Light Green
        "Washroom": "#a3d9f7",  # Light Blue
        "Bedroom": "#f7a3a3",   # Light Pink/Coral
        "Common": "#f7f7a3",    # Light Yellow (Stairs/Lobby)
        "Garden": "#82b35c",    # Deep Green (Exterior)
        "Porch": "#bdbdbd",     # Light Gray (Exterior)
        "utility": "#cccccc",   # Gray (Fallback)
    }
    
    # Scaling factor for area to plot size
    scale = (plot_w * plot_h) / max(total_area, 1.0)
    
    # Simple rectangular packing logic
    pad = min(plot_w, plot_h) * 0.01
    x, y = pad, pad
    row_h = 0
    
    # Sort rooms for better packing (e.g., largest first, or by type)
    rooms.sort(key=lambda r: r["area"], reverse=True) 

    for i, r in enumerate(rooms):
        desired_area = max(0.1, r["area"])
        rect_area = desired_area * scale
        
        # Simple non-optimized shape placement
        # Try to make the room more square-ish, or fit a target aspect ratio
        w = math.sqrt(rect_area) * 1.2 # Make it slightly rectangular
        h = rect_area / w
        
        # Check if the room fits in the current row
        if x + w + pad > plot_w:
            x = pad
            y += row_h + pad
            row_h = 0
        
        # Check if the room fits vertically
        if y + h + pad > plot_h:
            # We skip rooms that overflow the plot area
            continue
            
        room_type = r.get("type", "utility")
        color = color_map.get(room_type, color_map["utility"])

        # Draw the room
        rect = plt.Rectangle((x, y), w, h, 
                             facecolor=color, 
                             edgecolor='black', 
                             linewidth=1.5)
        ax.add_patch(rect)
        
        # Add the label
        ax.text(x + w / 2, y + h / 2, 
                f"{r['name']}\n{r['area']:.2f} m²", 
                ha='center', va='center', 
                fontsize=9, 
                color=('white' if room_type in ["Garden", "Porch"] else 'black'),
                fontweight='bold')
        
        # Move to the next position
        x += w + pad
        row_h = max(row_h, h)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add a simple legend for room types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='black', label=t) 
                       for t, c in color_map.items() if t in [r.get("type") for r in rooms]]
    
    if legend_elements:
        # Place legend outside the plot area
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), title="Room Types")
        
    return fig


# ----------------------------
# Streamlit UI (UNCHANGED from original, except for the calls)
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
    # Assuming 'QR.png' exists
    st.image("QR.png", width=110)
    st.markdown("<p style='font-size:13px; color:gray; text-align:right;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)

st.markdown("---")
mode = st.radio("Select Model:", ["GAN Generator", "Optimized Layout"], horizontal=True)

# ----------------------------
# Mode 1: GAN Floorplan Generation (UNCHANGED)
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
                # Segmentation now uses the updated function which only needs the image and room count
                seg_img = apply_segmentation(img, bedrooms) 
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                
                col.image(img, caption=f"Plan {i+1}", use_column_width=True)
                col.image(seg_img, caption=f"Segmented Plan {i+1}", use_container_width=True)
                col.download_button(
                    label=f"Download Plan {i+1}",
                    data=buf.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png", 
                    mime="image/png",
                )

# ----------------------------
# Mode 2: Optimized Layout (UPDATED CALLS)
# ----------------------------
else:
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        # Note: num_rooms_input now represents (Bedrooms + 3 Fixed Rooms)
        num_rooms_input = st.number_input("Enter Total Number of Functional Rooms (e.g., 4 for 1 Bed, 5 for 2 Bed)", min_value=1, value=5)
        
    st.markdown("<p style='font-size:13px; color:gray;'>Functional rooms include the Living Area, Kitchen, Washroom, and all Bedrooms. Exterior spaces (Garden, Porch) are calculated automatically.</p>", unsafe_allow_html=True)
    
    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Bungalow"])
    plot_shape = st.selectbox("Plot Shape", ["Square", "Rectangular"])
    
    colW, colH = st.columns(2)
    with colW:
        plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=12.0)
    with colH:
        plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=10.0)
        
    if st.button("Generate Optimized Layout"):
        with st.spinner("Generating conceptual layout and area distribution..."):
            
            # Generate semantic layout and extract the calculated number of bedrooms
            layout, _ = generate_semantic_layout(total_area, num_rooms_input, property_type, plot_shape, plot_w, plot_h)
            
            # Use M² and calculated bedrooms for dwelling type prediction
            dwelling_type = predict_dwelling_type(total_area, layout["num_bedrooms"], RF_MODEL) 
            
            st.success(f"Predicted Dwelling Type: **{dwelling_type}** | **{layout['num_bedrooms']} Bedroom Design**")
            
            # Plot the new, schematic layout
            fig = plot_layout(layout, plot_w, plot_h, f"{property_type} Conceptual Layout")
            st.pyplot(fig)
