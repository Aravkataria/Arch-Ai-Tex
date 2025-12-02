import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
import warnings
import os

warnings.filterwarnings("ignore", message="missing ScriptRunContext")
st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

# ---------------------------
# Config / constants
# ---------------------------
DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256  # generator output resolution (assumed square)
GEN_WEIGHTS_PATH = "generator_epoch100.pth"
RF_MODEL_PATH = "room_predictor.joblib"

# ---------------------------
# Generator definition
# ---------------------------

class DCGAN_Generator(nn.Module):
    @staticmethod
    def block(in_f, out_f):
        return nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ConvTranspose2d(in_f, out_f, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=LATENT_DIM, channels=CHANNELS):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.gen = nn.Sequential(
            DCGAN_Generator.block(512, 256),  # 16 -> 32
            DCGAN_Generator.block(256, 128),  # 32 -> 64
            DCGAN_Generator.block(128, 64),   # 64 -> 128
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), 512, 16, 16)
        return self.gen(out)


# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource
def load_models():
    rf_model = None
    generator = DCGAN_Generator().to(DEVICE)
    if os.path.exists(RF_MODEL_PATH):
        try:
            rf_model = joblib.load(RF_MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load RF model `{RF_MODEL_PATH}`: {e}")
            rf_model = None
    if os.path.exists(GEN_WEIGHTS_PATH):
        try:
            state_dict = torch.load(GEN_WEIGHTS_PATH, map_location=DEVICE)
            generator.load_state_dict(state_dict)
        except Exception as e:
            st.warning(f"Could not load generator weights `{GEN_WEIGHTS_PATH}`: {e}")
    else:
        st.info(f"Generator weights not found at `{GEN_WEIGHTS_PATH}` — using randomly initialized generator.")
    generator.eval()
    return rf_model, generator

RF_MODEL, GAN_MODEL = load_models()

# ---------------------------
# Prediction + generation
# ---------------------------
def predict_dwelling_type(area_m2, bedrooms, rf_model):
    if rf_model is None:
        return "Unknown (no RF model)"
    try:
        features = np.array([[float(area_m2), int(bedrooms)]])
        pred = rf_model.predict(features)
        return pred[0]
    except Exception:
        return "Prediction Failed"

def generate_final_plans(generator, area_m2, bedrooms, count=3, denoise=False, rf_model=None):
    dwelling_type = predict_dwelling_type(area_m2, bedrooms, rf_model)
    images = []

    area_m2 = max(100.0, float(area_m2)) 
    pixel_area = area_m2 / (IMG_SIZE * IMG_SIZE)

    for _ in range(count):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = generator(z)
            img_np = img_tensor.squeeze().cpu().numpy()

            if CHANNELS == 1:
                if img_np.ndim == 3:
                    img_np = img_np[0]
                img_np = np.clip(((img_np + 1) * 127.5), 0, 255).astype(np.uint8)
            else:
                if img_np.ndim == 3 and img_np.shape[0] == CHANNELS:
                    img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip(((img_np + 1) * 127.5), 0, 255).astype(np.uint8)

            if denoise:
                try:
                    if CHANNELS == 1:
                        img_np = cv2.fastNlMeansDenoising(img_np, None, h=10)
                    else:
                        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10)
                except Exception:
                    pass

            mode = 'L' if CHANNELS == 1 else 'RGB'
            img = Image.fromarray(img_np, mode)
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(img)

    return dwelling_type, images, pixel_area

# ---------------------------
# Segmentation logic
# ---------------------------
def apply_segmentation(image):
    if image.mode != "L":
        img_cv = np.array(image.convert("L"))
    else:
        img_cv = np.array(image)
    
    # 1. Thresholding to separate rooms/walls (assumes white walls/black rooms or vice versa)
    # Using adaptive thresholding might be better for real-world images, but simple binary works for GAN output
    _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Find connected components (each component is a potential room)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    
    seg_rgb = np.zeros((*img_cv.shape, 3), dtype=np.uint8)
    
    # Define colors for different room segments
    room_colors = [
        (255, 199, 107), # Light Orange
        (130, 202, 157), # Light Green
        (174, 199, 232), # Light Blue
        (255, 152, 150), # Light Red
        (197, 176, 213), # Light Purple
        (255, 237, 111), # Yellow
        (188, 189, 34),  # Olive
        (140, 86, 75),   # Brown
    ]
    
    # 3. Color each significant component
    for i in range(1, num_labels):
        # Filter out small components (noise/small wall gaps)
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        
        color_index = (i - 1) % len(room_colors)
        color = room_colors[color_index]
        
        # Apply the color to all pixels belonging to this component
        seg_rgb[labels == i] = color
        
    seg_pil = Image.fromarray(seg_rgb).resize(image.size)
    return seg_pil

# ---------------------------
# Semantic layout generator + visualiser
# ---------------------------
def generate_semantic_layout(total_area, num_bedrooms, property_type=None, plot_shape=None, plot_w=None, plot_h=None):
    total_area = float(total_area)
    num_bedrooms = max(0, int(num_bedrooms))

    fixed_ratios = {"living+dining": 0.28, "kitchen": 0.08, "bathroom": 0.06}
    fixed_total = sum(fixed_ratios.values())

    remaining_ratio = max(0.0, 1.0 - fixed_total)

    rooms = []
    for name, ratio in fixed_ratios.items():
        rooms.append({"name": name, "area": round(total_area * ratio, 2)})

    actual_bedrooms = num_bedrooms
    if actual_bedrooms > 0:
        per_bed_ratio = remaining_ratio / actual_bedrooms
        for i in range(actual_bedrooms):
            rooms.append({"name": f"bedroom_{i+1}", "area": round(total_area * per_bed_ratio, 2)})
    else:
        rooms.append({"name": "utility/other", "area": round(total_area * remaining_ratio, 2)})

    current_sum = round(sum(r["area"] for r in rooms), 2)
    diff = round(total_area - current_sum, 2)
    if abs(diff) >= 0.01 and len(rooms) > 0:
        rooms[0]["area"] = round(rooms[0]["area"] + diff, 2)

    return {"rooms": rooms}, ""

def plot_layout(layout, plot_w, plot_h, title="Layout"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, plot_h)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(plt.Rectangle((0, 0), plot_w, plot_h, fill=False, edgecolor='black', linewidth=1.2))

    rooms = layout.get("rooms", [])
    total_area = sum(r["area"] for r in rooms) if rooms else 1.0

    scale = (plot_w * plot_h) / max(total_area, 1.0)

    pad = min(plot_w, plot_h) * 0.02
    x, y = pad, pad
    row_h = 0

    colors = ["#f4cccc", "#d9ead3", "#cfe2f3", "#fff2cc", "#d9d2e9", "#c2f0c2"]

    for i, r in enumerate(rooms):
        desired_area = max(0.1, r["area"])
        rect_area = desired_area * scale

        w = math.sqrt(rect_area) * 1.0
        h = rect_area / max(w, 1e-6)

        if x + w + pad > plot_w:
            x = pad
            y += row_h + pad
            row_h = 0

        if y + h + pad > plot_h:
            break

        rect = plt.Rectangle((x, y), w, h, facecolor=colors[i % len(colors)], edgecolor='black', linewidth=1.1)
        ax.add_patch(rect)

        ax.text(x + w / 2, y + h / 2, f"{r['name']}\n{r['area']} m²", ha='center', va='center', fontsize=8)

        x += w + pad
        row_h = max(row_h, h)

    ax.set_title(title)
    plt.tight_layout()
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 1.0em;
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
    if os.path.exists("QR.png"):
        st.image("QR.png", width=110)
    else:
        st.empty()
st.markdown("<p style='font-size:13px; color:gray; text-align:right;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)
st.markdown("---")

mode = st.radio("Select Model:", ["GAN Generator", "Optimized Layout"], horizontal=True)

if mode == "GAN Generator":
    col_len, col_wid = st.columns(2)
    with col_len:
        house_length = st.number_input("Enter House Length (m)", min_value=1.0, value=10.0, step=0.5)
    with col_wid:
        house_width = st.number_input("Enter House Width (m)", min_value=1.0, value=12.0, step=0.5)

    area_m2 = float(house_length * house_width)
    if area_m2 < 100:
        area_m2 = max(100.0, area_m2)

    area_sqft = area_m2 * 10.7639

    st.markdown(f"**Calculated Total Area:** {area_m2:.2f} m² (≈ {area_sqft:.0f} sq ft)")

    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=0, value=3, step=1)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)
    
    if st.button("Generate Floorplans"):
        dwelling_type, floor_plan_images, pixel_area = generate_final_plans(
            GAN_MODEL, area_m2, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
        )

        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        st.markdown(f"**Area to Pixel Ratio:** 1 pixel ≈ {pixel_area:.4f} m²")
        st.markdown("Generated Floorplans:")

        cols = st.columns(3)
        seg_cols = st.columns(3) # New row for segmented images

        for i, img in enumerate(floor_plan_images):
            # Display Original GAN image
            cols[i].image(img, caption=f"Plan {i+1} (GAN Output)", use_column_width=True)
            
            # Perform and display Segmentation
            seg_img = apply_segmentation(img)
            seg_cols[i].image(seg_img, caption=f"Plan {i+1} (Segmented)", use_column_width=True)

            # Download button for the original image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            cols[i].download_button(
                label=f"Download Plan {i+1}",
                data=buf.getvalue(),
                file_name=f"plan_{i+1}_Area{int(area_m2)}sqm_Beds{bedrooms}.png",
                mime="image/png",
            )
            
            # Optional: Download button for the segmented image
            seg_buf = io.BytesIO()
            seg_img.save(seg_buf, format="PNG")
            seg_buf.seek(0)
            seg_cols[i].download_button(
                label=f"Download Segmented {i+1}",
                data=seg_buf.getvalue(),
                file_name=f"plan_{i+1}_SEGMENTED_Area{int(area_m2)}sqm_Beds{bedrooms}.png",
                mime="image/png",
            )


elif mode == "Optimized Layout":
    st.header("Optimized Layout Generator")
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        num_rooms_input = st.number_input("Enter Total Number of Rooms (bedrooms)", min_value=0, value=3, step=1)

    plot_w = st.number_input("Plot Width (m) - for preview", min_value=3.0, value=10.0)
    plot_h = st.number_input("Plot Height (m) - for preview", min_value=3.0, value=12.0)

    if st.button("Generate Optimized Layout"):
        layout, msg = generate_semantic_layout(total_area, num_rooms_input)
        rooms = layout.get("rooms", [])
        st.subheader("Optimized Room Area Distribution")
        for r in rooms:
            st.write(f"**{r['name'].title()}** → {r['area']} m²")

        st.markdown("### 2D Layout Preview")
        fig2d = plot_layout(layout, plot_w, plot_h, "Optimized 2D Layout")
        st.pyplot(fig2d)

        st.success("Optimized Layout Generated Successfully!")
