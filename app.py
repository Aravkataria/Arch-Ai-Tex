import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
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
            nn.BatchNorm2d(in_f),
            nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.gen = nn.Sequential(
            DCGAN_Generator.block(512, 256),
            DCGAN_Generator.block(256, 128),
            DCGAN_Generator.block(128, 64),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.gen(out)


# ----------------------------
# Model Loading
# ----------------------------
@st.cache_resource
def load_models():
    rf_model = None
    generator = DCGAN_Generator().to(DEVICE)
    try:
        rf_model = joblib.load("room_predictor.joblib")
    except Exception:
        rf_model = None

    loaded = False
    for fname in ("generator_epoch100.pth", "generator_epoch_100.pth", "generator.pth"):
        try:
            state_dict = torch.load(fname, map_location=DEVICE)
            generator.load_state_dict(state_dict, strict=False)
            loaded = True
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Error loading generator model {fname}: {e}")
            continue

    if not loaded:
        st.error("GAN generator weights not found or failed to load. The output may be noisy.")

    generator.eval()
    return rf_model, generator, None


RF_MODEL, GAN_MODEL, SEG_MODEL = load_models()


# ----------------------------
# Dwelling Type Prediction
# ----------------------------
def predict_dwelling_type(area, bedrooms, rf_model):
    if rf_model is None:
        return "Unknown Type"
    try:
        features = np.array([[float(area), int(bedrooms)]])
        return rf_model.predict(features)[0]
    except Exception:
        return "Prediction Failed"


# ----------------------------
# Enhanced GAN-based Plan Generation
# ----------------------------
def apply_segmentation(image, num_rooms):
    """
    Draws a clean, colored, grid-based floorplan layout with connected rooms,
    automatically adding kitchen, washroom, stairs, porch, and garden.
    Blends the GAN output softly underneath.
    """
    img_cv = np.array(image.convert("RGB"))
    h, w, _ = img_cv.shape

    # Define total number of spaces
    extra_spaces = ["Kitchen", "Washroom", "Staircase", "Porch", "Garden"]
    total_rooms = num_rooms + len(extra_spaces)

    layout = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Grid config
    cols = math.ceil(math.sqrt(total_rooms))
    rows = math.ceil(total_rooms / cols)
    pad = 5  # wall thickness

    cell_w = (w - (cols + 1) * pad) // cols
    cell_h = (h - (rows + 1) * pad) // rows

    # Pleasant pastel colors
    colors = [
        (242, 214, 189), (204, 235, 197), (222, 203, 228),
        (255, 255, 204), (179, 222, 233), (252, 205, 229),
        (253, 180, 98), (201, 222, 136), (255, 242, 174)
    ]

    room_labels = [f"Room {i+1}" for i in range(num_rooms)] + extra_spaces

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= total_rooms:
                break
            x1 = pad + c * (cell_w + pad)
            y1 = pad + r * (cell_h + pad)
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            color = colors[idx % len(colors)]
            cv2.rectangle(layout, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(layout, (x1, y1), (x2, y2), (0, 0, 0), 2)

            label = room_labels[idx]
            cv2.putText(
                layout,
                label,
                (x1 + 8, y1 + cell_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 60, 60),
                2,
                cv2.LINE_AA
            )
            idx += 1

    # Optional GAN texture blend
    try:
        gan_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        gan_gray = cv2.merge([gan_gray, gan_gray, gan_gray])
        layout = cv2.addWeighted(layout, 0.85, gan_gray, 0.15, 0)
    except Exception:
        pass

    return Image.fromarray(layout)


def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    dwelling_type = predict_dwelling_type(area, bedrooms, rf_model)
    images = []

    if area < 100:
        area = 100

    pixel_area = area / (IMG_SIZE * IMG_SIZE)
    seed_base = int(area * 10 + bedrooms * 1234)

    for i in range(count):
        torch.manual_seed(seed_base + i)
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = generator(z)
            img_np = img_tensor.squeeze().cpu().numpy()
            img_np = np.clip(((img_np + 1) * 127.5), 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np, 'L')

            if denoise:
                img_np = cv2.fastNlMeansDenoising(img_np, None, h=10)
                img = Image.fromarray(img_np, 'L')

            img_rgb = img.convert("RGB")
            images.append(img_rgb)

    return dwelling_type, images, pixel_area


# ----------------------------
# Optimized Layout (Unchanged)
# ----------------------------
def generate_semantic_layout(total_area, num_rooms_input, property_type, plot_shape, plot_w, plot_h):
    total_area = float(total_area)
    num_rooms_input = max(0, int(num_rooms_input))
    fixed_ratios = {"living+dining": 0.28, "kitchen": 0.08, "bathroom": 0.06}
    fixed_total = sum(fixed_ratios.values())
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
        rooms.append({"name": "utility/other", "area": round(total_area * remaining_ratio, 2)})

    current_sum = round(sum(r["area"] for r in rooms), 2)
    diff = round(total_area - current_sum, 2)
    if abs(diff) >= 0.01 and rooms:
        rooms[0]["area"] = round(rooms[0]["area"] + diff, 2)

    return {"rooms": rooms, "num_bedrooms": num_bedrooms}, ""


def plot_layout(layout, plot_w, plot_h, title="Layout"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, plot_h)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), plot_w, plot_h, fill=False, edgecolor='black', linewidth=1.2))

    rooms = layout.get("rooms", [])
    total_area = sum(r["area"] for r in rooms)
    scale = (plot_w * plot_h) / max(total_area, 1.0)

    pad = min(plot_w, plot_h) * 0.02
    x, y = pad, pad
    row_h = 0
    colors = ["#f4cccc", "#d9ead3", "#cfe2f3", "#fff2cc", "#d9d2e9", "#c2f0c2"]

    for i, r in enumerate(rooms):
        desired_area = max(0.1, r["area"])
        rect_area = desired_area * scale
        w = math.sqrt(rect_area) * 1.3
        h = rect_area / w

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
    st.image("QR.png", width=110)
    st.markdown("<p style='font-size:13px; color:gray; text-align:right;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)

st.markdown("---")
mode = st.radio("Select Model:", ["GAN Generator", "Optimized Layout"], horizontal=True)


# ----------------------------
# Mode 1: GAN Generator
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
                seg_img = apply_segmentation(img, bedrooms)
                buf = io.BytesIO()
                img.save(buf, format="PNG")

                col.image(img, caption=f"GAN Plan {i+1}", use_column_width=True)
                col.image(seg_img, caption=f"Structured Plan {i+1}", use_column_width=True)
                col.download_button(
                    label=f"Download Plan {i+1}",
                    data=buf.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png",
                    mime="image/png",
                )


# ----------------------------
# Mode 2: Optimized Layout (Unchanged)
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
            layout, _ = generate_semantic_layout(total_area, num_rooms_input, property_type, plot_shape, plot_w, plot_h)
            dwelling_type = predict_dwelling_type(total_area, layout["num_bedrooms"], RF_MODEL)
            st.success(f"Predicted Dwelling Type: **{dwelling_type}**")
            fig = plot_layout(layout, plot_w, plot_h, f"{property_type} Layout")
            st.pyplot(fig)
