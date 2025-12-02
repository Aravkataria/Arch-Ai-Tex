import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2
import math
import warnings
from PIL import Image

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex (Clean)", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256
CEILING_HEIGHT = 3.0  # retained as a constant but not used for 3D


class DCGAN_Generator(nn.Module):
    @staticmethod
    def block(in_f, out_f):
        return nn.Sequential(
            nn.BatchNorm2d(in_f),
            nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
            nn.ReLU(True)
        )

    def __init__(self, latent_dim=LATENT_DIM, channels=CHANNELS):
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


@st.cache_resource
def load_models():
    rf_model = None
    generator = DCGAN_Generator().to(DEVICE)
    # try to load RF predictor
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
        st.warning("GAN generator weights not found. Generated outputs will be random noise until weights are provided.")
    generator.eval()
    return rf_model, generator


RF_MODEL, GAN_MODEL = load_models()


def predict_dwelling_type(area, bedrooms, rf_model):
    if rf_model is None:
        return "Unknown Type (RF model missing)"
    try:
        features = np.array([[float(area), int(bedrooms)]])
        return rf_model.predict(features)[0]
    except Exception:
        return "Prediction Failed"


def generate_final_plans(generator, area, bedrooms, count=3, denoise=False, rf_model=None):
    dwelling_type = predict_dwelling_type(area, bedrooms, rf_model)
    images = []
    if area < 100:
        area = 100
    pixel_area = area / (IMG_SIZE * IMG_SIZE)
    for i in range(count):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)
        with torch.no_grad():
            img_tensor = generator(z)
            img_np = img_tensor.squeeze().cpu().numpy()
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


def apply_segmentation(image, num_rooms):
    # simple connected components segmentation (2D only)
    if image.mode != "L":
        img_cv = np.array(image.convert("L"))
    else:
        img_cv = np.array(image)
    _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    seg_rgb = np.zeros((*img_cv.shape, 3), dtype=np.uint8)
    room_colors = [
        (255, 199, 107),
        (130, 202, 157),
        (174, 199, 232),
        (255, 152, 150),
        (197, 176, 213),
        (255, 237, 111),
    ]
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        color_index = (i - 1) % len(room_colors)
        color = room_colors[color_index]
        seg_rgb[labels == i] = color
    seg_pil = Image.fromarray(seg_rgb).resize(image.size)
    return seg_pil


def generate_semantic_layout(total_area, num_rooms_input):
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
        ax.text(x + w / 2, y + h / 2, f"{r['name']}\n{r['area']} m^2", ha='center', va='center', fontsize=8)
        x += w + pad
        row_h = max(row_h, h)
    ax.set_title(title)
    return fig


# ---- Streamlit UI ----
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
    st.title("Arch-Ai-Tex (Clean)")
    st.markdown("AI Floor Plan Generator — 2D only")
with col2:
    try:
        st.image("QR.png", width=110)
    except Exception:
        pass

st.markdown("---")

mode = st.radio(
    "Select Mode:",
    ["GAN Generator", "Optimized Layout"],
    horizontal=True
)

if mode == "GAN Generator":
    col_len, col_wid = st.columns(2)
    with col_len:
        house_length = st.number_input("Enter House Length (m)", min_value=1.0, value=10.0, step=1.0)
    with col_wid:
        house_width = st.number_input("Enter House Width (m)", min_value=1.0, value=8.0, step=1.0)
    area_m2 = max(100.0, house_length * house_width)
    area_sqft = area_m2 * 10.7639
    st.markdown(f"**Calculated Total Area:** {area_m2:.2f} m^2 (≈ {area_sqft:.0f} sq ft)")
    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=2, step=1)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)

    if st.button("Generate Floorplans", type="primary", use_container_width=True):
        dwelling_type, floor_plan_images, pixel_area = generate_final_plans(
            GAN_MODEL, area_m2, bedrooms, count=3, denoise=denoise_option, rf_model=RF_MODEL
        )
        st.subheader(f"Predicted Dwelling Type: {dwelling_type}")
        st.markdown(f"**Area to Pixel Ratio:** 1 pixel ≈ {pixel_area:.4f} m^2")
        st.markdown("Generated Floorplans:")
        cols = st.columns(3)
        for i, col in enumerate(cols):
            if i < len(floor_plan_images):
                img = floor_plan_images[i]
                seg_img = apply_segmentation(img, bedrooms)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                col.image(img, caption=f"Plan {i+1}", use_column_width=True)
                col.image(seg_img, caption=f"Segmented Plan {i+1}", use_column_width=True)
                col.download_button(
                    label=f"Download Plan {i+1}",
                    data=buf.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png",
                    mime="image/png",
                )

elif mode == "Optimized Layout":
    st.header("Optimized Layout Generator")
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        num_rooms_input = st.number_input("Enter Total Number of Rooms", min_value=1, value=3, step=1)

    plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=10.0)
    plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=12.0)

    if st.button("Generate Optimized Layout", type="primary", use_container_width=True):
        layout, msg = generate_semantic_layout(total_area, num_rooms_input)
        rooms = layout.get("rooms", [])
        st.subheader("Optimized Room Area Distribution")
        for r in rooms:
            st.write(f"**{r['name'].title()}** → {r['area']} m²")

        st.markdown("### 2D Layout Preview")
        fig2d = plot_layout(layout, plot_w, plot_h, "Optimized 2D Layout")
        st.pyplot(fig2d, use_container_width=True)

        st.success("Optimized Layout Generated Successfully!")


# End of cleaned app.py

# --- Segmentation Model Integration ---
import torchvision.transforms as T
import torchvision.models.segmentation as models

@st.cache_resource
def load_segmentation_model():
    model = models.deeplabv3_resnet50(weights="DEFAULT").to(DEVICE)
    model.eval()
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform

SEG_MODEL, SEG_TRANSFORM = load_segmentation_model()

st.markdown("---")
st.subheader("Segmentation Model (Optional)")
upload_seg = st.file_uploader("Upload an input image for segmentation", type=["png","jpg","jpeg"])

if upload_seg:
    img = Image.open(upload_seg).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    inp = SEG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = SEG_MODEL(inp)["out"]
    pred = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    seg_color = cv2.applyColorMap((pred * 10).astype(np.uint8), cv2.COLORMAP_JET)
    st.image(seg_color, caption="Segmentation Output", use_column_width=True)

