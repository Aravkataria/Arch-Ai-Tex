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
import requests
import time
import subprocess
from streamlit_3d import streamlit_3d
from st3d import st3d
import os

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

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
        st.error("GAN generator weights not found or failed to load. The output will likely be noise.")
    generator.eval()
    return rf_model, generator, None

RF_MODEL, GAN_MODEL, SEG_MODEL = load_models()

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
    seed_base = int(area * 10 + bedrooms * 1234)
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
        (188, 189, 34),
        (140, 86, 75),
    ]
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        color_index = (i - 1) % len(room_colors)
        color = room_colors[color_index]
        seg_rgb[labels == i] = color
    seg_pil = Image.fromarray(seg_rgb).resize(image.size)
    return seg_pil

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
mode = st.radio(
    "Select Mode:",
    ["GAN Generator", "Optimized Layout", "Real-Time Sensor Dashboard"],
    horizontal=True
)


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

                # --- Display Images ---
                col.image(img, caption=f"Plan {i+1}", use_column_width=True)
                col.image(seg_img, caption=f"Segmented Plan {i+1}", use_column_width=True)
                
                # Convert to bytes for download
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                
                # ------------------------------
                # NORMAL DOWNLOAD BUTTON (2D)
                # ------------------------------
                col.download_button(
                    label=f"Download Plan {i+1}",
                    data=img_bytes,
                    file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png",
                    mime="image/png",
                    key=f"download2d_gan_{i}"
                )
                
                # ------------------------------
                # GENERATE 3D MODEL BUTTON
                # ------------------------------
                # Using st.session_state to track 3D button click reliably
                if f"gen3d_gan_{i}" not in st.session_state:
                    st.session_state[f"gen3d_gan_{i}"] = False
                
                gen3d = col.button(f"Generate 3D Model", key=f"gen3d_gan_{i}_btn", use_container_width=True)

                if gen3d:
                    # Set state to show generation output
                    st.session_state[f"gen3d_gan_{i}"] = True
                    st.rerun()

                if st.session_state[f"gen3d_gan_{i}"] == True:
                    st.info(f"Generating 3D model for Plan {i+1}...")
                    
                    # --- 3D Generation Logic Placeholder ---
                    try:
                        # 1. Save temp floorplan
                        temp_png = f"temp_plan_{i}_{int(time.time())}.png"
                        img.save(temp_png)
                        output_glb = f"plan_{i+1}_3d_{int(time.time())}.glb"

                        # 2. Run Blender (Placeholder - Requires Blender installed and blender_make_3d.py)
                        # The following subprocess call is a placeholder and may not work in all environments.
                        try:
                            # Added check=True and timeout for robustness
                            result = subprocess.run([
                                "blender",
                                "--background",
                                "--python", "blender_make_3d.py",
                                "--",
                                temp_png,
                                output_glb
                            ], capture_output=True, text=True, check=True, timeout=60) 
                            
                            # Mocking file existence since we cannot guarantee Blender run
                            if not os.path.exists(output_glb):
                                glb_data = b"MOCK_GLB_DATA" 
                                st.warning("Blender or its dependencies were not found/accessible. Using mock data for download and skipping preview.")
                            else:
                                with open(output_glb, "rb") as f:
                                    glb_data = f.read()
                                st.success(f"3D Model Generated Successfully for Plan {i+1}!")
                                
                                # ---- SHOW 3D PREVIEW IN STREAMLIT ----
                                st.subheader(f"3D Preview for Plan {i+1}")
                                stl_plot(data=glb_data, width=400, height=400) # Mocked or real preview

                                # Clean up temporary files
                                os.remove(temp_png)
                                os.remove(output_glb)

                        except subprocess.CalledProcessError as e:
                            glb_data = b"MOCK_GLB_DATA" 
                            st.error(f"Blender process failed. Ensure Blender is callable and 'blender_make_3d.py' is correct. Error: {e.stderr}")
                        except FileNotFoundError:
                            glb_data = b"MOCK_GLB_DATA" 
                            st.error("Blender or 'blender_make_3d.py' not found. Please ensure external dependencies are set up.")
                        except ImportError:
                            glb_data = b"MOCK_GLB_DATA"
                            st.error("The 'streamlit-3d' library is required to show the preview. Please install it with 'pip install streamlit-3d'.")
                        except Exception as e:
                            glb_data = b"MOCK_GLB_DATA" 
                            st.error(f"An unexpected error occurred during 3D generation: {e}")
                            
                        # ---- DOWNLOAD 3D MODEL ----
                        col.download_button(
                            label=f"Download 3D Model (GLB)",
                            data=glb_data,
                            file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.glb",
                            mime="model/gltf-binary",
                            key=f"download3d_gan_{i}"
                        )
                        # Option to hide the 3D output
                        if col.button("Hide 3D Output", key=f"hide3d_gan_{i}", use_container_width=True):
                            st.session_state[f"gen3d_gan_{i}"] = False
                            st.rerun()

                    except Exception as e:
                        st.error(f"Critical error in 3D logic setup: {e}")

# ----------------------------------------------------------------------------------------------------
# REAL-TIME SENSOR DASHBOARD SECTION
# ----------------------------------------------------------------------------------------------------

elif mode == "Real-Time Sensor Dashboard":
    st.header("Cloud Sensor Dashboard")
    st.markdown("Fetch ultrasonic readings one at a time and confirm whether it’s **Length** or **Breadth**.")

    # Initialize session states
    for key in ["length", "breadth", "last_distance", "pir", "ir", "last_set"]:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Initialize 3D session states for sensor mode
    for i in range(3):
        if f"gen3d_sensor_{i}" not in st.session_state:
            st.session_state[f"gen3d_sensor_{i}"] = False

    st.divider()

    # --- CASE 1: Nothing yet — only show Get Sensor Data ---
    if st.session_state.length is None and st.session_state.breadth is None and st.session_state.last_distance is None:
        if st.button("Get Sensor Data", use_container_width=True):
            try:
                # Replace with your actual FastAPI/API endpoint
                r = requests.get("https://esp32-fastapi-server-uh47.onrender.com/data", timeout=5)
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    st.session_state.pir = d.get("pir")
                    st.session_state.ir = d.get("ir")
                    st.session_state.last_distance = d.get("ultrasonic")
                    if st.session_state.last_distance is None:
                        st.warning("No ultrasonic data found.")
                else:
                    st.error(f"Server responded with {r.status_code}")
            except Exception as e:
                st.error(f"Error connecting to sensor API: {e}")
            st.rerun()

    # --- CASE 2: Have a new distance waiting to assign ---
    elif st.session_state.last_distance is not None:
        st.subheader("Last Measured Distance")
        st.write(f"{st.session_state.last_distance} cm")

        # --- Subcase 2A: No dimensions set yet (Both Length and Breadth options available) ---
        if st.session_state.length is None and st.session_state.breadth is None:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Set as Length", use_container_width=True):
                    st.session_state.length = st.session_state.last_distance
                    st.session_state.last_set = "length"
                    st.session_state.last_distance = None
                    st.rerun()
            with col2:
                if st.button("Set as Breadth", use_container_width=True):
                    st.session_state.breadth = st.session_state.last_distance
                    st.session_state.last_set = "breadth"
                    st.session_state.last_distance = None
                    st.rerun()

        # --- Subcase 2B: Length is set, waiting for Breadth (Show Breadth button full width) ---
        elif st.session_state.length is not None and st.session_state.breadth is None:
            if st.button("Set as Breadth", use_container_width=True):
                st.session_state.breadth = st.session_state.last_distance
                st.session_state.last_set = "breadth"
                st.session_state.last_distance = None
                st.rerun()

        # --- Subcase 2C: Breadth is set, waiting for Length (Show Length button full width) ---
        elif st.session_state.breadth is not None and st.session_state.length is None:
            if st.button("Set as Length", use_container_width=True):
                st.session_state.length = st.session_state.last_distance
                st.session_state.last_set = "length"
                st.session_state.last_distance = None
                st.rerun()

        if st.button("Reset Last Value", use_container_width=True):
            st.session_state.last_distance = None
            st.info("Last value cleared.")
            st.rerun()

    # --- CASE 3: One dimension set, waiting for the other ---
    elif (st.session_state.length is not None) ^ (st.session_state.breadth is not None):
        st.info("Now get the other dimension.")
        if st.button("Get Sensor Data", use_container_width=True):
            try:
                r = requests.get("https://esp32-fastapi-server-uh47.onrender.com/data", timeout=5)
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    st.session_state.pir = d.get("pir")
                    st.session_state.ir = d.get("ir")
                    st.session_state.last_distance = d.get("ultrasonic")
                    if st.session_state.last_distance is None:
                        st.warning("No ultrasonic data found.")
                else:
                    st.error(f"Server responded with {r.status_code}")
            except Exception as e:
                st.error(f"Error connecting to sensor API: {e}")
            st.rerun()

        # show reset for whichever one is set
        if st.session_state.length is not None:
            if st.button("Reset Entered Length", use_container_width=True):
                st.session_state.length = None
                st.session_state.last_set = None
                st.info("Length cleared.")
                st.rerun()
        if st.session_state.breadth is not None:
            if st.button("Reset Entered Breadth", use_container_width=True):
                st.session_state.breadth = None
                st.session_state.last_set = None
                st.info("Breadth cleared.")
                st.rerun()

    # --- CASE 4: Both captured ---
    elif st.session_state.length and st.session_state.breadth:
        st.success("Both Length and Breadth captured successfully.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Latest", use_container_width=True):
                if st.session_state.last_set == "length":
                    st.session_state.length = None
                else:
                    st.session_state.breadth = None
                st.session_state.last_set = None
                st.info("Latest entry cleared.")
                st.rerun()
        with col2:
            if st.button("Reset All", use_container_width=True):
                for k in ["length", "breadth", "last_distance", "pir", "ir", "last_set"]:
                    st.session_state[k] = None
                # Also reset 3D states
                for i in range(3):
                    st.session_state[f"gen3d_sensor_{i}"] = False
                st.info("All cleared.")
                st.rerun()
        st.divider()
        
    st.divider()
    st.subheader("Current Measurements")
    # Convert cm to m for display
    length_m_disp = f"{st.session_state.length * 0.01:.2f}" if st.session_state.length else '—'
    breadth_m_disp = f"{st.session_state.breadth * 0.01:.2f}" if st.session_state.breadth else '—'

    st.write(f"Length: {length_m_disp} m ({st.session_state.length} cm)")
    st.write(f"Breadth: {breadth_m_disp} m ({st.session_state.breadth} cm)")

    if st.session_state.pir is not None or st.session_state.ir is not None:
        st.divider()
        st.subheader("Motion & Obstacle Sensors")
        pir_status = "Motion Detected" if st.session_state.pir else "No Motion"
        ir_status = "Obstacle Detected" if not st.session_state.ir else "Clear Path"
        st.write(f"PIR: {pir_status}")
        st.write(f"IR: {ir_status}")

    if st.session_state.length and st.session_state.breadth:
        st.divider()
        st.subheader("Generate Floorplan from Captured Dimensions")

        # Convert from cm → m
        length_m = st.session_state.length * 0.01
        breadth_m = st.session_state.breadth * 0.01
        area_m2 = length_m * breadth_m
        area_sqft = area_m2 * 10.7639

        st.write(f"**Final Dimensions:** {length_m:.2f} m × {breadth_m:.2f} m")
        st.write(f"**Calculated Total Area:** {area_m2:.2f} m² (≈ {area_sqft:.0f} sq ft)")

        bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1, key="sensor_beds")
        denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False, key="sensor_denoise")

        if st.button("Generate Floorplans", type="primary", use_container_width=True, key="sensor_generate_btn"):
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

                    # --- Display Images ---
                    col.image(img, caption=f"Plan {i+1}", use_column_width=True)
                    col.image(seg_img, caption=f"Segmented Plan {i+1}", use_column_width=True)
                    
                    # Convert to bytes for download
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()

                    # ------------------------------
                    # NORMAL DOWNLOAD BUTTON (2D)
                    # ------------------------------
                    col.download_button(
                        label=f"Download Plan {i+1}",
                        data=img_bytes,
                        file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png",
                        mime="image/png",
                        key=f"download2d_sensor_{i}"
                    )

                    # ------------------------------
                    # GENERATE 3D MODEL BUTTON
                    # ------------------------------
                    if f"gen3d_sensor_{i}" not in st.session_state:
                        st.session_state[f"gen3d_sensor_{i}"] = False
                    
                    gen3d_sensor = col.button(f"Generate 3D Model", key=f"gen3d_sensor_{i}_btn", use_container_width=True)

                    if gen3d_sensor:
                        # Set state to show generation output
                        st.session_state[f"gen3d_sensor_{i}"] = True
                        st.rerun()

                    if st.session_state[f"gen3d_sensor_{i}"] == True:
                        st.info(f"Generating 3D model for Plan {i+1}...")
                        
                        # --- 3D Generation Logic Placeholder ---
                        try:
                            # 1. Save temp floorplan
                            temp_png = f"temp_plan_{i}_{int(time.time())}.png"
                            img.save(temp_png)
                            output_glb = f"plan_{i+1}_3d_{int(time.time())}.glb"

                            # 2. Run Blender (Placeholder - Requires Blender installed and blender_make_3d.py)
                            try:
                                # Added check=True and timeout for robustness
                                result = subprocess.run([
                                    "blender",
                                    "--background",
                                    "--python", "blender_make_3d.py",
                                    "--",
                                    temp_png,
                                    output_glb
                                ], capture_output=True, text=True, check=True, timeout=60)
                                
                                if not os.path.exists(output_glb):
                                    glb_data = b"MOCK_GLB_DATA" 
                                    st.warning("Blender or its dependencies were not found/accessible. Using mock data for download and skipping preview.")
                                else:
                                    with open(output_glb, "rb") as f:
                                        glb_data = f.read()
                                    st.success(f"3D Model Generated Successfully for Plan {i+1}!")
                                    
                                    # ---- SHOW 3D PREVIEW IN STREAMLIT ----
                                    st.subheader(f"3D Preview for Plan {i+1}")
                                    stl_plot(data=glb_data, width=400, height=400)

                                    # Clean up temporary files
                                    os.remove(temp_png)
                                    os.remove(output_glb)

                            except subprocess.CalledProcessError as e:
                                glb_data = b"MOCK_GLB_DATA" 
                                st.error(f"Blender process failed. Ensure Blender is callable and 'blender_make_3d.py' is correct. Error: {e.stderr}")
                            except FileNotFoundError:
                                glb_data = b"MOCK_GLB_DATA" 
                                st.error("Blender or 'blender_make_3d.py' not found. Please ensure external dependencies are set up.")
                            except ImportError:
                                glb_data = b"MOCK_GLB_DATA"
                                st.error("The 'streamlit-3d' library is required to show the preview. Please install it with 'pip install streamlit-3d'.")
                            except Exception as e:
                                glb_data = b"MOCK_GLB_DATA" 
                                st.error(f"An unexpected error occurred during 3D generation: {e}")
                                
                            # ---- DOWNLOAD 3D MODEL ----
                            col.download_button(
                                label=f"Download 3D Model (GLB)",
                                data=glb_data,
                                file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.glb",
                                mime="model/gltf-binary",
                                key=f"download3d_sensor_{i}"
                            )
                            # Option to hide the 3D output
                            if col.button("Hide 3D Output", key=f"hide3d_sensor_{i}", use_container_width=True):
                                st.session_state[f"gen3d_sensor_{i}"] = False
                                st.rerun()

                        except Exception as e:
                            st.error(f"Critical error in 3D logic setup: {e}")

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
#https://esp32-fastapi-server-uh47.onrender.com/
