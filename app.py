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
import plotly.graph_objects as go

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256
CEILING_HEIGHT = 3.0  

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
        ax.text(x + w / 2, y + h / 2, f"{r['name']}\n{r['area']} m^2", ha='center', va='center', fontsize=8)
        x += w + pad
        row_h = max(row_h, h)
    ax.set_title(title)
    return fig

def rect_to_prism_vertices(x, y, w, h, z0=0.0, height=CEILING_HEIGHT):
    v0 = (x, y, z0)
    v1 = (x + w, y, z0)
    v2 = (x + w, y + h, z0)
    v3 = (x, y + h, z0)
    v4 = (x, y, z0 + height)
    v5 = (x + w, y, z0 + height)
    v6 = (x + w, y + h, z0 + height)
    v7 = (x, y + h, z0 + height)
    verts = [v0, v1, v2, v3, v4, v5, v6, v7]
    faces = [
        (0,1,2),(0,2,3),
        (4,6,5),(4,7,6),
        (0,4,5),(0,5,1),
        (1,5,6),(1,6,2),
        (2,6,7),(2,7,3),
        (3,7,4),(3,4,0)
    ]
    return verts, faces


def build_mesh_from_prisms(prism_list):
    verts_all = []
    i_faces, j_faces, k_faces = [], [], []
    vert_offset = 0
    for prism in prism_list:
        verts, faces = rect_to_prism_vertices(prism['x'], prism['y'], prism['w'], prism['h'], z0=0.0, height=prism.get('height', CEILING_HEIGHT))
        for v in verts:
            verts_all.append(v)
        for (a,b,c) in faces:
            i_faces.append(a + vert_offset)
            j_faces.append(b + vert_offset)
            k_faces.append(c + vert_offset)
        vert_offset += len(verts)
    if not verts_all:
        return None
    x_vals, y_vals, z_vals = zip(*verts_all)
    mesh = go.Mesh3d(
        x=list(x_vals), y=list(y_vals), z=list(z_vals),
        i=i_faces, j=j_faces, k=k_faces,
        opacity=0.9,
        flatshading=True,
        hoverinfo='skip'
    )
    return mesh


def layout_to_prisms(layout, plot_w, plot_h, ceiling_height=CEILING_HEIGHT):
    rooms = layout.get("rooms", [])
    total_area = sum(r["area"] for r in rooms) or 1.0
    scale = (plot_w * plot_h) / total_area
    pad = min(plot_w, plot_h) * 0.02
    x, y = pad, pad
    row_h = 0
    prisms = []
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
        prisms.append({'x': x, 'y': y, 'w': w, 'h': h, 'height': ceiling_height, 'name': r['name'], 'area': r['area']})
        x += w + pad
        row_h = max(row_h, h)
    return prisms


def plot_layout_3d(prisms_or_meshes, plot_w, plot_h, title="3D Layout"):
    if isinstance(prisms_or_meshes, list) and all(isinstance(p, dict) for p in prisms_or_meshes):
        mesh_data = [build_mesh_from_prisms(prisms_or_meshes)]
        labels = []
        for p in prisms_or_meshes:
            cx = p['x'] + p['w']/2
            cy = p['y'] + p['h']/2
            labels.append(go.Scatter3d(x=[cx], y=[cy], z=[p['height'] + 0.05], mode='text', text=[f"{p['name']} ({p['area']} m^2)"], textposition="middle center", hoverinfo='skip'))
    else:
        mesh_data = prisms_or_meshes
        labels = []
        
    if mesh_data is None or all(m is None for m in mesh_data):
        fig = go.Figure()
        fig.update_layout(title="No 3D geometry to show")
        return fig
    
    max_z = CEILING_HEIGHT
    
    plane = go.Scatter3d(
        x=[0, plot_w, plot_w, 0, 0],
        y=[0, 0, plot_h, plot_h, 0],
        z=[0, 0, 0, 0, 0],
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='skip',
        name='Boundary'
    )
    
    fig = go.Figure(data=mesh_data + [plane] + labels)
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (m)', backgroundcolor="rgb(240,240,240)", showgrid=False, zeroline=False),
            yaxis=dict(title='Y (m)', backgroundcolor="rgb(240,240,240)", showgrid=False, zeroline=False),
            zaxis=dict(title='Z (m)', backgroundcolor="rgb(250,250,250)", showgrid=False, zeroline=False, range=[0, max_z * 1.1]),
            aspectmode='manual',
            aspectratio=dict(x=plot_w/(plot_h if plot_h>0 else 1), y=1.0, z=0.5)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    return fig

def build_contour_mesh_3d(contours, m_per_pixel, ceiling_height=CEILING_HEIGHT, wall_thickness_m=0.15):
    mesh_elements = []
    
    WALL_COLOR = 'rgb(200, 200, 200)'
    FLOOR_COLOR = 'rgb(255, 255, 255)' 
    
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        pts_m = [(p[0][0] * m_per_pixel, p[0][1] * m_per_pixel) for p in approx]
        
        if len(pts_m) < 3:
            continue
            
        
        pts_2d = np.array(pts_m, dtype=np.float32)
        
        try:
            floor_x = [p[0] for p in pts_m]
            floor_y = [p[1] for p in pts_m]
            floor_z = [0.0] * len(pts_m)

            floor_i, floor_j, floor_k = [], [], []
            for i in range(1, len(pts_m) - 1):
                floor_i.append(0)
                floor_j.append(i)
                floor_k.append(i + 1)
            
            floor_mesh = go.Mesh3d(
                x=floor_x, y=floor_y, z=floor_z,
                i=floor_i, j=floor_j, k=floor_k,
                opacity=0.9,
                color=FLOOR_COLOR,
                name='Floor'
            )
            mesh_elements.append(floor_mesh)
            
        except Exception as e:
            st.warning(f"Failed to triangulate floor for a room. {e}")
            continue

        wall_verts = []
        wall_faces_i, wall_faces_j, wall_faces_k = [], [], []
        vert_offset = 0

        for i in range(len(pts_m)):
            p1_x, p1_y = pts_m[i]
            p2_x, p2_y = pts_m[(i + 1) % len(pts_m)]

            v0 = (p1_x, p1_y, 0.0)
            v1 = (p2_x, p2_y, 0.0)
            v2 = (p2_x, p2_y, ceiling_height)
            v3 = (p1_x, p1_y, ceiling_height)

            current_verts = [v0, v1, v2, v3]
            wall_verts.extend(current_verts)

            wall_faces_i.append(vert_offset + 0)
            wall_faces_j.append(vert_offset + 1)
            wall_faces_k.append(vert_offset + 2)
            
            wall_faces_i.append(vert_offset + 0)
            wall_faces_j.append(vert_offset + 2)
            wall_faces_k.append(vert_offset + 3)

            vert_offset += len(current_verts)

        if wall_verts:
            wall_x, wall_y, wall_z = zip(*wall_verts)
            wall_mesh = go.Mesh3d(
                x=list(wall_x), y=list(wall_y), z=list(wall_z),
                i=wall_faces_i, j=wall_faces_j, k=wall_faces_k,
                opacity=0.9,
                color=WALL_COLOR,
                name='Wall'
            )
            mesh_elements.append(wall_mesh)

    return mesh_elements


def segmentation_to_contour_meshes(seg_img_pil, img_display_w, img_display_h, ceiling_height=CEILING_HEIGHT, min_area_px=200):
    seg_np = np.array(seg_img_pil.convert("RGB"))
    gray = cv2.cvtColor(seg_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_px, w_px = gray.shape
    
    if not contours:
        return []

    m_per_pixel_w = img_display_w / w_px
    m_per_pixel_h = img_display_h / h_px
    m_per_pixel = (m_per_pixel_w + m_per_pixel_h) / 2 
    
    all_meshes = []
    
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    if cv2.contourArea(largest_contour) < min_area_px:
        return []
        
    all_meshes.extend(build_contour_mesh_3d([largest_contour], m_per_pixel, ceiling_height))
        
    return all_meshes

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
    st.markdown(f"**Calculated Total Area:** {area_m2:.2f} m^2 (≈ {area_sqft:.0f} sq ft)**")
    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)
    # Ceiling Height input removed, fixed at CEILING_HEIGHT = 3.0

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

                # 3D visualization is now automatic
                img_display_w = house_length
                img_display_h = house_width
                
                mesh_elements = segmentation_to_contour_meshes(
                    seg_img, img_display_w, img_display_h, ceiling_height=CEILING_HEIGHT
                )
                
                if mesh_elements:
                    fig3d = plot_layout_3d(mesh_elements, img_display_w, img_display_h, title=f"Plan {i+1} 3D (Contour-Based)")
                    col.markdown("---")
                    col.markdown(f"**3D View of Plan {i+1}**")
                    col.plotly_chart(fig3d, use_container_width=True)
                else:
                    col.info("No main building contour found to extrude for 3D.")

elif mode == "Real-Time Sensor Dashboard":
    st.header("Cloud Sensor Dashboard")
    st.markdown("Fetch ultrasonic readings one at a time and confirm whether it’s **Length** or **Breadth**.")

    for key in ["length", "breadth", "last_distance", "pir", "ir", "last_set"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.divider()

    if st.session_state.length is None and st.session_state.breadth is None and st.session_state.last_distance is None:
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
                st.error(f"Error: {e}")

    elif st.session_state.last_distance is not None:
        st.subheader("Last Measured Distance")
        st.write(f"{st.session_state.last_distance} cm")

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

        elif st.session_state.length is not None and st.session_state.breadth is None:
            if st.button("Set as Breadth", use_container_width=True):
                st.session_state.breadth = st.session_state.last_distance
                st.session_state.last_set = "breadth"
                st.session_state.last_distance = None
                st.rerun()

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
                st.error(f"Error: {e}")

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

    elif st.session_state.length and st.session_state.breadth:
        st.success("Both Length and Breadth captured successfully.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Latest", use_container_width=True):
                if st.session_state.last_set == "length":
                    st.session_state.length = None
                else:
                    st.session_state.breadth = None
                st.info("Latest entry cleared.")
                st.rerun()
        with col2:
            if st.button("Reset All", use_container_width=True):
                for k in ["length", "breadth", "last_distance", "pir", "ir", "last_set"]:
                    st.session_state[k] = None
                st.info("All cleared.")
                st.rerun()
        st.divider()
    st.divider()
    st.subheader("Current Measurements")
    st.write(f"Length: {st.session_state.length if st.session_state.length else '—'} cm")
    st.write(f"Breadth: {st.session_state.breadth if st.session_state.breadth else '—'} cm")

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

        length_m = st.session_state.length * 0.01
        breadth_m = st.session_state.breadth * 0.01
        area_m2 = length_m * breadth_m
        area_sqft = area_m2 * 10.7639

        st.write(f"**Final Dimensions:** {length_m:.2f} m × {breadth_m:.2f} m")
        st.write(f"**Calculated Total Area:** {area_m2:.2f} m^2 (≈ {area_sqft:.0f} sq ft)")

        bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1)
        denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)
        # Ceiling Height input removed, fixed at CEILING_HEIGHT = 3.0

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

                    # 3D visualization is now automatic
                    img_display_w = length_m
                    img_display_h = breadth_m
                    
                    mesh_elements = segmentation_to_contour_meshes(
                        seg_img, img_display_w, img_display_h, ceiling_height=CEILING_HEIGHT
                    )
                    
                    if mesh_elements:
                        fig3d = plot_layout_3d(mesh_elements, img_display_w, img_display_h, title=f"Plan {i+1} 3D (Contour-Based)")
                        col.markdown("---")
                        col.markdown(f"**3D View of Plan {i+1}**")
                        col.plotly_chart(fig3d, use_container_width=True)
                    else:
                        col.info("No main building contour found to extrude for 3D.")

elif mode == "Optimized Layout":

    st.header("Optimized Layout Generator")

    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        num_rooms_input = st.number_input("Enter Total Number of Rooms", min_value=1, value=3, step=1)

    st.markdown("Select Plot Shape:")
    plot_shape = st.radio("Plot Shape", ["Rectangle",'square'], horizontal=True)

    plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=10.0)
    plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=12.0)

    if st.button("Generate Optimized Layout", type="primary", use_container_width=True):

        # Create semantic layout (room area distribution)
        layout, msg = generate_semantic_layout(total_area, num_rooms_input,
                                               property_type=None,
                                               plot_shape=plot_shape,
                                               plot_w=plot_w,
                                               plot_h=plot_h)
        rooms = layout.get("rooms", [])
        st.subheader("Optimized Room Area Distribution")
        for r in rooms:
            st.write(f"**{r['name'].title()}** → {r['area']} m²")

        # 2D PLOT
        st.markdown("### 2D Layout Preview")
        fig2d = plot_layout(layout, plot_w, plot_h, "Optimized 2D Layout")
        st.pyplot(fig2d, use_container_width=True)

        # Convert 2D → 3D (prisms)
        prisms = layout_to_prisms(layout, plot_w, plot_h, CEILING_HEIGHT)

        if not prisms:
            st.error("Failed to generate 3D geometry.")
        else:
            fig3d = plot_layout_3d(prisms, plot_w, plot_h, "3D Optimized Layout")
            st.markdown("### 3D Layout Visualization")
            st.plotly_chart(fig3d, use_container_width=True)

        st.success("Optimized Layout Generated Successfully!")



st.sidebar.header("Arch-Ai-Bot")

api_key = st.secrets.get("ARCH_AI_TEX_CHATBOT1")
if not api_key:
    st.sidebar.error("ARCH_AI_TEX_CHATBOT1 not found in Streamlit secrets. Add it in app settings.")
else:
    def ask_groq(messages):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": 0.2,
        }
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling LLM API: {e}"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": (
                "You are an expert architect and interior designer. "
                "Answer clearly and concisely. Provide checklists and step-by-step guidance when helpful."
            )}
        ]

    for msg in st.session_state.chat_history[1:]:
        with st.sidebar.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.sidebar.chat_input("Ask anything about Architecture or Interior Design…")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.sidebar.chat_message("user").write(user_input)

        answer = ask_groq(st.session_state.chat_history)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.sidebar.chat_message("assistant").write(answer)

#https://esp32-fastapi-server-uh47.onrender.com/
