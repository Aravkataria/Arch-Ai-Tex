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
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    O3D_AVAILABLE = False
    class DummyO3D:
        def __init__(self): pass
        class geometry:
            class TriangleMesh:
                def compute_vertex_normals(self): pass
                def __init__(self): pass
        class utility:
            def Vector3dVector(self, data): return data
            def Vector3iVector(self, data): return data
        class io:
            def write_triangle_mesh(self, *args): pass
    o3d = DummyO3D()

st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

def floorplan_to_3d_mesh(segmented_img, height=3.0):
    if not O3D_AVAILABLE:
        return None
        
    img_gray = np.array(segmented_img.convert("L"))
    h, w = img_gray.shape
    vertices = []
    faces = []

    for y in range(h):
        for x in range(w):
            if img_gray[y, x] < 128:
                base_idx = len(vertices)
                vertices.extend([
                    [x, y, 0],
                    [x+1, y, 0],
                    [x+1, y+1, 0],
                    [x, y+1, 0],
                    [x, y, height],
                    [x+1, y, height],
                    [x+1, y+1, height],
                    [x, y+1, height],
                ])
                cube_faces = [
                    [0,1,2], [0,2,3],
                    [4,5,6], [4,6,7],
                    [0,1,5], [0,5,4],
                    [1,2,6], [1,6,5],
                    [2,3,7], [2,7,6],
                    [3,0,4], [3,4,7],
                ]
                faces.extend([[idx+base_idx for idx in face] for face in cube_faces])
    
    if not vertices:
        return None
        
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    mesh.compute_vertex_normals()
    return mesh

def plotly_preview(mesh):
    if mesh is None:
        return None
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    x, y, z = verts[:,0], verts[:,1], verts[:,2]
    i, j, k = faces[:,0], faces[:,1], faces[:,2]
    
    fig = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='lightcoral',
        opacity=0.8,
        lighting=dict(ambient=0.5, diffuse=0.5, specular=0.2, roughness=0.5, fresnel=0.0)
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, title=""),
            yaxis=dict(visible=False, title=""),
            zaxis=dict(visible=False, title=""),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    return fig


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
                    img_np = cv2.fastNlMeansDenoising(img_np, None, h=10, templateWindowSize=7, searchWindowSize=21)
                else:
                    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
            
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
    
    x, y = pad, plot_h - pad
    row_h = 0
    colors = ["#f4cccc", "#d9ead3", "#cfe2f3", "#fff2cc", "#d9d2e9", "#c2f0c2"]
    
    current_x = pad
    current_y = pad
    max_h_in_row = 0
    
    for i, r in enumerate(rooms):
        desired_area = max(0.1, r["area"])
        rect_area = desired_area * scale
        
        w = math.sqrt(rect_area) * 1.3 
        h = rect_area / w
        
        if current_x + w + pad > plot_w:
            current_x = pad
            current_y += max_h_in_row + pad
            max_h_in_row = 0
            
        if current_y + h + pad > plot_h:
            break
            
        rect = plt.Rectangle((current_x, current_y), w, h, 
                             facecolor=colors[i % len(colors)], 
                             edgecolor='black', 
                             linewidth=1.1)
        ax.add_patch(rect)
        
        ax.text(current_x + w / 2, current_y + h / 2, 
                f"{r['name'].replace('_', ' ').title()}\n{r['area']} m²", 
                ha='center', va='center', fontsize=8, color='black')
        
        current_x += w + pad
        max_h_in_row = max(max_h_in_row, h)
        
    ax.set_title(title, fontsize=14, fontweight='bold')
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
    st.image("https://placehold.co/110x110/000000/FFFFFF?text=QR", width=110) 
    st.markdown("<p style='font-size:13px; color:gray; text-align:right;'>Scan the QR to view the full project.</p>", unsafe_allow_html=True)

st.markdown("---")

mode = st.radio(
    "Select Mode:",
    ["GAN Generator", "Optimized Layout", "Real-Time Sensor Dashboard"],
    horizontal=True
)

if mode == "GAN Generator":
    st.header("GAN Floor Plan Generation")
    
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
                
                col.image(img, caption=f"Plan {i+1} (Raw GAN)", use_column_width=True)
                col.image(seg_img, caption=f"Segmented Plan {i+1}", use_column_width=True)

                col.download_button(
                    label=f"Download 2D Plan {i+1} (.png)",
                    data=buf.getvalue(),
                    file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png",
                    mime="image/png",
                )
                
                if O3D_AVAILABLE:
                    if col.button(f"Show 3D Model {i+1}"):
                        with st.spinner(f"Generating 3D mesh for Plan {i+1}..."):
                            mesh = floorplan_to_3d_mesh(seg_img)
                            if mesh:
                                st.success("3D mesh generated!")
                                
                                fig3d = plotly_preview(mesh)
                                if fig3d:
                                    st.plotly_chart(fig3d, use_container_width=True)
                                
                                temp_path = f"/tmp/floorplan_{i+1}.ply"
                                o3d.io.write_triangle_mesh(temp_path, mesh)
                                
                                st.download_button(
                                    label=f"Download 3D Model {i+1} (.ply)",
                                    data=open(temp_path, "rb").read(),
                                    file_name=f"floorplan_{i+1}.ply",
                                    mime="application/octet-stream"
                                )
                            else:
                                st.warning("Failed to generate 3D mesh from this floorplan.")
                else:
                    col.info("3D modeling disabled due to missing 'open3d' dependency.")


elif mode == "Real-Time Sensor Dashboard":
    st.header("Cloud Sensor Dashboard")
    st.markdown("Fetch ultrasonic readings one at a time and confirm whether it’s **Length** or **Breadth**.")

    for key in ["length", "breadth", "last_distance", "pir", "ir", "last_set"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.divider()

    if st.session_state.length is None and st.session_state.breadth is None and st.session_state.last_distance is None:
        if st.button("Get Sensor Data", use_container_width=True, type="primary"):
            try:
                r = requests.get("https://esp32-fastapi-server-uh47.onrender.com/data", timeout=5)
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    st.session_state.pir = d.get("pir")
                    st.session_state.ir = d.get("ir")
                    st.session_state.last_distance = d.get("ultrasonic")
                    if st.session_state.last_distance is None:
                        st.warning("No ultrasonic data found.")
                    st.rerun()
                else:
                    st.error(f"Server responded with {r.status_code}")
            except Exception as e:
                st.error(f"Error connecting to server: {e}")

    elif st.session_state.last_distance is not None:
        st.subheader("Last Measured Distance")
        st.write(f"**{st.session_state.last_distance} cm**")

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

        if st.button("Reset Last Measured Value", use_container_width=True):
            st.session_state.last_distance = None
            st.info("Last value cleared.")
            st.rerun()

    elif (st.session_state.length is not None) ^ (st.session_state.breadth is not None):
        st.info("Now get the other dimension.")
        if st.button("Get Sensor Data", use_container_width=True, type="primary"):
            try:
                r = requests.get("https://esp32-fastapi-server-uh47.onrender.com/data", timeout=5)
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    st.session_state.pir = d.get("pir")
                    st.session_state.ir = d.get("ir")
                    st.session_state.last_distance = d.get("ultrasonic")
                    if st.session_state.last_distance is None:
                        st.warning("No ultrasonic data found.")
                    st.rerun()
                else:
                    st.error(f"Server responded with {r.status_code}")
            except Exception as e:
                st.error(f"Error connecting to server: {e}")

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
            if st.button("Reset All", use_container_width=True, type="primary"):
                for k in ["length", "breadth", "last_distance", "pir", "ir", "last_set"]:
                    st.session_state[k] = None
                st.info("All cleared.")
                st.rerun()
        st.divider()
    
    st.divider()
    st.subheader("Current Measurements")
    st.write(f"Length: **{st.session_state.length if st.session_state.length else '—'} cm**")
    st.write(f"Breadth: **{st.session_state.breadth if st.session_state.breadth else '—'} cm**")

    if st.session_state.pir is not None or st.session_state.ir is not None:
        st.divider()
        st.subheader("Motion & Obstacle Sensors")
        pir_status = "Motion Detected (1)" if st.session_state.pir else "No Motion (0)"
        ir_status = "Obstacle Detected (0)" if not st.session_state.ir else "Clear Path (1)"
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
        st.write(f"**Calculated Total Area:** {area_m2:.2f} m² (≈ {area_sqft:.0f} sq ft)")

        bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1, key="sensor_beds")
        denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False, key="sensor_denoise")

        if st.button("Generate Floorplans", type="primary", use_container_width=True, key="sensor_generate"):
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
                    
                    col.image(img, caption=f"Plan {i+1} (Raw GAN)", use_column_width=True)
                    col.image(seg_img, caption=f"Segmented Plan {i+1}", use_column_width=True)
                    
                    col.download_button(
                        label=f"Download 2D Plan {i+1} (.png)",
                        data=buf.getvalue(),
                        file_name=f"plan_{i+1}_Area{int(area_sqft)}sqft_Beds{bedrooms}.png",
                        mime="image/png",
                    )
                    
                    if O3D_AVAILABLE:
                        if col.button(f"Show 3D Model {i+1}", key=f"sensor_3d_btn_{i}"):
                            with st.spinner(f"Generating 3D mesh for Plan {i+1}..."):
                                mesh = floorplan_to_3d_mesh(seg_img)
                                if mesh:
                                    st.success("3D mesh generated!")
                                    
                                    fig3d = plotly_preview(mesh)
                                    if fig3d:
                                        st.plotly_chart(fig3d, use_container_width=True)

                                    temp_path = f"/tmp/sensor_floorplan_{i+1}.ply"
                                    o3d.io.write_triangle_mesh(temp_path, mesh)
                                    
                                    st.download_button(
                                        label=f"Download 3D Model {i+1} (.ply)",
                                        data=open(temp_path, "rb").read(),
                                        file_name=f"sensor_floorplan_{i+1}.ply",
                                        mime="application/octet-stream"
                                    )
                                else:
                                    st.warning("Failed to generate 3D mesh from this floorplan.")
                    else:
                        col.info("3D modeling disabled due to missing 'open3d' dependency.")


elif mode == "Optimized Layout":
    st.header("Semantic Layout Optimization")
    
    colA, colB = st.columns(2)
    with colA:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
    with colB:
        num_rooms_input = st.number_input("Enter Total Number of Rooms", min_value=1, value=3)
        
    st.markdown("<p style='font-size:13px; color:gray;'>Note: The total number of rooms usually includes the kitchen and bathroom.</p>", unsafe_allow_html=True)
    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Bungalow"])
    plot_shape = st.selectbox("Plot Shape", ["Square", "Rectangular"])
    
    colW, colH = st.columns(2)
    with colW:
        plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=10.0)
    with colH:
        plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=10.0)
        
    if st.button("Generate Optimized Layout", type="primary"):
        with st.spinner("Generating layout..."):
            layout, _ = generate_semantic_layout(total_area, num_rooms_input, property_type, plot_shape, plot_w, plot_h)
            
            dwelling_type = predict_dwelling_type(total_area, layout["num_bedrooms"], RF_MODEL)
            st.success(f"Predicted Dwelling Type: **{dwelling_type}**")
            
            fig = plot_layout(layout, plot_w, plot_h, f"{property_type} Layout - {layout['num_bedrooms']} Bed")
            st.pyplot(fig)
            
            st.subheader("Room Area Breakdown (Conceptual)")
            
            room_data = [
                {"Room": r['name'].replace('_', ' ').title(), "Area (m²)": r['area']}
                for r in layout['rooms']
            ]
            st.table(room_data)

st.sidebar.header("Arch-Ai-Tex Chatbot")

api_key = ""

if not api_key:
    st.sidebar.error("LLM API key (ARCH_AI_TEX_CHATBOT) is not provided. Chatbot is running in mock mode.")
    
    def ask_llm(messages):
        user_query = messages[-1]['content']
        if "layout" in user_query.lower():
            return "Based on architectural best practices, a good functional layout prioritizes zoning: public (living, dining), private (bedrooms, bathrooms), and service (kitchen, utility). Aim for clear, straight pathways and minimize dead-end circulation space."
        else:
            return "That's a fascinating question! I can provide expert advice on that topic. Could you tell me more about the project, like the climate zone or desired aesthetic (e.g., minimalist, industrial)?"
    
else:
    def ask_llm(messages):
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

    answer = ask_llm(st.session_state.chat_history)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.sidebar.chat_message("assistant").write(answer)
