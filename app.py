# app.py — Arch-Ai-Tex (clean full rewrite)
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
import requests
import time

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Arch-Ai-Tex", layout="centered", initial_sidebar_state="auto")

# ----------------------------
# Global constants
# ----------------------------
DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256

# ----------------------------
# Simple DCGAN-like Generator (small)
# ----------------------------
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

# ----------------------------
# Load models (generator + RF)
# ----------------------------
@st.cache_resource
def load_models():
    rf_model = None
    generator = DCGAN_Generator().to(DEVICE)
    # Try load RF (joblib)
    try:
        rf_model = joblib.load("room_predictor.joblib")
    except Exception:
        rf_model = None
    # Try generator weights
    loaded = False
    for fname in ("generator_epoch100.pth", "generator_epoch_100.pth", "generator.pth"):
        try:
            state_dict = torch.load(fname, map_location=DEVICE)
            generator.load_state_dict(state_dict, strict=False)
            loaded = True
            break
        except FileNotFoundError:
            continue
        except Exception:
            continue
    if not loaded:
        # keep generator but warn user
        st.warning("GAN generator weights not found — generator will produce random noise.")
    generator.eval()
    return rf_model, generator

RF_MODEL, GAN_MODEL = load_models()

# ----------------------------
# Utility functions
# ----------------------------
def predict_dwelling_type(area, bedrooms, rf_model):
    if rf_model is None:
        return "Unknown (RF model missing)"
    try:
        features = np.array([[float(area), int(bedrooms)]])
        return str(rf_model.predict(features)[0])
    except Exception:
        return "Prediction Failed"

def generate_final_plans(generator, area, bedrooms, count=3, denoise=False):
    if area < 100:
        area = 100
    pixel_area = area / (IMG_SIZE * IMG_SIZE)
    images = []
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
            img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(img)
    return images, pixel_area

def apply_segmentation(image):
    # Convert to grayscale and simple connected components coloring
    img_cv = np.array(image.convert("L"))
    _, thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    seg_rgb = np.zeros((*img_cv.shape, 3), dtype=np.uint8)
    room_colors = [
        (255, 199, 107), (130, 202, 157), (174, 199, 232),
        (255, 152, 150), (197, 176, 213), (255, 237, 111),
        (188, 189, 34), (140, 86, 75),
    ]
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        color = room_colors[(i - 1) % len(room_colors)]
        seg_rgb[labels == i] = color
    seg_pil = Image.fromarray(seg_rgb).resize(image.size)
    return seg_pil

def generate_semantic_layout(total_area, num_rooms_input, property_type, plot_w, plot_h):
    total_area = float(total_area)
    num_rooms_input = max(1, int(num_rooms_input))
    fixed_ratios = {"living+dining": 0.28, "kitchen": 0.08, "bathroom": 0.06}
    fixed_total = sum(fixed_ratios.values())
    num_bedrooms = max(0, num_rooms_input - len(fixed_ratios))
    remaining_ratio = max(0.0, 1.0 - fixed_total)
    rooms = []
    for name, ratio in fixed_ratios.items():
        rooms.append({"name": name, "area": round(total_area * ratio, 2)})
    if num_bedrooms > 0:
        per_bed_ratio = remaining_ratio / max(1, num_bedrooms)
        for i in range(num_bedrooms):
            rooms.append({"name": f"bedroom_{i+1}", "area": round(total_area * per_bed_ratio, 2)})
    else:
        if remaining_ratio > 0.01:
            rooms.append({"name": "utility/other", "area": round(total_area * remaining_ratio, 2)})
    current_sum = round(sum(r["area"] for r in rooms), 2)
    diff = round(total_area - current_sum, 2)
    if abs(diff) >= 0.01 and rooms:
        rooms[0]["area"] = round(rooms[0]["area"] + diff, 2)
    return {"rooms": rooms, "num_bedrooms": num_bedrooms}

def plot_layout(layout, plot_w, plot_h, title="Layout"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, plot_h)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), plot_w, plot_h, fill=False, edgecolor='black', linewidth=1.0))
    rooms = layout.get("rooms", [])
    total_area = sum(r["area"] for r in rooms) or 1.0
    scale = (plot_w * plot_h) / total_area
    pad = min(plot_w, plot_h) * 0.02
    x, y = pad, pad
    row_h = 0
    colors = ["#f4cccc", "#d9ead3", "#cfe2f3", "#fff2cc", "#d9d2e9", "#c2f0c2"]
    for i, r in enumerate(rooms):
        desired_area = max(0.1, r["area"])
        rect_area = desired_area * scale
        w = math.sqrt(rect_area) * 1.1
        h = rect_area / w
        if x + w + pad > plot_w:
            x = pad
            y += row_h + pad
            row_h = 0
        if y + h + pad > plot_h:
            break
        rect = plt.Rectangle((x, y), w, h, facecolor=colors[i % len(colors)], edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{r['name']}\n{r['area']} m²", ha='center', va='center', fontsize=8)
        x += w + pad
        row_h = max(row_h, h)
    ax.set_title(title)
    plt.tight_layout()
    return fig

# ----------------------------
# Styles & Header
# ----------------------------
st.markdown("""
<style>
/* App primary button style */
.stButton>button {
    background-color: #16a34a;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 1.0em;
    border: none;
}
.stButton>button:hover { transform: translateY(-2px); filter: brightness(0.95); }
.container-centered { max-width: 1100px; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("Arch-Ai-Tex")
    st.markdown("AI Floor Plan Generator")
with col2:
    try:
        st.image("QR.png", width=90)
    except Exception:
        pass

st.markdown("---")

# ----------------------------
# Main Mode Selector
# ----------------------------
mode = st.radio("Select Mode:", ["GAN Generator", "Optimized Layout", "Real-Time Sensor Dashboard"], horizontal=True)

# ----------------------------
# Mode: GAN Generator
# ----------------------------
if mode == "GAN Generator":
    c1, c2 = st.columns(2)
    with c1:
        house_length = st.number_input("Enter House Length (m)", min_value=1.0, value=50.0, step=1.0)
    with c2:
        house_width = st.number_input("Enter House Width (m)", min_value=1.0, value=30.0, step=1.0)
    area_m2 = max(100.0, house_length * house_width)
    area_sqft = area_m2 * 10.7639
    st.markdown(f"**Calculated Total Area:** {area_m2:.2f} m² (≈ {area_sqft:.0f} sq ft)")
    bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1)
    denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False)
    generate_count = st.slider("Number of Plans to Generate", 1, 6, 3)
    if st.button("Generate Floorplans", use_container_width=True):
        with st.spinner("Generating..."):
            images, pixel_area = generate_final_plans(GAN_MODEL, area_m2, bedrooms, count=generate_count, denoise=denoise_option)
        st.subheader(f"Generated {len(images)} Floorplans")
        st.markdown(f"**Area to Pixel Ratio:** 1 pixel ≈ {pixel_area:.4f} m²")
        cols = st.columns(min(3, len(images)))
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            cols[i % 3].image(img, caption=f"Plan {i+1}", use_column_width=True)
            cols[i % 3].download_button(
                label="Download PNG",
                data=buf.getvalue(),
                file_name=f"plan_{i+1}_area_{int(area_sqft)}sqm_beds_{bedrooms}.png",
                mime="image/png"
            )

# ----------------------------
# Mode: Real-Time Sensor Dashboard
# ----------------------------
elif mode == "Real-Time Sensor Dashboard":
    st.header("Cloud Sensor Dashboard")
    st.markdown("Fetch ultrasonic readings and set Length / Breadth from sensor.")

    # initialize session keys
    for k in ("length", "breadth", "last_distance", "pir", "ir", "last_set"):
        if k not in st.session_state:
            st.session_state[k] = None

    if st.session_state.length is None and st.session_state.breadth is None and st.session_state.last_distance is None:
        if st.button("Get Sensor Data"):
            try:
                r = requests.get("https://esp32-fastapi-server-uh47.onrender.com/data", timeout=5)
                r.raise_for_status()
                d = r.json().get("data", {})
                st.session_state.pir = d.get("pir")
                st.session_state.ir = d.get("ir")
                st.session_state.last_distance = d.get("ultrasonic")
                if st.session_state.last_distance is None:
                    st.warning("No ultrasonic data found.")
            except Exception as e:
                st.error(f"Error fetching sensor data: {e}")

    # If a last reading exists, let user assign it
    if st.session_state.last_distance is not None:
        st.subheader("Last Measured Distance")
        st.write(f"{st.session_state.last_distance} cm")
        if st.session_state.length is None:
            if st.button("Set as Length"):
                st.session_state.length = st.session_state.last_distance
                st.session_state.last_set = "length"
                st.session_state.last_distance = None
        if st.session_state.breadth is None:
            if st.button("Set as Breadth"):
                st.session_state.breadth = st.session_state.last_distance
                st.session_state.last_set = "breadth"
                st.session_state.last_distance = None
        if st.button("Reset Last Value"):
            st.session_state.last_distance = None
            st.info("Cleared last measurement.")

    st.markdown("---")
    st.subheader("Current Measurements")
    st.write(f"Length: {st.session_state.length if st.session_state.length else '—'} cm")
    st.write(f"Breadth: {st.session_state.breadth if st.session_state.breadth else '—'} cm")

    if st.session_state.length and st.session_state.breadth:
        st.divider()
        st.subheader("Generate Floorplan from Captured Dimensions")
        length_m = st.session_state.length * 0.01
        breadth_m = st.session_state.breadth * 0.01
        area_m2 = max(100.0, length_m * breadth_m)
        area_sqft = area_m2 * 10.7639
        st.write(f"Final Dimensions: {length_m:.2f} m × {breadth_m:.2f} m")
        st.write(f"Calculated Area: {area_m2:.2f} m² (≈ {area_sqft:.0f} sq ft)")
        bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, value=3, step=1, key="sensor_beds")
        denoise_option = st.checkbox("Apply Denoiser (OpenCV)", value=False, key="sensor_denoise")
        if st.button("Generate Floorplans from Sensor Dimensions"):
            with st.spinner("Generating..."):
                images, pixel_area = generate_final_plans(GAN_MODEL, area_m2, bedrooms, count=3, denoise=denoise_option)
            cols = st.columns(3)
            for i, img in enumerate(images):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                cols[i].image(img, caption=f"Plan {i+1}")
                cols[i].download_button("Download", buf.getvalue(), file_name=f"sensor_plan_{i+1}.png", mime="image/png")

# ----------------------------
# Mode: Optimized Layout
# ----------------------------
elif mode == "Optimized Layout":
    st.header("Optimized Layout Generator")
    left, right = st.columns(2)
    with left:
        total_area = st.number_input("Enter Total Area (sqm)", min_value=30.0, value=120.0, step=10.0)
        num_rooms = st.number_input("Enter Total Number of Rooms", min_value=1, value=3, step=1)
    with right:
        property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Bungalow"])
        plot_w = st.number_input("Plot Width (m)", min_value=5.0, value=10.0)
        plot_h = st.number_input("Plot Height (m)", min_value=5.0, value=10.0)
    if st.button("Generate Optimized Layout"):
        with st.spinner("Optimizing layout..."):
            layout = generate_semantic_layout(total_area, num_rooms, property_type, plot_w, plot_h)
            dwelling_type = predict_dwelling_type(total_area, layout.get("num_bedrooms", 0), RF_MODEL)
            fig = plot_layout(layout, plot_w, plot_h, title=f"{property_type} Layout")
        st.success(f"Predicted Dwelling Type: {dwelling_type}")
        st.pyplot(fig)

# ----------------------------
# Floating Chatbot widget (Left corner) — stable implementation
# ----------------------------
# This implementation uses a Streamlit-managed floating button (styled) to toggle a chat panel.
# The panel itself contains Streamlit elements (so no fragility with raw JS or experimental_rerun).

# Ensure chat session keys
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "floating_chat_history" not in st.session_state:
    # store messages as list of {"role":"user"/"assistant","content": "..."}
    st.session_state.floating_chat_history = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are an expert AEC/BIM architect and engineer. Answer clearly and concisely. "
        "Provide checklists and step-by-step guidance when helpful."
    )

# Groq HTTP helper (uses requests) — no groq SDK required
def call_groq_chat(messages, model="deepseek-r1-distill-llama-70b", max_tokens=400, timeout=30):
    api_key = st.secrets.get("ARCH_AI_TEX_CHATBOT") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return "Error: ARCH_AI_TEX_CHATBOT not set in Streamlit secrets."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        choice = j.get("choices", [{}])[0]
        # new shape: choice.message.content or fallback choice.text
        if isinstance(choice.get("message"), dict):
            return choice["message"].get("content", "")
        else:
            return choice.get("text", "")
    except Exception as e:
        return f"API Error: {e}"

# Toggle chat callback (Streamlit-safe)
def toggle_chat():
    st.session_state.chat_open = not st.session_state.chat_open

# Floating button CSS (left-bottom)
st.markdown(
    """
    <style>
    .floating-button-area {
        position: fixed;
        left: 18px;
        bottom: 20px;
        z-index: 2000;
    }
    .floating-button-area .stButton>button {
        width: 65px;
        height: 65px;
        padding: 0;
        border-radius: 50%;
        background-color: #ff4b4b;
        color: white;
        font-size: 28px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        border: none;
    }
    .chat-panel {
        position: fixed;
        left: 18px;
        bottom: 100px;
        width: 380px;
        height: 520px;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        z-index: 2000;
        padding: 12px;
        overflow-y: auto;
    }
    .chat-user { background: #d0f0ff; padding: 8px 12px; border-radius: 10px; margin: 8px 0; width: 85%; }
    .chat-bot  { background: #fff3cd; padding: 8px 12px; border-radius: 10px; margin: 8px 0; width: 85%; }
    .chat-controls { display:flex; gap:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render floating button area
floating_button = st.empty()
with floating_button.container():
    st.markdown('<div class="floating-button-area">', unsafe_allow_html=True)
    st.button("💬", key="floating_chat_toggle_btn", on_click=toggle_chat)
    st.markdown("</div>", unsafe_allow_html=True)

# Render chat panel if open
if st.session_state.chat_open:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

    # header and close
    header_cols = st.columns([0.8, 0.2])
    with header_cols[0]:
        st.markdown("<b>Arch-Ai-Tex ChatBot</b>", unsafe_allow_html=True)
    with header_cols[1]:
        if st.button("Close", key="floating_chat_close"):
            st.session_state.chat_open = False

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Show history
    for msg in st.session_state.floating_chat_history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(f"<div class='chat-user'><b>You:</b> {content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'><b>Bot:</b> {content}</div>", unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Input + actions
    user_input = st.text_input("Type your message...", value="", key="floating_chat_input_box")
    col_send, col_clear = st.columns([0.6, 0.4])
    with col_send:
        if st.button("Send", key="floating_chat_send"):
            if user_input.strip():
                st.session_state.floating_chat_history.append({"role": "user", "content": user_input})
                # prepare messages: start with system prompt then history
                messages_for_api = [{"role": "system", "content": st.session_state.system_prompt}]
                for m in st.session_state.floating_chat_history:
                    # API expects "user" / "assistant"
                    api_role = "assistant" if m["role"] == "assistant" else "user"
                    messages_for_api.append({"role": api_role, "content": m["content"]})
                # call API
                with st.spinner("Thinking..."):
                    reply = call_groq_chat(messages_for_api)
                st.session_state.floating_chat_history.append({"role": "assistant", "content": reply})
                # clear text input state
                st.session_state.floating_chat_input_box = ""
    with col_clear:
        if st.button("Clear Chat", key="floating_chat_clear"):
            st.session_state.floating_chat_history = []

    st.markdown("</div>", unsafe_allow_html=True)

# Footer / small note
st.markdown("---")
st.markdown("Built with ❤️ — Arch-Ai-Tex")

# https://esp32-fastapi-server-uh47.onrender.com/data
