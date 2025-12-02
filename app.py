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
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels

warnings.filterwarnings("ignore", message="missing ScriptRunContext")
st.set_page_config(page_title="Arch-Ai-Tex", layout="centered")

# ---------------------------
# Config / constants
# ---------------------------
DEVICE = torch.device("cpu")
LATENT_DIM = 100
CHANNELS = 1
IMG_SIZE = 256
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
            DCGAN_Generator.block(512, 256),
            DCGAN_Generator.block(256, 128),
            DCGAN_Generator.block(128, 64),
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.size(0), 512, 16, 16)
        return self.gen(out)

# ---------------------------
# Load segmentation model
# ---------------------------

@st.cache_resource
def load_segmentation_model():
    model = segmodels.fcn_resnet50(pretrained=True).to(DEVICE)
    model.eval()
    return model

SEG_MODEL = load_segmentation_model()

SEG_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229]) if CHANNELS == 1 else
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Apply segmentation
# ---------------------------

def run_segmentation(pil_img):
    if CHANNELS == 1:
        img3 = np.repeat(np.array(pil_img)[..., None], 3, axis=2)
        pil_img = Image.fromarray(img3)

    img_t = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])(pil_img).unsqueeze(0)

    with torch.no_grad():
        out = SEG_MODEL(img_t)["out"][0]
        seg = out.argmax(0).byte().cpu().numpy()

    # convert to color map
    colors = np.random.randint(0,255,(21,3),dtype=np.uint8)
    seg_rgb = colors[seg]
    return Image.fromarray(seg_rgb)

# ---------------------------
# Load RF + GAN
# ---------------------------
@st.cache_resource
def load_models():
    rf_model = None
    gen = DCGAN_Generator().to(DEVICE)

    if os.path.exists(RF_MODEL_PATH):
        try:
            rf_model = joblib.load(RF_MODEL_PATH)
        except:
            rf_model = None

    if os.path.exists(GEN_WEIGHTS_PATH):
        try:
            gen.load_state_dict(torch.load(GEN_WEIGHTS_PATH, map_location=DEVICE))
        except:
            pass

    gen.eval()
    return rf_model, gen

RF_MODEL, GAN_MODEL = load_models()

# ---------------------------
# Prediction
# ---------------------------

def predict_dwelling_type(area_m2, bedrooms, rf_model):
    if rf_model is None:
        return "Unknown (No RF Model)"
    try:
        x = np.array([[float(area_m2), int(bedrooms)]])
        return rf_model.predict(x)[0]
    except:
        return "Prediction Failed"

# ---------------------------
# GAN generation + segmentation
# ---------------------------

def generate_final_plans(generator, area_m2, bedrooms, count=3, denoise=False, rf_model=None):
    dwelling_type = predict_dwelling_type(area_m2, bedrooms, rf_model)

    images = []
    seg_images = []

    area_m2 = max(100.0, float(area_m2))
    pixel_area = area_m2 / (IMG_SIZE * IMG_SIZE)

    for _ in range(count):
        z = torch.randn(1, LATENT_DIM).to(DEVICE)

        with torch.no_grad():
            img = generator(z).squeeze().cpu().numpy()

        if CHANNELS == 1:
            if img.ndim == 3: img = img[0]
            img = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)
        else:
            img = np.transpose(img, (1,2,0))
            img = np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)

        if denoise:
            img = cv2.fastNlMeansDenoising(img, None, 10)

        pil = Image.fromarray(img, mode="L" if CHANNELS==1 else "RGB")
        pil = pil.resize((IMG_SIZE, IMG_SIZE))

        seg = run_segmentation(pil)

        images.append(pil)
        seg_images.append(seg)

    return dwelling_type, images, seg_images, pixel_area

# ---------------------------
# Layout functions (unchanged)
# ---------------------------

def generate_semantic_layout(total_area, num_bedrooms):
    total_area = float(total_area)
    num_bedrooms = max(0, int(num_bedrooms))
    fixed = {"living+dining":0.28,"kitchen":0.08,"bathroom":0.06}
    rooms=[]
    for n,r in fixed.items(): rooms.append({"name":n,"area":round(total_area*r,2)})

    rem = 1-sum(fixed.values())
    if num_bedrooms>0:
        each = rem/num_bedrooms
        for i in range(num_bedrooms):
            rooms.append({"name":f"bedroom_{i+1}","area":round(total_area*each,2)})
    else:
        rooms.append({"name":"utility","area":round(total_area*rem,2)})

    # fix rounding
    diff = round(total_area-sum(r["area"] for r in rooms),2)
    rooms[0]["area"]+=diff
    return {"rooms":rooms},""

def plot_layout(layout, w, h, title="Layout"):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0,w); ax.set_ylim(0,h)
    ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(plt.Rectangle((0,0),w,h,fill=False,edgecolor='black'))

    rooms = layout["rooms"]
    tot = sum(r["area"] for r in rooms)
    scale = (w*h)/tot
    pad=0.1; x=pad; y=pad; rh=0
    colors=["#f4cccc","#d9ead3","#cfe2f3","#fff2cc","#d9d2e9","#c2f0c2"]

    for i,r in enumerate(rooms):
        a=r["area"]*scale
        rw=math.sqrt(a); rh_rect=a/max(rw,1e-6)

        if x+rw+pad>w: x=pad; y+=rh+pad; rh=0
        if y+rh_rect+pad>h: break

        ax.add_patch(plt.Rectangle((x,y),rw,rh_rect,facecolor=colors[i%6],edgecolor='black'))
        ax.text(x+rw/2,y+rh_rect/2,f"{r['name']}\n{r['area']} m²",ha="center",va="center",fontsize=8)

        x+=rw+pad
        rh=max(rh,rh_rect)

    ax.set_title(title)
    return fig

# ---------------------------
# UI
# ---------------------------

st.title("Arch-Ai-Tex")
st.markdown("---")

mode = st.radio("Select Model:", ["GAN Generator","Optimized Layout"], horizontal=True)

# ---------------------------
# GAN MODE
# ---------------------------
if mode=="GAN Generator":
    l,w = st.columns(2)
    with l:
        L = st.number_input("House Length (m)", min_value=1.0, value=10.0)
    with w:
        W = st.number_input("House Width (m)", min_value=1.0, value=12.0)

    area = max(100.0, L*W)
    st.write(f"**Area:** {area:.2f} m²")

    beds = st.number_input("Bedrooms", min_value=0, value=3)
    denoise = st.checkbox("Apply Denoise", False)

    if st.button("Generate Floorplans"):
        dw, imgs, segs, px = generate_final_plans(GAN_MODEL, area, beds,
                                                  count=3, denoise=denoise,
                                                  rf_model=RF_MODEL)

        st.subheader(f"Predicted Dwelling Type: {dw}")
        st.write(f"1 pixel ≈ {px:.4f} m²")

        cols = st.columns(3)
        for i,(im,seg) in enumerate(zip(imgs,segs)):
            with cols[i]:
                st.image(im, caption=f"Plan {i+1}")
                st.image(seg, caption="Segmentation")

# ---------------------------
# LAYOUT MODE
# ---------------------------
else:
    st.header("Optimized Layout")
    A = st.number_input("Total Area (sqm)", value=120.0)
    B = st.number_input("Bedrooms", value=3)
    pw = st.number_input("Plot Width", value=10.0)
    ph = st.number_input("Plot Height", value=12.0)

    if st.button("Generate Layout"):
        layout,_ = generate_semantic_layout(A,B)
        for r in layout["rooms"]:
            st.write(f"**{r['name']}** → {r['area']} m²")

        fig = plot_layout(layout,pw,ph)
        st.pyplot(fig)
