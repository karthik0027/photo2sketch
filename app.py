import io
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from models import UNetGenerator

st.set_page_config(page_title="Photo → Sketch GAN", page_icon="✏️", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = Path(__file__).parent / "G_final.pth"
WEIGHTS_URL = "https://github.com/karthik0027/photo2sketch/releases/download/v1/G_final.pth"


@st.cache_resource
def load_generator():
    if not WEIGHTS.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    G = UNetGenerator().to(DEVICE)
    G.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    G.eval()
    return G


def gan_sketch(G, rgb):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = G(x)[0, 0].cpu().numpy()
    out = ((y * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


def show_image(container, img, **kwargs):
    try:
        container.image(img, width="stretch", **kwargs)
    except Exception:
        container.image(img, use_container_width=True, **kwargs)


def to_png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


st.title("✏️ Photo → Sketch")
st.caption("A Pix2Pix GAN (U-Net generator + PatchGAN discriminator) that draws your photo as a pencil sketch.")

with st.spinner("Loading model (first run downloads the weights)..."):
    G = load_generator()

file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
if file is None:
    st.info("Upload a photo to get started.")
    st.stop()

rgb = np.array(Image.open(file).convert("RGB"))
with st.spinner("Sketching..."):
    sketch = gan_sketch(G, rgb)

left, right = st.columns(2)
left.subheader("Original")
show_image(left, rgb)
right.subheader("GAN sketch")
show_image(right, sketch, clamp=True)
right.download_button("Download sketch", data=to_png_bytes(sketch),
                      file_name="sketch.png", mime="image/png")
