import streamlit as st
from PIL import Image
import torch
import numpy as np

# ===== Load Models (replace with your actual models) =====
@st.cache_resource
def load_models():
    encoder = None  # load your encoder
    decoder = None  # load your decoder
    return encoder, decoder

encoder, decoder = load_models()

# ===== Helper functions =====
def preprocess(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    return torch.tensor(image).float().unsqueeze(0)

def postprocess(tensor):
    tensor = tensor.squeeze().detach().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(tensor)

# ===== UI =====
st.title("🔐 ViT-StegaNoGAN Demo")

tab1, tab2 = st.tabs(["Encode", "Decode"])

# ================= ENCODE =================
with tab1:
    st.header("Hide Secret Image")

    cover_file = st.file_uploader("Upload Cover Image", type=["png","jpg"])
    secret_file = st.file_uploader("Upload Secret Image", type=["png","jpg"])

    if cover_file and secret_file:
        cover_img = Image.open(cover_file).convert("RGB")
        secret_img = Image.open(secret_file).convert("RGB")

        st.image([cover_img, secret_img], caption=["Cover", "Secret"], width=250)

        if st.button("Encode"):
            cover = preprocess(cover_img)
            secret = preprocess(secret_img)

            # ===== YOUR MODEL HERE =====
            with torch.no_grad():
                stego = cover  # replace with encoder(cover, secret)

            stego_img = postprocess(stego)

            st.success("Encoding Complete")
            st.image(stego_img, caption="Stego Image")

# ================= DECODE =================
with tab2:
    st.header("Recover Secret Image")

    stego_file = st.file_uploader("Upload Stego Image", type=["png","jpg"])

    if stego_file:
        stego_img = Image.open(stego_file).convert("RGB")
        st.image(stego_img, caption="Stego Image", width=250)

        if st.button("Decode"):
            stego = preprocess(stego_img)

            # ===== YOUR MODEL HERE =====
            with torch.no_grad():
                recovered = stego  # replace with decoder(stego)

            recovered_img = postprocess(recovered)

            st.success("Decoding Complete")
            st.image(recovered_img, caption="Recovered Secret")
