import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# ✅ correct import
from Image_Sgan.model import DenseEncoder, DenseDecoder

# ✅ config (must match training)
DEVICE = torch.device("cpu")
DATA_DEPTH = 4
HIDDEN_SIZE = 32

ENCODER_PATH = "weights and deployment files/encoder.pth"
DECODER_PATH = "weights and deployment files/decoder.pth"

# ✅ image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ✅ load model once
@st.cache_resource
def load_models():
    encoder = DenseEncoder(DATA_DEPTH, HIDDEN_SIZE).to(DEVICE)
    decoder = DenseDecoder(DATA_DEPTH, HIDDEN_SIZE).to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

encoder, decoder = load_models()

# ✅ UI
st.title("🔐 Steganography Demo (ViT-SGAN)")

mode = st.radio("Select Mode", ["Encode", "Decode"])

# =========================
# 🔒 ENCODE
# =========================
if mode == "Encode":
    cover_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
    secret_file = st.file_uploader("Upload Secret Image", type=["png", "jpg", "jpeg"])

    if cover_file and secret_file:
        cover_img = Image.open(cover_file).convert("RGB")
        secret_img = Image.open(secret_file).convert("L")  # grayscale

        st.image(cover_img, caption="Cover Image", use_container_width=True)
        st.image(secret_img, caption="Secret Image", use_container_width=True)

        cover_tensor = transform(cover_img).unsqueeze(0).to(DEVICE)
        secret_tensor = transform(secret_img).unsqueeze(0).to(DEVICE)

        # ensure correct shape for data_depth=1
        if secret_tensor.shape[1] != DATA_DEPTH:
            secret_tensor = secret_tensor[:, :DATA_DEPTH, :, :]

        with torch.no_grad():
            stego = encoder(cover_tensor, secret_tensor)

        stego_img = stego.squeeze().permute(1, 2, 0).cpu().numpy()
        stego_img = np.clip(stego_img, 0, 1)

        st.image(stego_img, caption="Stego Image", use_container_width=True)

# =========================
# 🔓 DECODE
# =========================
if mode == "Decode":
    stego_file = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg"])

    if stego_file:
        stego_img = Image.open(stego_file).convert("RGB")
        st.image(stego_img, caption="Stego Image", use_container_width=True)

        stego_tensor = transform(stego_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            recovered = decoder(stego_tensor)

        recovered_img = recovered.squeeze().cpu().numpy()

        # normalize for display
        recovered_img = (recovered_img - recovered_img.min()) / (
            recovered_img.max() - recovered_img.min() + 1e-8
        )

        st.image(recovered_img, caption="Recovered Secret", use_container_width=True)
