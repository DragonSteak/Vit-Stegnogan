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

        # ✅ FORCE SAME SIZE
        cover_img = cover_img.resize((256, 256))
        secret_img = secret_img.resize((256, 256))

        st.image(cover_img, caption="Cover Image")
        st.image(secret_img, caption="Secret Image")

        # tensors
        cover_tensor = transforms.ToTensor()(cover_img).unsqueeze(0).to(DEVICE)
        secret_tensor = transforms.ToTensor()(secret_img).unsqueeze(0).to(DEVICE)

        # ✅ CRITICAL FIX: ensure correct channel depth
        if secret_tensor.shape[1] != DATA_DEPTH:
            secret_tensor = secret_tensor[:, 0:1, :, :]  # force 1 channel

        # ✅ FINAL SAFETY: match spatial dims
        if cover_tensor.shape[2:] != secret_tensor.shape[2:]:
            secret_tensor = torch.nn.functional.interpolate(
                secret_tensor, size=cover_tensor.shape[2:]
            )

        with torch.no_grad():
            stego = encoder(cover_tensor, secret_tensor)

        stego_img = stego.squeeze().permute(1, 2, 0).cpu().numpy()
        stego_img = np.clip(stego_img, 0, 1)

        st.image(stego_img, caption="Stego Image")

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
