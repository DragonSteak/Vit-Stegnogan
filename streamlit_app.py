import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import os

# ===== IMPORT YOUR MODEL =====
from Image_Sgan.model import Encoder, Decoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== PATHS (based on your repo) =====
ENCODER_PATH = "weights and deployment files/encoder.pth"
DECODER_PATH = "weights and deployment files/decoder.pth"

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))

    encoder.eval()
    decoder.eval()

    return encoder, decoder

encoder, decoder = load_models()

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def preprocess(img):
    return transform(img).unsqueeze(0).to(DEVICE)

def postprocess(tensor):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor)

# ===== METRIC =====
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ===== UI =====
st.set_page_config(page_title="StegaNoGAN Demo", layout="centered")

st.title("🔐 StegaNoGAN Image Steganography")

tab1, tab2 = st.tabs(["Encode", "Decode"])

# ================= ENCODE =================
with tab1:
    st.header("Hide Secret Image")

    cover_file = st.file_uploader("Upload Cover Image", type=["png", "jpg"])
    secret_file = st.file_uploader("Upload Secret Image", type=["png", "jpg"])

    if cover_file and secret_file:
        cover_img = Image.open(cover_file).convert("RGB")
        secret_img = Image.open(secret_file).convert("RGB")

        st.image([cover_img, secret_img], caption=["Cover", "Secret"], width=250)

        if st.button("Encode Image"):
            cover = preprocess(cover_img)
            secret = preprocess(secret_img)

            with torch.no_grad():
                # YOUR MODEL USES CONCAT
                x = torch.cat([cover, secret], dim=1)
                stego = encoder(x)

            stego_img = postprocess(stego)

            psnr_val = psnr(cover, stego).item()

            st.success("Encoding Complete")
            st.image(stego_img, caption="Stego Image")
            st.write(f"PSNR: {psnr_val:.2f}")

            # download
            import io
            buf = io.BytesIO()
            stego_img.save(buf, format="PNG")

            st.download_button(
                "Download Stego Image",
                data=buf.getvalue(),
                file_name="stego.png",
                mime="image/png"
            )

# ================= DECODE =================
with tab2:
    st.header("Recover Secret Image")

    stego_file = st.file_uploader("Upload Stego Image", type=["png", "jpg"])

    if stego_file:
        stego_img = Image.open(stego_file).convert("RGB")
        st.image(stego_img, caption="Stego Image", width=250)

        if st.button("Decode Image"):
            stego = preprocess(stego_img)

            with torch.no_grad():
                recovered = decoder(stego)

            recovered_img = postprocess(recovered)

            st.success("Decoding Complete")
            st.image(recovered_img, caption="Recovered Secret")
