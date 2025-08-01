## Image_Sgan — SGAN-Based Image Steganography

This module implements a **Simple Generative Adversarial Network (SGAN)** for image steganography — hiding and recovering secret images inside cover images using convolutional encoder-decoder networks and optional GAN-based training.

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries              |
|-----------------|-------------------------------|
| **Language**    | Python 3.8+                   |
| **Framework**   | PyTorch                       |
| **GAN Training**| Binary CrossEntropy Loss, Adam |
| **Visualization** | TensorBoard, Matplotlib     |
| **Dataset**     | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) paired cover/secret images |

---

## Architecture Overview

- **Encoder (CNN)**  
  Hides a secret image inside a cover image → generates a stego image.

- **Decoder (CNN)**  
  Recovers the secret from the stego image.

- **Discriminator (Optional)**  
  Used during GAN training to classify between real and stego images.

---

---

## Model Artifacts

| File                              | Format  | Purpose                                       |
|-----------------------------------|---------|-----------------------------------------------|
| `encoder.pth`, `decoder.pth`      | PyTorch | Trained encoder and decoder weights           |
| `encoder.onnx`, `decoder.onnx`    | ONNX    | Exported models for deployment/inference      |
| `DenseEncoder_DenseDecoder_*.dat` | Binary  | Snapshot of both models with metadata         |

---

## Why SGAN for Steganography?

- 🔹 Lightweight and fast
- 🔹 High perceptual quality
- 🔹 Easy to train and extend
- 🔹 Can be adapted to other modalities (e.g., text)

---

## Metrics

- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio

---

