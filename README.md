# SteganoGAN Suite: Image & Text Hiding

This repository contains two modular steganography frameworks for securely embedding data inside images using adversarial learning techniques:

1. **Image_Sgan** – A GAN-based pipeline for hiding an text within an image using convolutional networks.
2. **Vit_Stegnogan** – A transformer-powered steganography system that hides arbitrary text data as image inside high-resolution RGB images.

Both systems are designed with reproducibility, deployment readiness, and metric-based evaluation in mind.

---

## 🔐 1. Image_Sgan

> A convolutional GAN (SGAN)-based steganography system that hides a secret image inside a cover image using a Dense or Residual encoder-decoder structure, trained adversarially against a CNN discriminator.

### Features

- Text-to-image hiding with minimal visual distortion
- Discriminator ensures realism of stego images
- Evaluated using PSNR, SSIM, and adversarial accuracy
- Export support: PyTorch `.pth` and ONNX formats

### Tech Stack

- **Programming Language**: Python 3.8+
- **Framework**: PyTorch
- **Models**: CNN-based Encoder, Decoder, and Discriminator
- **Training**: Binary Crossentropy, GAN loss
- **Evaluation**: PSNR, SSIM
- **Utilities**: ONNX export, TensorBoard logging

---

## 2. Vit-Stegnogan

> Vit-Stegnogan implements a Transformer-based steganography system using a Vision Transformer (ViT) encoder and convolutional decoder. It hides arbitrary **text data** within high-resolution images and retrieves it with high accuracy. Adversarial learning principles ensure the generated stego images are indistinguishable from clean ones.

### Features

- Transformer-based encoder for improved spatial-textual correlation
- Text-to-image steganography: embeds binary text as image inside RGB images
- End-to-end training with GAN architecture
- Compatible with ONNX, `.h5`, and `.pb` model export
- TensorBoard visualization support for metrics and outputs

### Tech Stack

- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: TensorFlow 2.x (Keras API)
- **Core Modules**: Vision Transformer (ViT), Conv2DTranspose Decoder, CNN-based Discriminator
- **Data Processing**: `tf.data` API, custom preprocessing pipelines
- **Training**: Binary Crossentropy, Adversarial Loss, Reconstruction Loss
- **Evaluation**: SSIM, PSNR, Bit Accuracy
- **Visualization**: TensorBoard

### Dataset

- **DIV2K**: Used for high-resolution cover images
- **Text Payload**: Converted into binary images, then embedded using ViT

### Model Evaluation

- **PSNR**: Measures similarity between cover and stego image
- **SSIM**: Assesses structural integrity
- **Text Accuracy**: Exact bitwise match between input and recovered text


## 🧪 Shared Evaluation Metrics

| Metric        | Description                                                   |
|---------------|---------------------------------------------------------------|
| **PSNR**      | Measures pixel-level similarity between cover and stego image |
| **SSIM**      | Assesses perceptual similarity and structure                  |
| **Bit Accuracy** | Evaluates the exact match of hidden and recovered data     |
| **Discriminator Confidence** | Real-vs-fake accuracy from the GAN discriminator |

---

## 📂 Checkpoints & Assets

| File                      | Type            | Description                              |
|---------------------------|------------------|------------------------------------------|
| `encoder.pth`, `decoder.pth` | PyTorch Weights | SGAN Encoder/Decoder models              |
| `encoder.onnx`, `decoder.onnx` | ONNX Models     | Optimized for cross-platform deployment  |
| `.h5`, `.pb` files         | TF/Keras Models  | ViT-Stegnogan model exports              |
| `*.dat`                   | Archive          | Combined model/data snapshots            |

---

