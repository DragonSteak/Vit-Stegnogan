# Vit-Stegnogan

Vit-Stegnogan implements a Transformer-based Steganography system using a Vision Transformer (ViT) encoder and convolutional decoder. It hides arbitrary text data within high-resolution images and later retrieves the text from stego images. This project builds on adversarial learning (GAN) principles to generate visually indistinguishable stego images.

## Tech Stack

- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: TensorFlow 2.x (Keras API)
- **Core Modules**: Vision Transformer (ViT), Conv2DTranspose Decoder, CNN-based Discriminator
- **Data Processing**: tf.data API, custom preprocessing pipelines
- **Training**: Binary Crossentropy, Adversarial Loss, Reconstruction Loss
- **Evaluation**: SSIM, PSNR, Bit Accuracy
- **Visualization**: TensorBoard

## Features

- Transformer-based encoder for improved spatial-textual understanding
- Text-to-image steganography: embeds binary text into RGB images
- End-to-end training of encoder, decoder, and discriminator
- Supports ONNX export and `.h5`/`.pb` formats for deployment
- Visual quality monitoring through TensorBoard


## Dataset

- **DIV2K**: Used as the cover image base
- Text payload is converted to image tensor for Vision Transformer to process

## Model Evaluation
-**PSNR**: Measures similarity between cover and stego image

-**SSIM**: Measures structural similarity

-**Text Accuracy**: Measures exact bit match between embedded and recovered text
