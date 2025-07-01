# Image-resolution-Visual-Cryptography
# ***Advanced Color Visual Cryptography Model utilizing Disentangled Graph Variational Autoencoder for High Quality Image Reconstruction with Noise Minimization***

This project presents an advanced Color Visual Cryptography Model enhanced with a Disentangled Graph Variational Autoencoder (DVAE) for high-quality image reconstruction and noise minimization. It combines deep learning with cryptographic techniques for secure and accurate image sharing.


Introduction
Dataset
Model Architecture
Optimization Algorithm
Training
Results
Visual Cryptography Implementation
Performance Metrics

Performance Metrics
Introduction
Visual cryptography encrypts images into share images that, when stacked, reveal the original. This project enhances VC with DVAE to reconstruct cleaner, high-quality images and ensure secure transmission using deep learning.

Dataset
High- and low-resolution image pairs organized as:

Raw Data: Original images

train: Training set

val: Validation set
Data split, loading, and visualization are included in the notebook.

 Model Architecture
The DVAE consists of:

Encoder: Downsampling + residual blocks + PReLU

Bottleneck: Convolutional layers

Decoder: Transposed Conv + skip connections + CBAM (attention)

CBAM: Enhances features via channel & spatial attention

 Optimization Algorithm
Introduces the Superb Fairy-wren Optimization Algorithm (SFOA) for optimal thresholding directly used in DVAE training.

 Training
Optimizer: AdamW with Cosine Decay

Loss: Mean Absolute Error (MAE)

Metrics: PSNR, SSIM, MSE

Callbacks: EarlyStopping, ModelCheckpoint

 Results
Visualizes:
High-resolution original
Low-resolution input
DVAE-reconstructed output
Evaluates using PSNR, SSIM, and correlation.
Visual Cryptography Implementation
Includes functions:
decompose_channels, compute_histogram, group_pixels

optimize_color_level_sfoa, generate_boolean_shares

combine_color_shares, decrypt_color_shares

encrypt_image, generate_color_shares, process_visual_cryptography

Encrypts RGB images into two shares and decrypts via XOR.

Performance Metrics
Metric	Value
PSNR	30.75 dB
SSIM	0.9017
MSE	0.00106
RMSE	0.0325
MAE	0.0191
Corr	0.9892

Usage
Install dependencies: tensorflow, keras, cv2, etc.

Set your dataset path (base_path)

Run the notebook cells sequentially

