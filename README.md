# Image-resolution-Visual-Cryptography

# ***Advanced Color Visual Cryptography Model utilizing Disentangled Graph Variational Autoencoder for High Quality Image Reconstruction with Noise Minimization***

This notebook presents an advanced color visual cryptography model that leverages a Disentangled Graph Variational Autoencoder (DVAE) for high-quality image reconstruction with noise minimization. The project explores the application of visual cryptography techniques combined with deep learning for secure image sharing and enhanced reconstruction.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Model Architecture](#model-architecture)
4.  [Optimization Algorithm](#optimization-algorithm)
5.  [Training](#training)
6.  [Results](#results)
7.  [Visual Cryptography Implementation](#visual-cryptography-implementation)
8.  [Performance Metrics](#performance-metrics)

## Introduction

Visual cryptography encrypts images into multiple "shares" that reveal the original image when combined. DVAE is used to enhance image reconstruction quality and reduce noise.

## Dataset

Raw Data: High/low-res image pairs.

train / val: Used for model training/validation.

Includes visualization and split logic.

The notebook includes code to load the dataset, visualize the data distribution, and split the data into training, validation, and test sets.

## Model Architecture

Encoder: Residual + PReLU layers.

Bottleneck: Conv + PReLU.

Decoder: Conv2DTranspose, skip connections, CBAM attention blocks.

Learns disentangled representations for accurate reconstruction.

## Optimization Algorithm

Superb Fairy-wren Optimization Algorithm (SFOA) is included.
Demonstrated on sample function applied directly to DVAE

## Training
The DVAE model is trained using the AdamW optimizer with a Cosine Decay learning rate schedule
.Loss: Mean Absolute Error (MAE)
.Optimizer: AdamW + Cosine Decay
.Callbacks: EarlyStopping, ModelCheckpoint
.Metrics: PSNR, SSIM, MSE


## Results
Visualizes original, low-res, and DVAE-reconstructed images.
Quantitative performance shown using standard metrics.
The loss function used is Mean Absolute Error (MAE). The model is compiled with PSNR, SSIM, and MSE as evaluation metrics.

## Visual Cryptography Implementation

Functions:

Channel decomposition, histogram, binning

Otsu-based thresholding via optimize_color_level_sfoa

Share generation using XOR

Share combination and decryption

End-to-end processing with visualization.


## Performance Metrics

The notebook calculates and prints the average performance metrics (PSNR, SSIM, Correlation, MSE, RMSE, and MAE) for the DVAE model's predictions on the test set. These metrics provide a quantitative evaluation of the model's ability to reconstruct high-quality images.
Metric	Value
PSNR	30.75 dB
SSIM	0.9018
MSE	0.00106
RMSE	0.0325
MAE	0.0191
Corr	0.9892
## Usage

To run this notebook:

1.  Ensure you have the required libraries installed (tensorflow, keras, numpy, cv2, matplotlib, tqdm, re, skimage).
2.  Place your dataset in the appropriate directory structure as expected by the `base_path` variable.
3.  Run the cells sequentially.

This notebook provides a comprehensive approach to image reconstruction using a DVAE and integrates visual cryptography concepts.
