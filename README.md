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

Visual cryptography is a cryptographic technique that allows for the encryption of images into multiple share images. When these shares are stacked together, the original image is revealed without the need for any complex computations. This project enhances the traditional visual cryptography approach by integrating a Disentangled Graph Variational Autoencoder to improve the quality of reconstructed images and minimize noise.

## Dataset

The dataset used in this project consists of high-resolution and low-resolution image pairs. The data is organized into `train`, `val`, and `Raw Data` folders.

-   **Raw Data**: Contains the original high-resolution and low-resolution images used to create the training and validation sets.
-   **train**: Images used for training the DVAE model.
-   **val**: Images used for validating the DVAE model during training.

The notebook includes code to load the dataset, visualize the data distribution, and split the data into training, validation, and test sets.

## Model Architecture

The core of the image reconstruction process is a Disentangled Graph Variational Autoencoder (DVAE). The DVAE architecture includes:

-   **Encoder**: Downsampling blocks with residual connections and PReLU activation.
-   **Bottleneck**: Convolutional layers with PReLU activation.
-   **Decoder**: Upsampling blocks with Conv2DTranspose, dropout (optional), concatenation with encoder outputs (skip connections), and CBAM blocks.
-   **CBAM Block**: Convolutional Block Attention Module for enhancing feature representation through channel and spatial attention.

The model is designed to learn a disentangled representation of the image data, which aids in reconstructing high-quality images from low-resolution or noisy inputs.

## Optimization Algorithm

The Superb Fairy-wren Optimization Algorithm (SFOA) is introduced and implemented. While the notebook includes the SFOA class and a convergence plot for a simple objective function, its direct application to optimizing the DVAE model's training process is not explicitly shown in the provided code. The SFOA section appears to be a separate exploration of an optimization algorithm.

## Training

The DVAE model is trained using the AdamW optimizer with a Cosine Decay learning rate schedule. The loss function used is Mean Absolute Error (MAE). The model is compiled with PSNR, SSIM, and MSE as evaluation metrics.

Training utilizes ModelCheckpoint and EarlyStopping callbacks to save the best model based on validation loss.

## Results

After training, the notebook loads the best saved model weights and evaluates its performance on the test set. The `plot_images` function is used to visualize the original high-resolution image, the low-resolution input image, and the image reconstructed by the DVAE. Performance metrics (PSNR, SSIM, and Correlation) are computed and displayed for these visual examples.

## Visual Cryptography Implementation

The notebook includes functions for implementing a basic color visual cryptography scheme:

-   `decompose_channels`: Splits an RGB image into individual color channels.
-   `compute_histogram`: Computes the histogram for a single color channel.
-   `group_pixels`: Groups pixel intensities into a specified number of bins.
-   `optimize_color_level_sfoa`: Applies Otsu's method (referenced as using SFOA in the function name) to find a threshold for binarization.
-   `generate_boolean_shares`: Generates two binary shares from a segmented channel.
-   `combine_color_shares`: Combines the individual channel shares into two color share images.
-   `decrypt_color_shares`: Reconstructs the image by XORing the two color shares.
-   `encrypt_image`: Encrypts a single RGB image into two shares.
-   `generate_color_shares`: A simpler implementation of generating color shares using bitwise XOR.
-   `process_visual_cryptography`: Applies the visual cryptography process to a set of images, including optional blurring and plotting of shares and reconstructed images.

The visual cryptography section demonstrates the process of generating shares and reconstructing the image by XORing the shares.

## Performance Metrics

The notebook calculates and prints the average performance metrics (PSNR, SSIM, Correlation, MSE, RMSE, and MAE) for the DVAE model's predictions on the test set. These metrics provide a quantitative evaluation of the model's ability to reconstruct high-quality images.

## Usage

To run this notebook:

1.  Ensure you have the required libraries installed (tensorflow, keras, numpy, cv2, matplotlib, tqdm, re, skimage).
2.  Place your dataset in the appropriate directory structure as expected by the `base_path` variable.
3.  Run the cells sequentially.

This notebook provides a comprehensive approach to image reconstruction using a DVAE and integrates visual cryptography concepts.
