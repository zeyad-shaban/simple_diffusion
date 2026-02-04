# 🎨 MNIST Diffusion: From Math to U-Net

This repository is a step-by-step implementation of Denoising Diffusion Probabilistic Models (DDPM). The goal is to understand the underlying Gaussian math, the forward noise process, and the neural architecture required to generate images from pure noise.

## 📖 Primary Reference
This project follows the logic and parameters established in the seminal paper:
* **Denoising Diffusion Probabilistic Models (DDPM)** by Ho et al. (2020).

---

## 🚀 Project Roadmap

The implementation is broken down into three logical stages:

### 1. In-Depth Inspection (`1_indepth_inspect.ipynb`)
This notebook focuses on the **Forward Diffusion Process**. 
* **Mathematical Proof:** Verifies that jumping directly to any timestep $t$ using the $\alpha$ cumulative product is equivalent to the iterative looping approach of adding noise.
* **Variance Preservation:** Demonstrates how the $\sqrt{\alpha}$ scaling factor prevents variance from exploding, keeping the pixel values within a stable range ($std \approx 1$).
* **Noise Prediction:** Introduces the concept of training the model to predict the **noise ($\epsilon$)** added to an image rather than the image itself, and the derivation to recover $x_0$ from that prediction.

### 2. Proof of Concept: Custom MLP (`2_custom_mlp.ipynb`)
A transition from math to code using a simplified architecture.
* **Simpler Model:** Uses a basic Linear MLP to test if the "predict noise" logic works on flattened MNIST digits.
* **DDPM Sampling:** Implements the formal reverse-diffusion sampling formula to denoise images step-by-step, moving from pure Gaussian noise back to the data manifold.

### 3. The Full U-Net (`3_unet.ipynb`)
The final, production-grade architecture used in the DDPM paper.
* **U-Net Architecture:** Implements a convolutional encoder-decoder with skip connections to preserve spatial details.
* **Sinusoidal Time Embeddings:** Implements the "Clock of Diffusion"—a frequency-based embedding that allows the model to know the current timestep $t$ at every layer.
* **Context Injection:** Shows how to project and broadcast time embeddings into Downsample, Upsample, and Bottleneck blocks.

---

## 🛠️ Key Concepts Explored
* **Forward Diffusion:** Adding Gaussian noise via a linear beta schedule.
* **Reverse Diffusion:** Learning to reverse the Markov Chain to generate data.
* **Broadcasting:** Managing 4D tensor shapes $(B, C, H, W)$ to inject global time context into local spatial features.
* **Numerical Stability:** Using log-space calculations for sinusoidal frequencies to prevent overflow/underflow.

---

## 📋 Requirements
* PyTorch
* Torchvision
* Matplotlib
* TQDM (for progress tracking)