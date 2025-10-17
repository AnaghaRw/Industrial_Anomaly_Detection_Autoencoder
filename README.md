# 🔧 Industrial Equipment Anomaly Detection using Deep Autoencoders

This project implements deep learning–based anomaly detection for visual inspection of industrial components (such as screws, metal nuts, and cables).  
The system uses **Convolutional Autoencoders (CAEs)** to learn normal visual patterns and detect anomalies via reconstruction error.

---

## Project Overview

| Section | Description |
|----------|-------------|
| **Objective 1** | Build and train Deep Autoencoder models to detect anomalies with >90% accuracy. |
| **Objective 2** | Visualize reconstruction results and highlight defects using heatmaps. |
| **Dataset** | [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) — categories like *screw*, *metal_nut*, *cable*. |
| **Frameworks** | PyTorch, NumPy, OpenCV, Matplotlib, Seaborn. |
---

## Model Architectures

### 1️. Base Convolutional Autoencoder (BaseCAE)
**Input Layer:** 256×256×3  
**Encoder:**
- Conv2D(32, 3×3) → ReLU  
- Conv2D(64, 3×3) → ReLU  
- Conv2D(128, 3×3) → ReLU  

**Decoder:**
- ConvTranspose2D(128, 3×3) → ReLU  
- ConvTranspose2D(64, 3×3) → ReLU  
- ConvTranspose2D(32, 3×3) → ReLU  
- ConvTranspose2D(3, 3×3) → Sigmoid  

---

### 2️. Deep Convolutional Autoencoder (DeepCAE)
**Input Layer:** 256×256×3  
**Encoder:**
- Conv2D(32, 3×3) → ReLU  
- Conv2D(64, 3×3) → ReLU  
- Conv2D(128, 3×3) → ReLU  
- Conv2D(256, 3×3) → ReLU + Dropout(0.2)

**Decoder:**
- ConvTranspose2D(128, 3×3) → ReLU  
- ConvTranspose2D(64, 3×3) → ReLU  
- ConvTranspose2D(32, 3×3) → ReLU  
- ConvTranspose2D(3, 3×3) → Sigmoid  

---

### 3️. Denoising Convolutional Autoencoder (DenoisingCAE)
**Input Layer:** 256×256×3 + Gaussian Noise (σ = 0.05)  
**Encoder:**
- Conv2D(32, 3×3) → ReLU  
- Conv2D(64, 3×3) → ReLU  
- Conv2D(128, 3×3) → ReLU  

**Decoder:**
- ConvTranspose2D(128, 3×3) → ReLU  
- ConvTranspose2D(64, 3×3) → ReLU  
- ConvTranspose2D(32, 3×3) → ReLU  
- ConvTranspose2D(3, 3×3) → Sigmoid  

---

## Training & Hyperparameter Tuning

| **Hyperparameter** | **Tried Values** | **Best Value** |
|--------------------|------------------|----------------|
| Learning Rate | 0.001, 0.0001 | 0.001 |
| Batch Size | 16, 32 | 32 |
| Epochs | 10, 20 | 20 |
| Optimizer | Adam, RMSProp | Adam |
| Dropout Rate | 0.3, 0.5 | 0.3 |
| Models Tested | BaseCAE, DeepCAE, DenoisingCAE | **DenoisingCAE** |

**Best Reconstruction Loss:** 0.00136  
**Training Time:** ~1116 seconds (CPU)

---

## Visualization & Analysis

- **Loss Comparison:** Bar plots of reconstruction loss across models and optimizers.  
- **Learning Rate vs Loss:** Scatter plot showing LR impact on stability.  
- **Training Time Comparison:** Efficiency trade-off visualization.  
- **Anomaly Heatmaps:** Visual overlays highlighting defect regions.

Example:
```python
visualize(model, "data/mvtec_ad/screw/test/scratch_neck/010.png")
