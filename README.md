# 🏞️ DCGAN for Outdoor Scene Synthesis

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** in **PyTorch** to generate synthetic outdoor scenes.  
It leverages **data augmentation**, **adaptive learning rate schedulers**, and **hyperparameter grid search** to enhance training stability and image quality.

---

## 📘 Table of Contents
- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Data Preparation](#-data-preparation)
- [Training Setup](#-training-setup)
- [Results](#-results)
- [Visualization](#-visualization)
- [Dependencies](#-dependencies)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🚀 Overview

| Feature | Description |
|----------|--------------|
| **Framework** | PyTorch |
| **Dataset** | Intel Image Classification (Outdoor Scenes) |
| **Model** | DCGAN (Conv / ConvTranspose layers) |
| **Image Size** | 64×64 |
| **Batch Size** | 128 |
| **Training Images** | 60K+ (with augmentation) |
| **Improvement** | ~27% reduction in training instability |

---

## 🧠 Key Contributions

- **Built and trained a DCGAN** on an **augmented dataset of 60K+ outdoor images** for synthetic data generation.  
- **Applied data augmentation** via `torchvision.transforms` — random flips, rotations, color jitter, and affine transformations — to diversify the dataset.  
- **Integrated adaptive schedulers (`ReduceLROnPlateau`)** to automatically adjust learning rates during plateaus.  
- **Implemented hyperparameter grid search** to find optimal learning rates and Adam betas:
lrG = 0.0001, lrD = 0.0002, betas = (0.4, 0.9)

- **Achieved a 27% reduction** in training instability (based on loss variance) compared to default parameters.

---

## 🧩 Data Preparation
Unzip and organize the dataset:

 ```` ``` ````  
unzip -qo intel-image-classification.zip -d /content/


mkdir -p /content/data

mv /content/seg_train/seg_train /content/data/
 ```` ``` ````  


## 🖼️ Augmentation Pipeline

 ```` ``` ````  
from torchvision import transforms

image_size = 64

transform = transforms.Compose([

    transforms.Resize((image_size, image_size)),
    
    transforms.RandomHorizontalFlip(p=0.5),
    
    transforms.RandomRotation(10),
    
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    
    transforms.ToTensor(),
    
    transforms.Normalize([0.5]*3, [0.5]*3)
])
 ```` ``` ````  

## ⚙️ Training Setup

Run the training script:


 ```` ``` ````  

python train_dcgan.py
 ```` ``` ````  


This script:

Trains the DCGAN on the augmented dataset.

Tunes learning rates and beta values via grid search.

Applies adaptive schedulers to reduce instability.

## 📊 Results
   
Metric: Instability Score         
Baseline: 0.0060
Optimized: 0.00437
Improvement:    ↓ 27.1%
✅ The optimized configuration produced smoother convergence, reduced discriminator oscillations, and generated sharper, more consistent images.




