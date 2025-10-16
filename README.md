
# 🌄 Outdoor Scene Generation using DCGAN

**Project:** Gyaan Uday Hackathon-2024, Central Research Laboratory, BEL  
**Duration:** Feb 2024 – Jun 2024  
**Team Size:** 2  

## 🧠 Overview
This project focuses on generating **synthetic outdoor scenes** to support **surveillance robot training**.  
We trained a **Deep Convolutional GAN (DCGAN)** on **60K+ outdoor images** from the [Intel Image Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification).  

The model learns to reproduce realistic textures and structures of natural environments—buildings, forests, seas, and mountains—helping surveillance robots generalize better to unseen outdoor conditions.  

💡 **Key achievement:**  
We **reduced training instability by 27%** through **adaptive learning-rate schedulers** and **hyperparameter grid search**, leading to smoother generator–discriminator convergence and more stable sample quality.

---

## ⚙️ Technical Details
- Framework: **PyTorch**
- Environment: **Google Colab**
- Architecture: **DCGAN**
- Dataset: **Intel Image Classification Dataset**
- Image size: 64×64 RGB
- **Stability optimization:**
  - Adaptive LR schedulers (ReduceLROnPlateau & CosineAnnealing)
  - Grid search on optimizers (Adam/AdamW) and batch sizes
  - Label smoothing for discriminator
  - Gradient clipping to avoid mode collapse
- Training stability improved by **27%**, measured by variance reduction in discriminator loss and FID consistency.

---

## 📊 Results
Generated images closely mimic natural outdoor distributions.  
Sample outputs are in `outputs/generated_samples/`.

| Real | Generated |
|------|------------|
| ![real](outputs/sample_real.jpg) | ![fake](outputs/sample_fake.jpg) |

---

## 🧩 Repository Structure
