# Detecting Hand Gestures for Gaming Using Adaptive AI

This project investigates the effectiveness of deep learning models—LSTM, Vision Transformer, CNN-LSTM, and CNN-Transformer—for accurate and real-time hand gesture recognition in gaming. By leveraging the HaGRID dataset and hybrid architectures, the system aims to provide an adaptive, low-latency solution for gesture-controlled gaming.

---

## Table of Contents
- [Abstract](#abstract)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architectures](#model-architectures)
- [Training & Evaluation](#training--evaluation)
- [Results](#results-summary)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

---

## Abstract

Gesture-based interaction is redefining modern gaming by allowing more immersive and intuitive control mechanisms. This study presents a comparative analysis of four deep learning models—LSTM, Vision Transformer, CNN-LSTM, and CNN-Transformer—using the HaGRID dataset. It evaluates the trade-offs in accuracy, latency, and adaptability, with the hybrid CNN-Transformer achieving the highest performance for real-time gaming.

---

## 🧠 Hand Gesture Recognition System Architecture

This diagram illustrates the overall architecture of the deep learning pipeline, including preprocessing, spatial feature extraction, temporal modeling, and gesture classification.

![Hand Gesture Recognition Architecture](images/architecture.png)

---

## Dataset

- **Primary Dataset:** HaGRID (Hand Gesture Recognition Image Dataset)
- **Additional References:** 20BN-Jester, EgoGesture
- **Content:** Thousands of gesture sequences across lighting conditions, user variations, and gesture types.
- **Format:** Sequences of RGB frames, preprocessed for landmark-based gesture representation.

---

## Preprocessing

- Resizing, normalization, and denoising
- Hand landmark extraction via **OpenCV** and **MediaPipe**
- Data augmentation: rotation, flip, Gaussian noise
- Label encoding and temporal segmentation

---

## Model Architectures

### LSTM
- Captures temporal dependencies in gesture sequences
- Strong at modeling motion over time

### Vision Transformer (ViT)
- Utilizes self-attention for spatial feature extraction
- Best at global feature representation

### CNN-LSTM Hybrid
- CNN: Spatial features → LSTM: Temporal patterns
- Good trade-off between speed and accuracy

### CNN-Transformer Hybrid *(Best Performer)*
- Combines CNN for feature maps with Transformer for context
- Achieved the highest overall performance across metrics

---

## Training & Evaluation

- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Techniques:** Early stopping, data augmentation, mini-batch training

---

## Results Summary

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| LSTM               | 95.20%   | 95.75%    | 95.20% | 95.50%   |
| Vision Transformer | 93.20%   | 93.43%    | 93.20% | 92.90%   |
| CNN-LSTM Hybrid    | 95.66%   | 95.99%    | 95.66% | 95.50%   |
| **CNN-Transformer**| **96.34%**| **96.60%**| **96.34%**| **96.24%** |

*CNN-Transformer* outperforms all others in every category, with robust real-time performance.

---

## Conclusion

- **Hybrid models** (CNN-LSTM, CNN-Transformer) significantly improve classification accuracy and response time
- **Transformer-based models** enhance spatial understanding, especially when fused with CNNs
- Future improvements: better regularization, class-weight tuning, and deployment on edge devices

---

## How to Run

```bash
# Clone the repository
gh repo clone C00LOO5/Study-Hand-Gesture-Recognition-using-AI-models
cd Study-Hand-Gesture-Recognition-using-AI-models

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run model training (example)
python training_eval.py #project files contain .ipybn file

# For real-time webcam testing
python demo.py  #included in project files
