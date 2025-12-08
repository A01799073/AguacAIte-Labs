# 🥑 AguacAIte-Labs

AguacAIte is a machine learning and deep learning framework for detecting tuberculosis in chest X-ray images.  
The project compares classical ML baselines with modern convolutional neural networks and transfer learning architectures, providing both quantitative metrics and visual interpretability through Grad-CAM.

---

## Project Structure
```
📁 AguacAIte-Labs
 ├── 📂 data
 |    ├── 📂 notebook
 │    ├── 📁 raw
 │    └── 📁 processed
 ├── 📂 src
 │    ├── 📁 data
 │    ├── 📁 models
 │    ├── 📁 training
 │    ├── 📁 evaluation
 │    └── 📁 visualization
 ├── 📂 models
 ├── 📂 results
 └── 📄 README.md
```
----

---
## Pipeline Overview
`DATASET → CLEANING → PREPROCESSING → FEATURE EXTRACTION → BASELINE CLASSIFIERS → CNN → TRANSFER LEARNING → METRICS → GRADCAM → REPORT`

## Models Implemented

### Classical ML Baselines
- K-Nearest Neighbors (KNN)
- Decision Tree
- HOG + SVM

### Deep Learning
- Custom CNN
- DenseNet121 (Transfer Learning)
- ResNet50 (Transfer Learning)
- EfficientNet-B0 (Transfer Learning)

---

## Datasets
- Shenzhen Hospital Chest X-ray Set (TB vs Normal)

---

## Goal
A clean, reproducible pipeline for TB detection using classic ML and modern deep learning.

---
