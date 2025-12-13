# 🥑 AguacAIte-Labs

AguacAIte is a machine learning and deep learning framework designed to detect Tuberculosis (TB) form chest X-ray images.

The project follorws a full medical-AI pipeline-from dataset exploration and cleaning, to prepocessing, classical ML baselines, CNNs, tranfer learning modesl, evaluation metrics, interpretability using Grad-CAM, and final reporting.

---
## Technologies Used
 - Python
 - PyTorch
 - scikit-learn
 - OpenCV
 - NumPy / Matplotlib
 - Grad-CAM
 
---

## Project Structure
```
📁 AguacAIte-Labs
 ├── 📂 data
 │    ├── 📁 raw
 │    │     ├── 📁 normal/
 │    │     └── 📁 tuberculosis/
 │    └── 📁 processed
 ├── 📂 notebooks
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
- K-Nearest Neighbors (KNN) - Simple distance-based classifier using HOG features
- Decision Tree -  Basic interpretable model for comparison
- HOG + SVM - Strong baseline using handcrafted features

### Deep Learning
- Custom CNN - Lightweight convolutional architectural build form scratch
- DenseNet121 (Transfer Learning) - Hight accuracu & best performance for medical images.
- ResNet50 (Transfer Learning) -  Strong, stable architecure
- EfficientNet-B0 (Transfer Learning) - Balanced accurancy vs efficiency.

---

## Datasets
- Shenzhen Hospital Chest X-ray Set (TB vs Normal)

---

## Outputs
- Confusion matrix
- Classification metrics
- ROC curve
- Model weights
- Training curves
- Grad-CAM lung heatmaps
---

## Goal
Provide a full end-to-end AI pipeline for chest X-ray TB detection

Compare classical ML vs CNN vs transfer learning

Demonstrate interpretability for medical safety

Serve as a reproducible academic and scientific framework
---
