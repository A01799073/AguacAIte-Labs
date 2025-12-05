## AguacAIte-Labs
AguacAIte is an AI-powered diagnostic tool designed to detect early signs of tuberculosis in chest X-rays. Using deep learning and advanced image recognition, it differentiates healthy lungs from TB-affected ones with high accuracy and interpretable visual outputs.

Algoritms models
K-Nearest Neighbors (KNN)
HOG + SVM

## Project Structure
AguacAIte-Labs/
│
├── data/
│ ├── processed/ # Cleaned,normalized, and resized images
│ ├── raw/ # Original datasets
├──READ.md ---Links for used for the datasets
│
├── models/ # Trained model weights
│
├── notebooks/ # Jupyter notebooks for experimentation
│
├── results/ 
|
├── src/ # Production-ready Python code
│ ├── data/
│ ├── evaluation/
│ ├── models/
│ ├── training/
│ └── visualization/
│
└── README.md
