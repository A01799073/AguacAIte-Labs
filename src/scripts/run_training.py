import torch
import pickle
from pathlib import Path

from src.models.cnn import SimpleCNN
from src.data.loaders import build_dataloaders
from src.training.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = build_dataloaders("data/processed")

model = SimpleCNN().to(device)

history = train_model(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    device = device,
    epochs = 10
)

# Save history for evaluation
with open("models/training_history.pkl", "wb") as f:
    pickle.dump(history, f)
