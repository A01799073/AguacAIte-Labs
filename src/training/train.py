import torch
import torch.nn as nn
from pathlib import Path

def train_model(model,train_loader,val_loader, device,epochs = 10,lr = 1e-3,save_path = "models/best_cnn.pth"):
    """
        Tarain CNN model with BCE loss and accuracy tracking.
        Saves the best model based on validation accuracy.
    """

    # Loss function for binary classification (logits output)
    criterion = nn.BCEWithLogitsLoss()

    # Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    best_val_acc = 0.0

    save_path = Path(save_path)
    save_path.parent.mkdir(parents = True, exist_ok = True)

    # Store metrics for later analysis (used in 4.4)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0 

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5

                val_correct += (preds == labels.bool()).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Logging
        print(
            f"Epoch [{epoch +1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} |" 
            f"Train Acc: {train_acc:.4f} |"
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Model saved")
        
    print("Training finished")
    return history