import torch
import pickle
from pathlib import Path

from src.data.loaders import build_dataloaders
from src.training.train import train_model
from src.training.utils import unfreeze_last_layers

from src.models.densenet import build_densenet121
from src.models.resnet import build_resnet50
from src.models.efficientnet import build_efficientnet_b0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/processed"
OUT_DIR = Path("models/transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS_HEAD = 5
EPOCHS_FINETUNE = 5
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4


def train_model_pipeline(name, model):
    print(f"\n=== Training {name} ===")

    train_loader, val_loader = build_dataloaders(DATA_DIR)

    model = model.to(DEVICE)

    # ---- Phase A: head only ----
    history_head = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=EPOCHS_HEAD,
        lr=LR_HEAD,
        save_path=OUT_DIR / f"{name}_head.pth"
    )

    # ---- Phase B: partial fine-tuning ----
    unfreeze_last_layers(model, n_layers=20)

    history_finetune = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=EPOCHS_FINETUNE,
        lr=LR_FINETUNE,
        save_path=OUT_DIR / f"{name}_finetuned.pth"
    )

    with open(OUT_DIR / f"{name}_history.pkl", "wb") as f:
        pickle.dump(
            {
                "head": history_head,
                "finetune": history_finetune
            },
            f
        )


if __name__ == "__main__":

    train_model_pipeline(
        "densenet121",
        build_densenet121(pretrained=True, freeze_backbone=True)
    )

    train_model_pipeline(
        "resnet50",
        build_resnet50(pretrained=True, freeze_backbone=True)
    )

    train_model_pipeline(
        "efficientnet_b0",
        build_efficientnet_b0(pretrained=True, freeze_backbone=True)
    )
