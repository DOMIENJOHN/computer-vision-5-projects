from pathlib import Path
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 4
NUM_WORKERS = 4
LEARNING_RATE = 1e-4

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "splits_clean.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CHECKPOINT = MODEL_DIR / "best_model.pth"

CLASSES = ["biodegradable", "non_biodegradable"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WasteDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path

        # Corrupt images were excluded from the cleaned manifest.
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        label = CLASS_TO_IDX[row["label"]]
        return tensor, label


def make_loader(dataset, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    targets_all, predictions_all = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * labels.size(0)
            predictions_all.extend(logits.argmax(1).detach().cpu().tolist())
            targets_all.extend(labels.detach().cpu().tolist())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(targets_all, predictions_all)
    macro_f1 = f1_score(
        targets_all, predictions_all, average="macro", zero_division=0
    )

    return mean_loss, accuracy, macro_f1


def main():
    seed_everything(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.is_file():
        raise SystemExit(
            f"Clean manifest not found: {MANIFEST}\n"
            "Run: python src/clean_manifest.py"
        )

    df = pd.read_csv(MANIFEST)

    required_columns = {"sha256", "split", "label", "path"}
    if not required_columns.issubset(df.columns):
        raise SystemExit(f"Manifest must contain columns: {required_columns}")

    if df["sha256"].duplicated().any():
        raise SystemExit("Duplicate image hashes detected in manifest.")

    if not set(df["split"]) >= {"train", "val", "test"}:
        raise SystemExit("Manifest must contain train, val and test splits.")

    if not set(df["label"]).issubset(set(CLASSES)):
        raise SystemExit("Unexpected class labels found in manifest.")

    # Confirm that no exact hash crosses between splits.
    split_count = df.groupby("sha256")["split"].nunique()
    if (split_count > 1).any():
        raise SystemExit("Data leakage detected: hashes cross dataset splits.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise SystemExit("Training or validation split is empty.")

    # Train-time augmentation; validation uses deterministic preprocessing.
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.12
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = WasteDataset(train_df, train_transform)
    val_dataset = WasteDataset(val_df, eval_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Manifest: {MANIFEST}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print("Training class counts:")
    print(train_df["label"].value_counts().to_string())

    train_loader = make_loader(train_dataset, True, device)
    val_loader = make_loader(val_dataset, False, device)

    print("Loading pretrained MobileNetV3-Small...")
    model = models.mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT
    )
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(CLASSES)
    )
    model = model.to(device)

    counts = train_df["label"].value_counts()
    weights = [
        len(train_df) / (len(CLASSES) * counts.get(name, 1))
        for name in CLASSES
    ]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("Class weights:", {
        name: round(float(class_weights[i]), 3)
        for i, name in enumerate(CLASSES)
    })

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            RESULTS_DIR / "training_history.csv", index=False
        )

        print(
            f"Train loss={train_loss:.4f} | Train acc={train_acc:.4f} "
            f"| Train macro-F1={train_f1:.4f}"
        )
        print(
            f"Val loss={val_loss:.4f} | Val acc={val_acc:.4f} "
            f"| Val macro-F1={val_f1:.4f} | LR={current_lr:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_count = 0

            torch.save({
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "class_to_idx": CLASS_TO_IDX,
                "image_size": IMAGE_SIZE,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "architecture": "mobilenet_v3_small",
                "mean": MEAN,
                "std": STD,
            }, CHECKPOINT)
            print(f"Saved best checkpoint: {CHECKPOINT}")
        else:
            patience_count += 1
            print(f"No improvement: {patience_count}/{PATIENCE}")

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    history_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(
        history_df["epoch"], history_df["train_accuracy"], label="Train accuracy"
    )
    axes[1].plot(
        history_df["epoch"], history_df["val_accuracy"], label="Val accuracy"
    )
    axes[1].plot(
        history_df["epoch"], history_df["val_macro_f1"], label="Val macro-F1"
    )
    axes[1].set_title("Training metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_history.png", dpi=160)
    plt.close(fig)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Best checkpoint: {CHECKPOINT}")
    print("The held-out test split has not been evaluated.")


if __name__ == "__main__":
    main()
