import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "source"
TRAIN_CSV = DATA / "train.csv"
IMAGE_DIR = DATA / "train" / "train"
MODEL_DIR = ROOT / "results" / "models"
METRIC_DIR = ROOT / "results" / "metrics"

SEED = 42
EPOCHS = 12
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
IMAGE_SIZE = 224
NUM_WORKERS = 4

CLASSES = [
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FruitDataset(Dataset):
    def __init__(self, frame, image_dir, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.image_dir / row["Id"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot read image: {image_path}") from exc

        image = self.transform(image)
        label = CLASS_TO_IDX[row["Label"]]
        return image, label

def main():
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {TRAIN_CSV}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image folder: {IMAGE_DIR}")

    df = pd.read_csv(TRAIN_CSV)
    required = {"Id", "Label"}
    if not required.issubset(df.columns):
        raise ValueError(f"train.csv must contain {required}")

    df["Id"] = df["Id"].astype(str)
    df["Label"] = df["Label"].astype(str)

    unknown = sorted(set(df["Label"]) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")

    missing = [name for name in df["Id"] if not (IMAGE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files missing. First examples: {missing[:5]}"
        )

    print("Class distribution:")
    print(df["Label"].value_counts().reindex(CLASSES, fill_value=0))
    print(f"\nTotal labelled training images: {len(df)}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["Label"],
    )

    weights = ResNet18_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = FruitDataset(train_df, IMAGE_DIR, train_transform)
    val_ds = FruitDataset(val_df, IMAGE_DIR, eval_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
    )

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_accuracy = -1.0
    best_epoch = 0
    history = []
    checkpoint_path = MODEL_DIR / "fruit_resnet18_best.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        train_true, train_pred = [], []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_true.extend(labels.detach().cpu().tolist())
            train_pred.extend(outputs.argmax(1).detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)

        model.eval()
        val_loss_sum = 0.0
        val_true, val_pred = [], []

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(outputs.argmax(1).cpu().tolist())

        val_loss = val_loss_sum / len(val_ds)
        val_acc = accuracy_score(val_true, val_pred)
        scheduler.step(val_acc)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "image_size": IMAGE_SIZE,
                "mean": mean,
                "std": std,
                "best_epoch": best_epoch,
                "validation_accuracy": best_accuracy,
                "seed": SEED,
            }, checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")

    pd.DataFrame(history).to_csv(METRIC_DIR / "training_history.csv", index=False)

    # Final validation report uses the best saved checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_true, val_pred = [], []
    with torch.inference_mode():
        for images, labels in val_loader:
            outputs = model(images.to(device, non_blocking=True))
            val_pred.extend(outputs.argmax(1).cpu().tolist())
            val_true.extend(labels.tolist())

    report = classification_report(
        val_true, val_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    with open(METRIC_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    summary = {
        "dataset": "menasi11/fruit-classification",
        "seed": SEED,
        "epochs_requested": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "training_images": len(train_ds),
        "validation_images": len(val_ds),
        "classes": CLASSES,
        "device": str(device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    with open(METRIC_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== TRAINING COMPLETE =====")
    print(json.dumps(summary, indent=2))
    print("\nValidation report:")
    print(classification_report(
        val_true, val_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, zero_division=0,
    ))

if __name__ == "__main__":
    main()
