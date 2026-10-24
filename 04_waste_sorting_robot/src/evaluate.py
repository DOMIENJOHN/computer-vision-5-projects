
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "splits_clean.csv"
CHECKPOINT = ROOT / "models" / "best_model.pth"
RESULTS = ROOT / "results"

CLASSES = ["biodegradable", "non_biodegradable"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4


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

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = CLASSES.index(row["label"])
        return image, label, str(path)


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    df = pd.read_csv(MANIFEST)
    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("The test split contains no images.")

    # Confirm that only the held-out test split is evaluated.
    assert (test_df["split"] == "test").all()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = WasteDataset(test_df, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=True
    )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(CLASSES)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred, paths = [], [], []

    with torch.inference_mode():
        for images, labels, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions)
            paths.extend(batch_paths)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    RESULTS.mkdir(parents=True, exist_ok=True)

    (RESULTS / "classification_report.txt").write_text(
        f"Checkpoint: {CHECKPOINT.name}\n"
        f"Best epoch: {checkpoint.get('best_epoch', 'unknown')}\n"
        f"Validation macro-F1 recorded in checkpoint: "
        f"{checkpoint.get('best_val_macro_f1', 'unknown')}\n"
        f"Test images: {len(y_true)}\n"
        f"Test accuracy: {accuracy:.4f}\n"
        f"Test macro-F1: {macro_f1:.4f}\n\n"
        f"{report}\n",
        encoding="utf-8",
    )

    pd.DataFrame({
        "path": paths,
        "true_label": [CLASSES[i] for i in y_true],
        "predicted_label": [CLASSES[i] for i in y_pred],
    }).to_csv(RESULTS / "test_predictions.csv", index=False)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Waste Classification — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('best_epoch', 'unknown')}")
    print(f"Test images: {len(y_true)}")
    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    print("\nClassification report:\n")
    print(report)
    print(f"Report: {RESULTS / 'classification_report.txt'}")
    print(f"Confusion matrix: {RESULTS / 'confusion_matrix.png'}")
    print(f"Predictions: {RESULTS / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
