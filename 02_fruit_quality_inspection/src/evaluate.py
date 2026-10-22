import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "source"
IMAGE_DIR = DATA / "test" / "test"
CSV_PATH = DATA / "sample_submission.csv"
CHECKPOINT = ROOT / "results" / "models" / "fruit_resnet18_best.pth"
OUT = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
CLASSES = checkpoint["classes"]
SIZE = checkpoint["image_size"]

transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
])

class FruitImages(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        path = IMAGE_DIR / name
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        return tensor, name

def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Test image folder not found: {IMAGE_DIR}")
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    csv = pd.read_csv(CSV_PATH)
    if not {"Id", "Label"}.issubset(csv.columns):
        raise ValueError("Expected Id and Label columns in sample_submission.csv")

    filenames = csv["Id"].astype(str).tolist()
    missing = [name for name in filenames if not (IMAGE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing test images, e.g. {missing[:5]}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    loader = DataLoader(
        FruitImages(filenames),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    records = []
    with torch.inference_mode():
        for images, names in tqdm(loader, desc="Predicting"):
            probabilities = torch.softmax(
                model(images.to(DEVICE)), dim=1
            ).cpu()
            confidence, indices = probabilities.max(dim=1)

            for name, conf, idx, probs in zip(
                names, confidence.tolist(), indices.tolist(), probabilities.tolist()
            ):
                row = {
                    "Id": name,
                    "Predicted_Label": CLASSES[idx],
                    "Confidence": conf,
                }
                row.update({
                    f"prob_{label}": float(prob)
                    for label, prob in zip(CLASSES, probs)
                })
                records.append(row)

    predictions = pd.DataFrame(records)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    print(f"\nSaved {len(predictions)} predictions to results/test_predictions.csv")
    print(predictions[["Id", "Predicted_Label", "Confidence"]].head(10).to_string(index=False))

    # Labels in sample_submission.csv are not yet verified as ground truth.
    # Any comparison is provisional and must not be presented as official test performance.
    labels = csv["Label"].astype(str)
    if labels.isin(CLASSES).all():
        merged = csv[["Id", "Label"]].merge(
            predictions[["Id", "Predicted_Label"]],
            on="Id",
            validate="one_to_one",
        )
        y_true = merged["Label"].tolist()
        y_pred = merged["Predicted_Label"].tolist()

        metrics = {
            "status": "PROVISIONAL: sample_submission.csv labels not independently verified",
            "images_compared": len(merged),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "classification_report": classification_report(
                y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
            ),
        }
        with open(OUT / "metrics" / "test_comparison_provisional.json", "w") as file:
            json.dump(metrics, file, indent=2)

        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
        )
        plt.xlabel("CSV label (unverified)")
        plt.ylabel("Predicted label")
        plt.title("Provisional Test Comparison")
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            OUT / "visualizations" / "test_confusion_matrix_provisional.png",
            dpi=180,
        )
        plt.close()

        print("\nPROVISIONAL comparison only — CSV labels are unverified.")
        print(classification_report(
            y_true, y_pred, labels=CLASSES, zero_division=0
        ))
    else:
        print("CSV labels are not valid class labels; skipping metric comparison.")

    print("\nDevice:", DEVICE)

if __name__ == "__main__":
    main()
