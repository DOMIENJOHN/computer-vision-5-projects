import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "cifar10"
CHECKPOINT = ROOT / "results" / "cifar10_resnet18.pth"
OUTPUT = ROOT / "results" / "checkpoint_evaluation.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    test_set = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform,
    )
    loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    classes = checkpoint["classes"]

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(DEVICE)).argmax(1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "checkpoint": str(CHECKPOINT),
        "test_images": len(test_set),
        "device": str(DEVICE),
        "accuracy": accuracy,
        "classification_report": report,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(test_set)}")
    print(f"Checkpoint accuracy recorded at training: "
          f"{checkpoint.get('accuracy', 'not recorded')}")
    print(f"Re-evaluated test accuracy: {accuracy:.4%}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, target_names=classes, zero_division=0
    ))
    print(f"Saved metrics: {OUTPUT}")


if __name__ == "__main__":
    main()
