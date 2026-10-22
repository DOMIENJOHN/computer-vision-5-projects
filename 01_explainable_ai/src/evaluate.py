import argparse
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ResNet18_Weights.DEFAULT


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained ResNet-18 on labelled images"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset directory with one subfolder per class",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset = ImageFolder(args.data, transform=WEIGHTS.transforms())
    if len(dataset) == 0:
        raise ValueError("The dataset contains no images.")

    model = resnet18(weights=WEIGHTS).to(DEVICE).eval()
    class_names = WEIGHTS.meta["categories"]
    y_true, y_pred = [], []

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            predictions = model(images).argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(
                class_names[label] for label in labels.tolist()
            )

    predicted_names = [class_names[index] for index in y_pred]
    accuracy = accuracy_score(y_true, predicted_names)

    print(f"Device: {DEVICE}")
    print(f"Images evaluated: {len(dataset)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predicted_names,
            labels=dataset.classes,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
