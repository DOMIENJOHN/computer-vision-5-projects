import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results" / "models" / "fruit_resnet18_best.pth"

def main():
    parser = argparse.ArgumentParser(
        description="Classify a fruit image by type and freshness."
    )
    parser.add_argument("image", help="Path to a fruit image")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    size = checkpoint["image_size"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])

    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, index = probabilities.max(dim=0)

    label = classes[index.item()]
    freshness, fruit = label.split("_", 1)

    print("\n===== FRUIT QUALITY PREDICTION =====")
    print(f"Image      : {image_path.name}")
    print(f"Fruit      : {fruit.title()}")
    print(f"Condition  : {freshness.title()}")
    print(f"Class      : {label}")
    print(f"Confidence : {confidence.item() * 100:.2f}%")
    print(f"Device     : {device}")
    print("\nAll class probabilities:")
    for i in probabilities.argsort(descending=True).tolist():
        print(f"  {classes[i]:15s} {probabilities[i].item() * 100:6.2f}%")

    print("\nNote: confidence is not a guarantee of correctness.")

if __name__ == "__main__":
    main()
