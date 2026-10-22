from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "cifar10"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])


def main():
    train_set = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=False,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=False,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {running_loss / len(train_set):.4f}",
            flush=True,
        )

    model.eval()
    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    classes = train_set.classes

    print(f"\nDevice: {DEVICE}")
    print(f"Test images: {len(test_set)}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, target_names=classes, zero_division=0
    ))

    model_path = RESULTS_DIR / "cifar10_resnet18.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "accuracy": accuracy,
    }, model_path)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
