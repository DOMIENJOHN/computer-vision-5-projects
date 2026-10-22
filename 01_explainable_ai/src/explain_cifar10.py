from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "cifar10"
MODEL_PATH = ROOT / "results" / "cifar10_resnet18.pth"
OUTPUT_DIR = ROOT / "results" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    ),
])


def main():
    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
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

    dataset = datasets.CIFAR10(
        root=str(DATA_DIR), train=False,
        download=False, transform=None
    )
    image, true_label = dataset[0]
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    activations = []
    gradients = []
    target_layer = model.layer4[-1].conv2

    h1 = target_layer.register_forward_hook(
        lambda module, inputs, output: activations.append(output)
    )
    h2 = target_layer.register_full_backward_hook(
        lambda module, grad_input, grad_output:
        gradients.append(grad_output[0])
    )

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        logits[0, prediction.item()].backward()

        feature_maps = activations[0][0].detach()
        grads = gradients[0][0].detach()
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * feature_maps).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam[None, None], size=(256, 256),
            mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        original = np.asarray(image.resize((256, 256)))
        heatmap = plt.get_cmap("jet")(cam)[..., :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = np.clip(
            0.55 * original + 0.45 * heatmap, 0, 255
        ).astype(np.uint8)

        output_path = OUTPUT_DIR / "cifar10_gradcam.png"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(original)
        axes[0].set_title("Test image")
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Explanation overlay")

        fig.suptitle(
            f"True: {classes[true_label]} | "
            f"Predicted: {classes[prediction.item()]} "
            f"({confidence.item():.1%} confidence)"
        )
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"True class: {classes[true_label]}")
        print(f"Predicted class: {classes[prediction.item()]}")
        print(f"Confidence: {confidence.item():.4f}")
        print(f"Saved Grad-CAM: {output_path}")
    finally:
        h1.remove()
        h2.remove()


if __name__ == "__main__":
    main()
