import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "visualizations"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ResNet18_Weights.DEFAULT
CLASSES = WEIGHTS.meta["categories"]

TRANSFORM = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def explain(image_path):
    model = resnet18(weights=WEIGHTS).to(DEVICE).eval()
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    activations = []
    gradients = []

    def save_activations(module, inputs, output):
        activations.append(output)

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1].conv2
    forward_handle = target_layer.register_forward_hook(save_activations)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)

    try:
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)

        model.zero_grad(set_to_none=True)
        logits[0, predicted.item()].backward()

        feature_maps = activations[0].detach()[0]
        grads = gradients[0].detach()[0]
        weights = grads.mean(dim=(1, 2))

        cam = (weights[:, None, None] * feature_maps).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam[None, None],
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        original = np.asarray(image)
        heatmap = plt.get_cmap("jet")(cam)[..., :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = np.clip(
            0.55 * original + 0.45 * heatmap, 0, 255
        ).astype(np.uint8)

        output_path = OUTPUT / f"{Path(image_path).stem}_gradcam.png"

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(original)
        axes[0].set_title("Input image")
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Explanation overlay")

        for axis in axes:
            axis.axis("off")

        fig.suptitle(
            f"Prediction: {CLASSES[predicted.item()]} "
            f"({confidence.item():.1%} model confidence)"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Device: {DEVICE}")
        print(f"Predicted class: {CLASSES[predicted.item()]}")
        print(f"Model confidence: {confidence.item():.4f}")
        print(f"Saved visualisation: {output_path}")

    finally:
        forward_handle.remove()
        backward_handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image classification with Grad-CAM explanations"
    )
    parser.add_argument("image", help="Path to an input image")
    args = parser.parse_args()
    explain(args.image)
