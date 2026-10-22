# Explainable AI for Object Recognition

## 1. Problem Statement
Deep-learning classifiers recognise objects, but their decisions can be difficult to interpret. This project combines image classification with Grad-CAM visualisation to inspect image regions that influence a prediction.

## 2. Objectives
- Train an image classifier on CIFAR-10.
- Evaluate it on a separate test set.
- Measure accuracy, precision, recall and F1-score.
- Generate Grad-CAM heatmaps to explain predictions.

## 3. Dataset
CIFAR-10 contains 60,000 colour images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

- Training images: 50,000
- Test images: 10,000
- Official source: https://www.cs.toronto.edu/~kriz/cifar.html

## 4. Methodology
1. Resize images to 64 × 64 pixels.
2. Convert images to tensors and normalise channels.
3. Apply random horizontal flipping during training.
4. Train ResNet-18 with a 10-class output layer for five epochs.
5. Evaluate on the 10,000-image test set.
6. Generate a Grad-CAM heatmap from the trained model.

Training configuration: batch size 16, Adam optimiser, learning rate 0.001, cross-entropy loss.

## 5. Tools and Libraries
- Python
- PyTorch and Torchvision
- NumPy
- Matplotlib
- scikit-learn
- Pillow

## 6. Results

| Metric | Measured result |
|---|---:|
| Test accuracy | 83.42% |
| Macro precision | 0.84 |
| Macro recall | 0.83 |
| Macro F1-score | 0.83 |
| Weighted precision | 0.84 |
| Weighted recall | 0.83 |
| Weighted F1-score | 0.83 |

Training loss decreased from 1.3903 in epoch 1 to 0.4417 in epoch 5.

### Class-level observations
- Automobile F1-score: 0.91
- Ship F1-score: 0.92
- Dog recall: 0.62
- Cat recall: 0.68

### Grad-CAM demonstration
The first test image was a cat. The model predicted cat with 98.22% confidence. The visualisation is saved at `results/visualizations/cifar10_gradcam.png`.

An earlier ImageNet demo predicted Samoyed for a dog image with 83.74% confidence. This is a separate experiment and is not included in the CIFAR-10 metrics.

## 7. Limitations
- Training used only five epochs.
- Grad-CAM is a visual aid, not proof that the model reasons correctly.
- The explanation demonstration covers one test image.
- CIFAR-10's small images do not represent every real-world condition.

## 8. Reproduction
Download and extract CIFAR-10 into `dataset/cifar10/`, then run these commands from the repository root:

```bash
python 01_explainable_ai/src/train_evaluate.py
python 01_explainable_ai/src/explain_cifar10.py
```

## 9. Project Files
- `src/main.py`: separate ImageNet classification and Grad-CAM demo.
- `src/train_evaluate.py`: CIFAR-10 training and evaluation.
- `src/explain_cifar10.py`: Grad-CAM for the trained CIFAR-10 model.
- `dataset/dataset_info.md`: dataset details.
- `report.md`: technical report.
- `results/visualizations/`: output images.