# Project Report: Explainable AI for Object Recognition

## 1. Abstract
This project combines image classification using ResNet-18 with Grad-CAM visualisation. The model was trained on CIFAR-10 and evaluated on a separate test set. Grad-CAM was used to visualise image regions associated with a selected prediction.

## 2. Problem Statement
Deep-learning image classifiers can produce predictions that are difficult to interpret. Visual explanation methods help users inspect which image regions influenced a prediction.

## 3. Objectives
- Train a 10-class image classifier.
- Evaluate performance on held-out test data.
- Report classification metrics.
- Generate a Grad-CAM heatmap and overlay.
- Document limitations and reproducibility.

## 4. Dataset
CIFAR-10 contains 60,000 labelled colour images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

- Training images: 50,000
- Test images: 10,000
- Source: https://www.cs.toronto.edu/~kriz/cifar.html

## 5. Methodology
ResNet-18 was modified to use a 10-class output layer and a smaller-image-friendly initial convolution. Images were resized to 64 × 64 pixels and normalised. Random horizontal flipping was applied during training.

Training configuration:
- Epochs: 5
- Batch size: 16
- Optimiser: Adam
- Learning rate: 0.001
- Loss function: cross-entropy
- Device: CUDA GPU

The model was evaluated on the separate 10,000-image test set. Grad-CAM used gradients and feature maps from the final convolutional block to create a class-specific heatmap.

## 6. Results

### Training loss

| Epoch | Training loss |
|---|---:|
| 1 | 1.3903 |
| 2 | 0.8590 |
| 3 | 0.6483 |
| 4 | 0.5282 |
| 5 | 0.4417 |

### Overall test performance

| Metric | Result |
|---|---:|
| Test images | 10,000 |
| Accuracy | 83.42% |
| Macro precision | 0.84 |
| Macro recall | 0.83 |
| Macro F1-score | 0.83 |
| Weighted precision | 0.84 |
| Weighted recall | 0.83 |
| Weighted F1-score | 0.83 |

### Per-class F1-score

| Class | F1-score |
|---|---:|
| Airplane | 0.84 |
| Automobile | 0.91 |
| Bird | 0.76 |
| Cat | 0.72 |
| Deer | 0.81 |
| Dog | 0.74 |
| Frog | 0.85 |
| Horse | 0.88 |
| Ship | 0.92 |
| Truck | 0.90 |

### Grad-CAM example
The first test image was a cat. The model predicted cat with 98.22% confidence. The visualisation was saved as `results/visualizations/cifar10_gradcam.png`.

This is one example only. Confidence for one image is not the same as overall test accuracy, and a heatmap does not prove that the model used appropriate reasoning.

## 7. Discussion
The model achieved 83.42% test accuracy after five epochs. The highest class-level F1-scores were for ships (0.92) and automobiles (0.91). Dog recall was 0.62 and cat recall was 0.68, indicating these classes were more challenging for this configuration.

Training loss decreased throughout the five epochs. However, this experiment does not establish that training has converged or that this is the best possible configuration.

## 8. Limitations
- Only five training epochs were used.
- No hyperparameter search or repeated-seed experiment was performed.
- CIFAR-10's small images do not represent all real-world conditions.
- The Grad-CAM demonstration covers one test image.
- Visual heatmaps can be ambiguous and are not proof of model reasoning.

## 9. Conclusion
A ResNet-18 classifier was trained and evaluated on CIFAR-10, achieving 83.42% test accuracy. Grad-CAM was applied to the trained model to produce a visual explanation for a correctly classified cat image. The experiment demonstrates an end-to-end classification and explanation workflow, with room for further training and broader explanation evaluation.

## 10. Reproducibility
Scripts:
- `src/train_evaluate.py`
- `src/explain_cifar10.py`
- `src/main.py` (separate ImageNet demo)

The CIFAR-10 archive must be downloaded and extracted before running the training script. Dependencies are listed in `requirements.txt`.