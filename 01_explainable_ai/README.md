# Explainable AI for Object Recognition

## Problem Statement
Image classifiers can produce predictions without clearly showing which
image regions influenced those predictions. This project combines object
classification with visual explanations.

## Objective
Build an image classifier and use Gradient-weighted Class Activation Mapping
(Grad-CAM) to visualise regions influencing its predictions.

## Dataset
CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

See [dataset information](dataset/dataset_info.md).

## Methodology
1. Load and preprocess CIFAR-10.
2. Train an image classification model.
3. Evaluate it on held-out test data.
4. Generate Grad-CAM heatmaps for sample images.
5. Overlay heatmaps on input images and save visualisations.

## Tools and Libraries
Python, PyTorch, Torchvision, OpenCV, NumPy, Matplotlib and scikit-learn.

## Results
To be populated with actual results after running the implementation.

## Performance Evaluation
Accuracy, precision, recall, F1-score and confusion matrix will be calculated
from predictions on the held-out test split.

## Limitations
Grad-CAM is a visual explanation technique, not proof that a model reasons
correctly. CIFAR-10 is limited in image resolution and class diversity.

## Installation and Usage
Install dependencies from `requirements.txt`. See the source-code instructions
once the implementation is complete.

## References
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- Grad-CAM: https://arxiv.org/abs/1610.02391
