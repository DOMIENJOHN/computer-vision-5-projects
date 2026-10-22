# Fruit Quality Inspection System

A deep-learning computer vision project that classifies fruit images into six categories based on fruit type and visible freshness condition.

## 1. Problem Statement

Manual fruit-quality inspection can be subjective and time-consuming. This project investigates image classification as a way to distinguish fresh and rotten apples, bananas, and oranges.

## 2. Objectives

- Classify fruit images into six categories.
- Apply transfer learning with ResNet-18.
- Train and validate the model using PyTorch.
- Generate class predictions and confidence scores for new images.
- Evaluate performance using classification metrics.

## 3. Dataset

**Source:** [Fruit Classification Dataset on Hugging Face](https://huggingface.co/datasets/menasi11/fruit-classification)

The downloaded dataset contains 1,058 labelled training images and 453 test images.

Classes:
- `fresh_apple`
- `fresh_banana`
- `fresh_orange`
- `rotten_apple`
- `rotten_banana`
- `rotten_orange`

Training labels are provided by `train.csv`. The test directory includes `sample_submission.csv`; its labels have not been independently verified and are not treated as authoritative test ground truth.

## 4. Methodology

1. Read image filenames and labels from the training CSV.
2. Split the labelled training data into training and validation subsets using a fixed random seed of 42 and stratification by class.
3. Resize images to 224 × 224 pixels.
4. Apply random horizontal flipping, rotation, and modest colour augmentation during training.
5. Normalize images using the pretrained ResNet-18 weights' normalization parameters.
6. Fine-tune a ResNet-18 model with a six-class output layer.
7. Optimize using AdamW and cross-entropy loss.
8. Select the checkpoint with the best validation accuracy.
9. Use the saved checkpoint to predict new images.

## 5. Technology Stack

- Python 3.12
- PyTorch and Torchvision
- ResNet-18 transfer learning
- Pillow, NumPy, Pandas
- scikit-learn
- Matplotlib and Seaborn
- NVIDIA GPU with CUDA

## 6. Recorded Results

| Metric | Result |
|---|---:|
| Training images | 846 |
| Validation images | 212 |
| Best epoch | 5 |
| Validation accuracy | 99.06% |
| Number of classes | 6 |

These are results from the recorded validation split. They are not independently verified test-set results.

A comparison against labels in `sample_submission.csv` produced only 16% agreement. Visual inspection of sampled test images showed apparent disagreements between several CSV labels and the visible fruit. Consequently, this comparison is considered unreliable and is not reported as official test accuracy.

## 7. Run the Project

Activate the project environment and install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run evaluation and generate test predictions:

```bash
python src/evaluate.py
```

Predict a single image:

```bash
python src/predict.py path/to/fruit_image.png
```

## 8. Output Files

- `results/models/fruit_resnet18_best.pth` — best validation checkpoint.
- `results/metrics/training_summary.json` — recorded training summary.
- `results/metrics/training_history.csv` — epoch-level metrics.
- `results/metrics/validation_report.json` — validation classification report.
- `results/test_predictions.csv` — test image predictions and class probabilities.
- `results/visualizations/` — diagnostic contact sheets and confusion-matrix visualizations.

## 9. Limitations

- The dataset is relatively small.
- Validation performance may not represent performance on unseen real-world images.
- The test CSV label provenance remains unresolved.
- Model confidence is not a guarantee of correctness.
- Lighting, background, camera quality, fruit variety, and the degree of decay may affect predictions.
- This prototype is not a substitute for food-safety inspection.

## 10. Conclusion

A six-class fruit classifier was trained using ResNet-18 transfer learning. The recorded validation accuracy was 99.06%. Further evaluation with independently verified test labels and additional real-world images is required before making claims about generalisation.
