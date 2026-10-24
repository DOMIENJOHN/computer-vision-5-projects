
# Waste Sorting Robot

## Overview
A computer vision project that classifies waste images into:
- Biodegradable
- Non-biodegradable

The model uses MobileNetV3-Small with transfer learning.

## Test Results

Evaluation on a held-out test set of 1,523 images:

| Metric | Result |
|---|---:|
| Test accuracy | 98.56% |
| Test macro-F1 | 96.48% |
| Biodegradable F1-score | 99.18% |
| Non-biodegradable F1-score | 93.79% |

These results reflect the supplied dataset and do not guarantee the same performance on real-world camera images.

## Project Structure

- `src/train.py` — trains the model
- `src/evaluate.py` — evaluates the saved model
- `src/prepare_splits.py` — prepares train, validation and test splits
- `src/clean_manifest.py` — excludes unreadable images
- `src/check_images.py` — checks image readability
- `src/dataset_check.py` — inspects the dataset
- `models/best_model.pth` — trained model checkpoint
- `results/` — reports, predictions, plots and logs

## Dataset

The dataset is not included in this repository. Configure the dataset path and prepare the cleaned split manifest before training or evaluation.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Evaluate the Model

Run from this project directory:

```bash
python src/evaluate.py
```

## Train the Model

```bash
python src/train.py
```

Training uses `results/splits_clean.csv` and saves the best checkpoint in `models/`.

## Results Files

- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/test_predictions.csv`
- `results/training_history.png`
