# Technical Report: Fruit Quality Inspection System

## Abstract

This project implements a six-class fruit image classifier using transfer learning with ResNet-18. The model distinguishes fresh and rotten apples, bananas, and oranges. The training pipeline includes image preprocessing, data augmentation, validation-based checkpoint selection, and command-line inference.

## 1. Problem Definition

Visual fruit-quality inspection can be inconsistent when performed manually. The project explores whether a convolutional neural network can classify fruit type and visible freshness condition from images.

## 2. Dataset

The dataset was obtained from the Hugging Face repository `menasi11/fruit-classification`.

- Labelled training images: 1,058
- Test images: 453
- Classes: 6
- Training labels: `train.csv`

A stratified split with random seed 42 allocated 846 images to training and 212 to validation.

The test CSV is named `sample_submission.csv`. Its labels were not independently verified. Visual inspection of sampled test images revealed apparent label-image mismatches; therefore, comparison against those labels is not considered a reliable test evaluation.

## 3. Method

The implementation uses a pretrained ResNet-18 backbone with a replacement fully connected layer for six classes. Images are resized to 224 × 224 pixels and normalized using the pretrained model's expected mean and standard deviation.

Training augmentation includes horizontal flipping, small rotations, and modest colour jitter. The optimizer is AdamW with cross-entropy loss. Validation accuracy is used to select the saved checkpoint.

## 4. Experimental Results

Recorded best checkpoint:
- Epoch: 5
- Training images: 846
- Validation images: 212
- Validation accuracy: 0.990566, approximately 99.06%
- Device: NVIDIA GPU using CUDA

The displayed validation classification report rounded per-class precision, recall, and F1-scores to 0.98–1.00.

A provisional comparison against `sample_submission.csv` yielded approximately 16% accuracy. Because the label provenance is unresolved and sampled labels appear inconsistent with visible images, this result is not accepted as a trustworthy estimate of test performance.

## 5. Interpretation

The validation result indicates strong performance on the selected validation subset. However, a small dataset and a single split limit the strength of generalisation claims. High prediction confidence does not establish correctness. A verified external test set is needed for a defensible final test metric.

## 6. Limitations and Future Work

- Verify every test label against an authoritative source.
- Evaluate against an independently labelled test set.
- Check for near-duplicate images and possible dataset leakage.
- Use repeated stratified splits or cross-validation.
- Test images from different cameras, lighting conditions, and fruit varieties.
- Consider calibration of confidence scores and explainability visualizations.

## 7. Conclusion

The project delivers a working ResNet-18 fruit classifier, training pipeline, validation report, test prediction export, and command-line inference tool. The recorded validation accuracy is 99.06%. Reliable independent test evaluation remains outstanding.
