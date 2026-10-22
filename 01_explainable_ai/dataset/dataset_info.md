# Dataset Information — CIFAR-10

## Official Source
https://www.cs.toronto.edu/~kriz/cifar.html

## Dataset Overview
CIFAR-10 is a labelled image dataset containing 60,000 colour images divided into 10 classes.

- Training set: 50,000 images
- Test set: 10,000 images
- Original image dimensions: 32 × 32 pixels
- Image channels: RGB
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck

## Dataset Usage
The dataset is used to train and evaluate the ResNet-18 classifier.

Preprocessing:
1. Resize images to 64 × 64 pixels.
2. Convert images to tensors.
3. Normalise the RGB channels.
4. Apply random horizontal flipping to training images only.

The model is evaluated on the separate 10,000-image test set.

## Experimental Results
- Test accuracy: 83.42%
- Macro precision: 0.84
- Macro recall: 0.83
- Macro F1-score: 0.83
- Training epochs: 5

These are results from the completed training run, not guaranteed results for every retraining run.

## Dataset and Sample Distinction
The `samples/dog.jpg` image is used by the separate ImageNet-pretrained ResNet-18 demonstration. It is not part of CIFAR-10 and is not included in the reported CIFAR-10 metrics.

## Limitations
CIFAR-10 contains small images and only 10 object categories. Performance on this dataset does not guarantee similar performance on real-world images with different lighting, viewpoints, backgrounds or image resolutions.