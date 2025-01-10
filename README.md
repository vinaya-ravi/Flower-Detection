**FloraVision: AI-Powered Flower Recognition**<br>

**Project Overview**<br>

This project aims to develop an automated flower detection and classification system using machine learning and AI techniques. The system leverages Convolutional Neural Networks (CNNs) for feature extraction and Logistic Regression for classification. The primary goal is to accurately identify various flower species from images.

**Dataset**<br>

The dataset used is the Oxford 102 Flower Dataset, containing 8,189 images of flowers belonging to 102 different classes.

**Preprocessing**<br>

Resizing: Images resized to 224x224 or 299x299 pixels based on model requirements.
Normalization: Pixel values scaled to a standard range.
Data Augmentation: Techniques like rotation, flipping, and zooming applied.

**Feature Extraction**<br>
Pretrained CNN models (VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet, Xception) are used to extract features from the images. These features are then flattened and saved for training the classifier.<br>
**Model Training**<br>
Logistic Regression is used as the classifier for the extracted features.<br>
**Model Evaluation**<br>
Rank-1 Accuracy: 96.32%
Rank-5 Accuracy: 100%

**Results**<br>

The model achieved high accuracy, demonstrating the effectiveness of combining CNNs for feature extraction with Logistic Regression for classification.

**Future Work**<br>

Experiment with more complex classifiers like SVMs or deep learning-based classifiers.
Evaluate on larger and more diverse datasets.
Implement advanced preprocessing and data augmentation techniques.
Incorporate feedback mechanisms and active learning.
