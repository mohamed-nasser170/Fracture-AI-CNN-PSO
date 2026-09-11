# FractureAI-CNN-PSO

## Bone Fracture Classification Using Deep Learning and Particle Swarm Optimization

An AI-based deep learning project for classifying X-ray images into two categories:

- Fractured
- Not Fractured

The project combines a Convolutional Neural Network (CNN) with Particle Swarm Optimization (PSO) to optimize important hyperparameters and improve classification performance.

---

## Project Overview

Bone fracture classification from X-ray images is an important computer vision task in medical imaging.

Manual examination of X-ray images can be time-consuming and may be affected by human error. Deep learning provides an opportunity to automatically analyze X-ray images and assist in distinguishing between fractured and non-fractured cases.

In this project, a CNN-based classification model was developed and then optimized using Particle Swarm Optimization (PSO).

The complete workflow includes:

1. Dataset analysis
2. Image preprocessing
3. Data augmentation
4. CNN model development
5. Model training
6. Model evaluation
7. Hyperparameter optimization using PSO
8. Comparison between the base and optimized models
9. Deployment using Streamlit

---

## Objectives

The main objectives of the project are:

- Build a deep learning model for bone fracture classification.
- Classify X-ray images into fractured and non-fractured classes.
- Apply image preprocessing and augmentation techniques.
- Develop a CNN baseline model.
- Optimize CNN hyperparameters using Particle Swarm Optimization.
- Compare the performance of the baseline and optimized models.
- Deploy the trained model through a Streamlit application.

---

## Dataset

The dataset contains X-ray images divided into three subsets:

- Training Set
- Validation Set
- Test Set

Each subset contains two classes:

- Fractured
- Not Fractured

### Dataset Analysis

The following analysis was performed:

- Class distribution visualization
- Image size analysis
- Channel inspection
- RGB / Grayscale inspection
- Detection of corrupted images

---

## Data Preprocessing

All images were resized to:

```text
224 × 224
