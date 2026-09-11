# 🦴 Bone Fracture Detection using CNN & PSO

A deep learning project for binary image classification — detecting **fractured** vs **non-fractured** bones from X-ray images. Two models are compared: a baseline CNN and a PSO-optimized CNN.

---

## 📁 Dataset Structure

```
dataset/
├── train/
│   ├── fractured/
│   └── not Fractured/
├── val/
│   ├── fractured/
│   └── not Fractured/
└── test/
    ├── fractured/
    └── not Fractured/
```

All images are resized to **224×224 RGB** during preprocessing.

---

## 🔧 Requirements

```bash
pip install tensorflow torch numpy matplotlib seaborn scikit-learn opencv-python pillow albumentations tqdm pyswarm
```

> **Note:** CUDA-compatible GPU is recommended for faster training. The code includes GPU availability checks via PyTorch and runs training on TensorFlow/Keras.

---

## 🔬 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Class distribution pie charts for train / val / test splits
- Random sample visualization (20 images per class)
- Image size and channel distribution analysis
- Broken/corrupted image detection

### 2. Preprocessing
| Split      | Augmentation |
|------------|-------------|
| Train      | Rotation ±20°, width/height shift ±20%, zoom ±20%, horizontal flip |
| Validation | Rescale only (1/255) |
| Test       | Rescale only (1/255) |

Batch size: **32** — Shuffled training, fixed seed for reproducibility.

---

## 🧠 Models

### Model 1 — Baseline CNN

| Layer         | Details                        |
|---------------|-------------------------------|
| Conv2D × 4    | 32 → 64 → 128 → 64 filters, 3×3, ReLU |
| MaxPooling2D  | 2×2 after each Conv block     |
| Dense         | 128 units, ReLU               |
| Dropout       | 0.3                           |
| Output        | 1 unit, Sigmoid               |

- **Optimizer:** AdamW (lr = 1e-4)  
- **Loss:** Binary Crossentropy  
- **Epochs:** 15

---

### Model 2 — CNN + PSO Hyperparameter Optimization

Same architecture as Model 1, but **learning rate** and **dropout rate** are optimized using **Particle Swarm Optimization (PSO)**.

| PSO Parameter   | Value          |
|-----------------|----------------|
| Swarm size      | 8 particles    |
| Max iterations  | 5              |
| LR search range | [1e-5, 1e-3]   |
| Dropout range   | [0.2, 0.5]     |
| Fitness metric  | Validation accuracy (maximized) |

Each particle trains for **3 epochs** to evaluate fitness. The best hyperparameters are then used to train the final model for **15 epochs**.

---

## 📊 Evaluation

Both models are evaluated on:

- Accuracy (Train / Validation / Test)
- Precision & Recall curves over epochs
- F1 Score on the test set
- Confusion matrices (Train & Test)
- Classification report

---

## 📈 Results Visualization

The following plots are generated automatically:

- Pie charts — class distribution per split
- Bar charts — top image sizes, channel distribution
- Training curves — Precision & Recall vs. epochs
- Confusion matrices — heatmaps for train and test sets
- Accuracy comparison bar chart — Train vs. Val vs. Test

---

## 🚀 How to Run

1. Update the dataset paths in the notebook to point to your local directories.
2. Run the **EDA** section to inspect and validate your data.
3. Run **Preprocessing** to set up data generators.
4. Train **Model 1** (baseline CNN).
5. Run **PSO optimization**, then train **Model 2** with best hyperparameters.
6. Compare results using the evaluation and visualization cells.

---

## 📌 Notes

- `ImageFile.LOAD_TRUNCATED_IMAGES = True` is set to handle partially corrupted images gracefully.
- `CUDA_LAUNCH_BLOCKING=1` is enabled for easier CUDA error debugging.
- Anomaly detection is enabled via `torch.autograd.set_detect_anomaly(True)`.

---

## 🏗️ Tech Stack

| Library        | Purpose                          |
|----------------|----------------------------------|
| TensorFlow / Keras | Model building & training    |
| PyTorch        | GPU verification                 |
| scikit-learn   | Metrics & class weight utilities |
| pyswarm        | PSO optimization                 |
| Albumentations | (Imported) Advanced augmentation |
| OpenCV / PIL   | Image reading & inspection       |
| Matplotlib / Seaborn | Visualization               |


| Split         | Dataset                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------- |
| 🏋️ Train     | [Google Drive](https://drive.google.com/drive/folders/1WrzNIjFjA7pxP7mQdDLjuqjBzG4sRIAF?usp=sharing) |
| 🔍 Validation | [Google Drive](https://drive.google.com/drive/folders/1A4TbX7qZFOFnM6VXvAIgIvC-PG5PwuD2?usp=sharing) |
| 🧪 Test       | [Google Drive](https://drive.google.com/drive/folders/1XGZjaZpTupouwynd_mK9-l0CCHNKIi5i?usp=sharing) |

