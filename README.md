
---

# 🫀 Arrhythmia Detection using 1D-CNN

## 📌 Project Overview

This project implements a **1D Convolutional Neural Network (1D-CNN)** to detect **arrhythmia from ECG heart signals**.
Given a raw ECG signal, the model classifies whether the patient has **arrhythmia (1)** or **normal heart rhythm (0)**.

---

## 📊 Dataset

* Format: CSV file
* Each row represents one ECG signal sample
* Features: 1D time-series signal (length = 187)
* Label:

  * `0` → Normal
  * `1` → Arrhythmia

---

## 🧠 Model Architecture

The model is built using **PyTorch** and consists of:

* 3 × Conv1D layers (feature extraction)
* ReLU activation functions
* MaxPooling layers (downsampling)
* Flatten layer
* Fully connected layers (classification head)

### Architecture Summary:

```
Input: (batch, 1, 187)

Conv1D → ReLU → MaxPool
Conv1D → ReLU → MaxPool
Conv1D → ReLU → MaxPool

Flatten
Linear (2688 → 64)
ReLU
Linear (64 → 1)
```

---

## ⚙️ Installation

```bash
pip install torch torchvision pandas numpy scikit-learn tensorboard
```

---

## 🚀 Training the Model



Training includes:

* Binary Cross Entropy Loss with Logits
* Adam Optimizer
* TensorBoard logging
* Model checkpoint saving (best & last model)

---

## 🧪 Testing the Model


Evaluation metrics:

* Accuracy
* Loss

---

## 📦 Data Preparation

Input tensor shape must be:

```
(batch_size, 1, 187)
```

Labels:

```
(batch_size, 1)
```

---

## 📈 Validation

Validation is performed after each epoch to monitor performance and save the best model.

---

## 💾 Model Saving

Two models are saved:

* `*_best.pth` → Best validation performance
* `*_last.pth` → Last training epoch

---

## 🔍 Key Features

* 1D-CNN for time-series ECG analysis
* GPU support (CUDA)
* TensorBoard visualization
* Modular PyTorch implementation
* Binary classification (arrhythmia detection)

---

## 🧠 Key Idea

The CNN automatically learns:

* ECG spikes (QRS complex)
* Rhythm patterns
* Abnormal heart behavior

---

## 📊 Results


```
Test Accuracy: 92.4%
```

---

## 👨‍💻 Author

Tewodros Bewuket
MSc Artificial Intelligence (Milano-Bicocca)

---

## 📌 Future Improvements

* Add Batch Normalization
* Improve dataset balancing
* Try LSTM / Transformer models
* Hyperparameter tuning
* Data augmentation for ECG signals

---


