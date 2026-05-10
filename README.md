# Batch Normalization with Keras (Sequential Model)
[![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-blue)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Library-Keras%20/%20TensorFlow-red)](https://www.tensorflow.org/)

## 🏗️ Project Overview
This repository focuses on **Optimization Techniques** in Deep Learning. Specifically, it demonstrates the implementation of **Batch Normalization (BN)** to address the challenges of training deep neural networks. By normalizing the activations of the previous layer at each batch, we can maintain a stable distribution of inputs (reducing internal covariate shift), allowing for higher learning rates and faster convergence.

To test the effectiveness of this technique, we utilize a **Concentric Circle Dataset**—a non-linearly separable challenge that requires a model to learn complex spatial relationships.

---

## 🛠️ Key Technical Implementations

### 1. The Concentric Circles Challenge
* **Dataset Generation:** Using Scikit-Learn's `make_circles` to generate synthetic data where one class completely surrounds the other.
* **Non-Linearity:** Proving that standard linear solvers fail and requiring a deep architecture with non-linear activation functions (ReLU).

### 2. Batch Normalization Layer
* **Internal Covariate Shift:** Implementing `BatchNormalization()` to ensure that the mean and variance of hidden layer activations stay consistent during training.
* **Placement Strategy:** Demonstrating the standard practice of placing BN layers after the linear transformation and before the activation function.

### 3. Model Performance & Optimization
* **Faster Convergence:** Analyzing how the model achieves lower loss in fewer epochs compared to a baseline model without BN.
* **Regularization Effect:** Observing how Batch Normalization provides a slight regularization effect, reducing the need for heavy Dropout in smaller architectures.

---

## 💻 Tech Stack
* **Language:** Python 3.9+
* **Deep Learning:** Keras / TensorFlow
* **Data Generation:** Scikit-Learn
* **Visualization:** Matplotlib, NumPy
* **Environment:** Jupyter Notebook

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/batch-normalization-keras-concentric-circles.git](https://github.com/your-username/batch-normalization-keras-concentric-circles.git)

2. **Install Dependencies:**
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy

3. **Run the Implementation:**

   Open `batch normalization using keras.ipynb` to see the comparison between standard training and Batch-Normalized training.   
