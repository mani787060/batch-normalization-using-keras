# Batch Normalization with Keras | Deep Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-purple)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Keras](https://img.shields.io/badge/Library-Keras-red)
![Status](https://img.shields.io/badge/Status-Learning%20Project-green)

A practical implementation of **Batch Normalization using Keras Sequential models**. This project compares a basic neural network with a Batch-Normalized neural network on a concentric-circle classification dataset to explore how normalization affects model training and performance.

---

## 📌 Project Overview

Training a neural network can be challenging when the distribution of activations changes throughout the learning process. Batch Normalization is a technique that helps stabilize training by normalizing layer activations using batch statistics.

This project demonstrates Batch Normalization through a binary classification experiment.

Two neural network architectures are implemented:

1. **Baseline Model:** A simple Sequential neural network without Batch Normalization.
2. **Batch-Normalized Model:** A Sequential neural network containing Keras `BatchNormalization()` layers.

Both models are trained on the same dataset so their training and validation behavior can be explored.

## 🎯 Project Objectives

- Understand the concept of Batch Normalization.
- Implement neural networks using Keras Sequential API.
- Compare a baseline model with a Batch-Normalized model.
- Understand trainable and non-trainable parameters in Batch Normalization.
- Observe training loss and accuracy over multiple epochs.
- Explore how normalization can influence neural network optimization.

## 📂 Dataset

The project uses the dataset:

**`concertriccir2.csv`**

The CSV file contains three columns:

| Column | Description |
|---|---|
| `X` | First input feature |
| `Y` | Second input feature |
| `class` | Binary classification target |

The two input features are used to classify observations into their respective classes.

The notebook loads this existing CSV file using Pandas and visualizes the data using Matplotlib.

### Data Loading

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("concertriccir2.csv")

df.columns = ["X", "Y", "class"]

plt.scatter(df["X"], df["Y"], c=df["class"])
plt.show()

X = df.iloc[:, 0:2].values
y = df.iloc[:, -1].values
```

## 🧠 Understanding Batch Normalization

Batch Normalization is a neural network technique that normalizes activations using the mean and variance calculated from a mini-batch during training.

Its general transformation is:

\[
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
\]

\[
y=\gamma\hat{x}+\beta
\]

Where:

- \(x\): Input activation
- \(\mu_B\): Mini-batch mean
- \(\sigma_B^2\): Mini-batch variance
- \(\epsilon\): Small constant for numerical stability
- \(\gamma\): Learnable scaling parameter
- \(\beta\): Learnable shifting parameter

Keras also maintains moving statistics for inference.

## 🏗️ Model Implementations

### 1. Baseline Neural Network

The first model uses a simple Sequential architecture without Batch Normalization.

**Architecture:**

```text
Input Features (2)
       |
       v
Dense Layer (2 neurons, ReLU)
       |
       v
Dense Layer (2 neurons, ReLU)
       |
       v
Output Layer (1 neuron, Sigmoid)
       |
       v
Binary Classification
```

Implementation:

```python
from keras import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(2, activation="relu", input_dim=2))
model.add(Dense(2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()
```

### 2. Neural Network with Batch Normalization

The second model introduces Batch Normalization layers between the Dense layers.

**Architecture:**

```text
Input Features (2)
       |
       v
Dense Layer (3 neurons, ReLU)
       |
       v
Batch Normalization
       |
       v
Dense Layer (2 neurons, ReLU)
       |
       v
Batch Normalization
       |
       v
Output Layer (1 neuron, Sigmoid)
       |
       v
Binary Classification
```

Implementation:

```python
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

model_2 = Sequential()

model_2.add(Dense(3, activation="relu", input_dim=2))
model_2.add(BatchNormalization())

model_2.add(Dense(2, activation="relu"))
model_2.add(BatchNormalization())

model_2.add(Dense(1, activation="sigmoid"))

model_2.summary()
```

**Implementation note:** The code places Batch Normalization after the Dense layers with ReLU activation. This README documents the existing implementation rather than changing its architecture.

## ⚙️ Model Training

Both models use the same training configuration:

| Parameter | Configuration |
|---|---|
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Evaluation Metric | Accuracy |
| Epochs | 200 |
| Validation Split | 20% |

### Baseline Model

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    X,
    y,
    epochs=200,
    validation_split=0.2
)
```

### Batch-Normalized Model

```python
model_2.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history2 = model_2.fit(
    X,
    y,
    epochs=200,
    validation_split=0.2
)
```

## 📊 Model Comparison

The notebook explores the training behavior of both architectures.

The comparison focuses on:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Model architecture and parameter counts

The experiment helps illustrate how adding Batch Normalization changes the model architecture and its optimization behavior.

**Important:** Batch Normalization does not guarantee better performance in every experiment. The outcome depends on the dataset, model architecture, batch size, learning rate, and other training settings.

## 💻 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep Learning framework |
| Keras | Neural network implementation |
| NumPy | Numerical operations |
| Pandas | Dataset loading and manipulation |
| Matplotlib | Data visualization and plotting |
| Jupyter Notebook | Interactive development environment |

## 📁 Repository Structure

```text
batch-normalization-using-keras/
│
├── batch_normalization_using_keras.ipynb
│   └── Complete implementation and model comparison
│
├── concertriccir2.csv
│   └── Concentric-circle classification dataset
│
└── README.md
    └── Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/batch-normalization-using-keras.git

cd batch-normalization-using-keras
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install tensorflow numpy pandas matplotlib jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

`batch_normalization_using_keras.ipynb`

Run the notebook cells sequentially. Keep `concertriccir2.csv` in the same directory as the notebook so that the dataset can be loaded correctly.

## 📚 Key Concepts Learned

Through this implementation, the project explores:

- Artificial Neural Networks (ANN)
- Keras Sequential API
- Dense layers and activation functions
- ReLU and Sigmoid activations
- Batch Normalization
- Mini-batch statistics
- Trainable and non-trainable parameters
- Binary Crossentropy
- Adam optimization
- Training and validation metrics
- Neural network architecture comparison

## 🔮 Future Improvements

Potential extensions to this project include:

- Compare models using identical hidden-layer architectures.
- Experiment with different batch sizes and learning rates.
- Test Batch Normalization before activation functions.
- Visualize training and validation loss curves together.
- Compare convergence across multiple random seeds.
- Experiment with larger neural networks and different datasets.
- Add a separate test dataset for more reliable evaluation.

## 🎯 Learning Objective

The main objective is to understand how Batch Normalization works and how it can be incorporated into neural network architectures using Keras.

This project is part of my ongoing journey of learning Deep Learning concepts through practical Python implementations.

---

## 👨‍💻 Author

**Mani Ratan**

- GitHub: [@mani787060](https://github.com/mani787060)
- LinkedIn: [Mani Ratan](https://www.linkedin.com/in/mani-ratan-159963289/)
- Kaggle: [@mani787060](https://www.kaggle.com/mani787060)

If you find this project useful for learning Deep Learning and Keras, feel free to explore the notebook and share your feedback.

⭐ If you find this repository helpful, consider giving it a star!
