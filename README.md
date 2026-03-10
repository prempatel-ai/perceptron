# Sigmoid Perceptron from Scratch

A sigmoid perceptron built using only numpy — no PyTorch, no TensorFlow, no sklearn. Just math and Python.

The goal was to actually understand what happens inside a neural network instead of just calling `.fit()` on a library and getting a number back. Trained on a small student pass/fail dataset to keep things simple and easy to follow.

---

## What it does

Takes input features, multiplies each by a learned weight, adds a bias, passes the result through a sigmoid function, and outputs a probability between 0 and 1. Learns by computing the error after each prediction and adjusting weights using gradient descent.

That's it. No magic.

---

## Project structure

```
Perceptron/
├── Perceptron.py          # the perceptron class
├── notebook.ipynb         # training, evaluation, predictions
├── perceptron_theory.md   # math and theory notes
└── README.md
```

---

## The math

Three equations run the whole thing:

**Forward pass**
$$\hat{y} = \frac{1}{1 + e^{-(\mathbf{x}\cdot\mathbf{w}+b)}}$$

**Error**
$$\text{error} = y - \hat{y}$$

**Weight update**
$$w \mathrel{+}= \alpha \cdot \text{error} \cdot \hat{y}(1-\hat{y}) \cdot x$$
$$b \mathrel{+}= \alpha \cdot \text{error} \cdot \hat{y}(1-\hat{y})$$

The term $\hat{y}(1-\hat{y})$ is the derivative of sigmoid. It makes the model learn fast when it is uncertain and slow down when it gets confident — that behavior comes from the math itself, not from any extra code.

---

## Dataset

10 students. Two features: study hours and sleep hours. Label: pass (1) or fail (0).

| Study hrs | Sleep hrs | Result |
|---|---|---|
| 9 | 8 | Pass |
| 8 | 7 | Pass |
| 7 | 8 | Pass |
| 6 | 6 | Pass |
| 5 | 7 | Pass |
| 4 | 5 | Fail |
| 3 | 4 | Fail |
| 2 | 6 | Fail |
| 1 | 3 | Fail |
| 2 | 4 | Fail |

8 students for training, 2 for testing. Features are scaled manually using z-score normalization before training.

---

## Usage

```python
from Perceptron import Perceptron
import numpy as np

# data
X_train = ...   # shape (n, features)
Y_train = ...   # shape (n,)

# scale
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std

# train
model = Perceptron(input_size=2)
model.fit(X_train_scaled, Y_train, num_epochs=100, learning_rate=0.1)

# evaluate
acc = model.evaluate(X_train_scaled, Y_train)
print(f"Accuracy: {round(acc*100, 2)}%")

# predict
new = np.array([6, 7], dtype=float)
new_scaled = (new - mean) / std
prob = model.predict(new_scaled)
print(f"Probability of passing: {round(float(prob)*100, 2)}%")
```

---

## Methods

| Method | What it does |
|---|---|
| `predict(inputs)` | returns probability between 0 and 1 |
| `fit(inputs, targets, num_epochs, learning_rate)` | trains the model, prints loss each epoch |
| `evaluate(inputs, targets)` | returns accuracy as a decimal |
| `save(filename)` | saves weights and bias to a .npy file |
| `load(filename)` | loads weights and bias from a .npy file |

---

## Training output

```
Epoch 1/100   |  Loss: 0.2481
Epoch 2/100   |  Loss: 0.2134
Epoch 3/100   |  Loss: 0.1821
...
Epoch 100/100 |  Loss: 0.0312
```

Loss tracks mean squared error across all training examples per epoch. Should go down consistently. If it is not going down, the learning rate is probably too high.

---

## Requirements

```
numpy
matplotlib   # only for plotting the loss curve
```

Nothing else.

---

## Notes

Feature scaling is done manually — no StandardScaler from sklearn. The same mean and std from the training set is applied to the test set. Computing stats from the test set would leak information and make results look better than they actually are.

The `random_state` is set to 42 before creating the model so results are reproducible across runs.

For a deeper look at the math behind each part, check `perceptron_theory.md`.
