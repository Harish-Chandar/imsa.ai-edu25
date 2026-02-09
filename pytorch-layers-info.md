# PyTorch Layers

---

### 1. Linear / Dense Layer (`nn.Linear`)

**Input:** number of input numbers and number of output numbers.

**Why you might use it:**

* To learn patterns between numbers in your data.
* Easy and simple for many models.

**Why you might not use it:**

* Can overfit small datasets.
* Doesn’t work directly with images or sequences.

**Example:**

```python
fc = nn.Linear(128, 64)   # 128 numbers in → 64 numbers out
```

---

### 2. Activation (`ReLU`, `Sigmoid`, `Softmax`)

**Input:** output from the previous layer.

**Why you might use it:**

* Lets the model learn more complex patterns.
* ReLU is fast and works well in deep networks.

**Why you might not use it:**

* Some activations can slow training.
* ReLU can sometimes stop some neurons from learning.

**Example:**

```python
x = F.relu(output)  # output from previous Linear layer
```

---

### 3. Dropout (`nn.Dropout`)

**Input:** probability of dropping neurons (between 0 and 1).

**Why you might use it:**

* Reduces overfitting.
* Forces the model to learn more robust features.

**Why you might not use it:**

* Slows training a little.
* Should be turned off when testing.

**Example:**

```python
drop = nn.Dropout(0.2)  # drop 20% of neurons
```

---

### 4. Batch Normalization (`nn.BatchNorm1d` / `nn.BatchNorm2d`)

**Input:** number of features (`BatchNorm1d`) or number of channels (`BatchNorm2d`).

**Why you might use it:**

* Helps the layer outputs stay in a good range, which makes training more stable.
* Can make the network learn faster and sometimes improve accuracy.

**Why you might not use it:**

* Adds extra steps for the computer, so training is a bit slower.
* Not always necessary for small or simple networks.

**Example:**

```python
bn = nn.BatchNorm1d(64)  # normalize 64 features
```

Ahhh, got it this time — you just want **LayerNorm on its own**, no Linear involved. Here’s a clean Markdown page just for **Layer Normalization**:

---

# Layer Normalization (`nn.LayerNorm`)

**Input:** number of features in the layer (1D for simple layers, 2D/3D for sequences/images)

---

**Why you would use it:**

* Keeps the outputs of a layer balanced so training is more stable.
* Works well for sequences or very small batches where BatchNorm struggles.
* Can improve training speed and model performance.

**Why you might not use it:**

* Adds extra computation compared to not normalizing.
* Might not be necessary for very simple or small networks.
* For large image batches, BatchNorm may still give better results.

---

**Example:**

```python
ln = nn.LayerNorm(64)   # normalize 64 features
```