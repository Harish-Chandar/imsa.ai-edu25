# General Neural Networks
`Lecture Notes from December 03, 2025`   

## Key Pieces of Vobabularly
* **Vector**: An ordered list of numbers, can represent basically anything
* **Matrix**: A 2D array of numbers
    - A complex vector type
    - Linear Algebra tells us that every transformation we do to vectors can be represented as a matrix multiplication
* **Graph**: A collection of **nodes** connected by **edges**
    - Can represent neural networks

## Philosophy of Neural networks
* Inspired by brains (supposedly)
* Supposed to "learn" patterns in data
* Composed of layers of interconnected nodes

## General Structure
* Layers of **nodes**
* Interconnected by **edges** or **weights**
* Each node in each layer connects to every node in the next layer - deeply interconnected structure

```
Input Layer      Hidden Layer(s)      Output Layer
   O                 O                   O
   | \               | \                 |
   O  O------------> O  O ------------>  O
   | /               | /                 |
   O                 O                   O
```     

### Layers
* **Input Layer**: Receives raw data 
* **Hidden Layer(s)**: Perform computations and extract features
* **Output Layer**: Produces final predictions or classifications

### Nodes
* Place for edges to connect
* Each node applies a mathematical **activation function** to its inputs
* Common functions:
  - Sigmoid (logistic function)
  - Tanh (hyperbolic tangent)
  - ReLU (Rectified Linear Unit)

### Edges
* Each edge has an associated **weight** that determines the strength of the connection
* Weights are adjusted during training to minimize prediction error (remember cost function?)

### Writing a Neural Network in PyTorch:
```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.layer1 = nn.Linear(3, 5)
        self.layer2 = nn.Linear(5, 6)
        self.layer3 = nn.Linear(6, 4)
        self.layer4 = nn.Linear(4, 7)
        self.output_layer = nn.Linear(7, 5)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.output_layer(x)
        return x

model = SimpleNN()
print(model)
```

## Feedforward Process (Inference)
[Youtube Animation](https://youtu.be/xtzVuln1PV8)
1. Input data is fed into the input layer - one quantity per node
2. Data "flows" through the network layer by layer, edge by edge   

Each node computes a weighted sum of its inputs, applies the activation function, 
and passes the result to the next layer. For a node $j$ in layer $l$:

$$
z_j^{(l)} = \sum_i w_{ij}^{(l-1)} a_i^{(l-1)} + b_j^{(l)}
$$

$$
a_j^{(l)} = f(z_j^{(l)})
$$

Let:

- $w_{ij}^{(l-1)}$ is the weight from node $i$ in layer $l-1$ to node $j$ in layer $l$
- $a_i^{(l-1)}$ is the activation of node $i$ in layer $l-1$
- $b_j^{(l)}$ is the bias term for node $j$ in layer $l$
- $f$ is the activation function

3. The output layer produces the final predictions

## Backpropagation (Training)
1. Compute the error at the output layer using a loss function (e.g., Mean Squared, Cross-Entropy, etc)
2. Propagate the error backward through the network to compute gradients
    - Calculate each loss gradient with respect to each weight using the Calculus chain rule (derivative of activation function)
    - Partial derivatives (Multivariable Calculus) tell us how to adjust each weight to reduce error in gradient descent
3. Update weights using an optimization algorithm (e.g., Stochastic Gradient Descent, Adam, etc)
    - New weight = Old weight - Learning Rate * Gradient

### Gradient Descent Optimizer
- As if we are descending down a hill to find the lowest point (minimum error)
- Learning rate controls the step size during weight updates

### Adam (Adaptive Movement Estimator) Optimizer
- Advanced optimization algorithm that adapts learning rates for each weight individually
- Uses estimates of previous moments of gradients to improve convergence (the way it reaches the minimum error point)
- Momentum (like the momentum of a ball rolling down a hill) helps smooth out updates by not reacting too strongly to recent changes
- Generally performs better than standard gradient descent

## Common Problems
* **Overfitting**: Model learns training data too well, performs poorly on new data
    - Regularization, Dropout, Early Stopping
* **Underfitting**: Model is too simple to capture underlying patterns
    - Increase model complexity, add more features (inputs)
* **Vanishing/Exploding Gradients**: Gradients become too small or too large during backpropagation
    - ReLU is good at fighting vanishing gradients (known, constant gradient)
    - Normalization techniques (BatchNorm, LayerNorm) and preprocessing inputs
* **Computational Cost**: Training deep networks can be resource-intensive
    - Use GPUs with Tensors, distributed training, efficient architectures (e.g., CNNs for images)

## New Topics
* Convolutional Neural Networks (CNNs)
* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory (LSTM) Networks

### Over break
* Review basic Python syntax for defining classes and functions
* Email/Messenger me any questions!

### Much Later...
* Transformers and Attention Mechanisms

