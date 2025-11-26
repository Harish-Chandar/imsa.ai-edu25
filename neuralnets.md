# Neural Networks - Lecture Notes

## Key Pieces of Vobabularly
* **Vector**: An ordered list of numbers, can represent basically anything
* **Matrix**: A 2D array of numbers
    - A complex vector type
    - Linear Algebra tells us that every transformation we do to vectors can be represented as a matrix multiplication
* **Graph**: A collection of **nodes** connected by **edges**
    - Can represent neural networks


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

## Feedforward Process
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
