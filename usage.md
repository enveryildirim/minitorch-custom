# MiniTorch Usage Documentation

Welcome to the MiniTorch documentation. This guide will help you understand the core concepts, build neural networks, and implement machine learning algorithms using the MiniTorch library.

## Installation

To install `minitorch` in editable mode, run the following command in the project root:

```bash
pip install -Ue .
```

## Core Components Deep Dive

MiniTorch is built around two main automatic differentiation primitives: **Scalar** and **Tensor**.

### Scalar

`minitorch.Scalar` represents a single floating-point number that tracks its history of operations for automatic differentiation.

#### Creation & Operations

```python
import minitorch

# Create Scalars with names for easier debugging
v1 = minitorch.Scalar(1.5, name="v1")
v2 = minitorch.Scalar(2.5, name="v2")

# Basic Arithmetic
v3 = v1 * v2 + v1
print(f"Result: {v3.data}")  # Access the underlying float value
```

#### Backpropagation

To compute derivatives, call `.backward()` on the final scalar.

```python
v3.backward()

print(f"d(v3)/d(v1) = {v1.derivative}")
print(f"d(v3)/d(v2) = {v2.derivative}")
```

### Tensor (Matrices)

`minitorch.Tensor` is a multidimensional array (like NumPy or PyTorch tensors) designed for efficiency and broadcasting.

#### Creation

```python
# From a list
t = minitorch.tensor([[1, 2], [3, 4]])

# Random tensor (0-1)
r = minitorch.rand((2, 3))

# Zeros or Ones
z = minitorch.zeros((2, 2))
```

#### Shapes and Views

Reshaping tensors is a common operation. MiniTorch uses `.view()` and `.permute()`.

```python
t = minitorch.rand((2, 3, 4))

# Change shape (must preserve total number of elements)
t_view = t.view(6, 4)

# Permute dimensions (e.g., transpose)
t_perm = t.permute(2, 0, 1)  # (4, 2, 3)
```

#### Matrix Multiplication

Use the `@` operator or `.matmul()` for matrix multiplication.

```python
A = minitorch.rand((3, 4))
B = minitorch.rand((4, 5))
C = A @ B  # Shape: (3, 5)
```

#### Backpropagation with Tensors

Backpropagation on a tensor requires reducing it to a scalar (e.g., sum or mean) or providing an upstream gradient.

```python
t = minitorch.tensor([1.0, 2.0, 3.0])
t.requires_grad_(True)

# Reduce to scalar then backward
loss = (t * 2).sum()
loss.backward()

print(t.grad)  # Should be [2.0, 2.0, 2.0]
```

### Optimizers

`minitorch.SGD` (Stochastic Gradient Descent) is used to update parameters based on computed gradients.

```python
from minitorch import SGD

# Assuming 'model' is a Module with parameters
optimizer = SGD(model.parameters(), lr=0.01)

# Inside training loop:
optimizer.zero_grad()  # 1. Clear old gradients
# ... forward pass ...
# ... backward pass ...
optimizer.step()       # 2. Update parameters
```

---

## Building Neural Networks

MiniTorch provides a `Module` system similar to PyTorch for organizing code.

### minitorch.Module

A `Module` can hold `Parameters` (trainable weights) and other `Modules`.

```python
class MyLayer(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # Initialize weights (using valid helper or direct Parameter creation)
        # Note: In practice, use a helper to initialize random weights
        self.weights = minitorch.Parameter(minitorch.rand((in_size, out_size)))
        self.bias = minitorch.Parameter(minitorch.zeros((out_size,)))

    def forward(self, x):
        return x @ self.weights.value + self.bias.value
```

### The Training Loop

A standard training loop involves:

1.  **Forward Pass**: Compute prediction.
2.  **Loss Computation**: Calculate error.
3.  **Backward Pass**: Compute gradients.
4.  **Optimizer Step**: Update weights.

```python
def train(model, X, y, epochs=100):
    optim = minitorch.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        optim.zero_grad()
        
        # 1. Forward
        out = model.forward(X)
        
        # 2. Loss (e.g., Mean Squared Error)
        loss = ((out - y) ** 2).sum()
        
        # 3. Backward
        loss.backward()
        
        # 4. Update
        optim.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
```

---

## Advanced Examples

### 1. Linear Regression

Fitting a line $y = wx + b$.

```python
import minitorch
import random

# Generate synthetic data
N = 100
X = minitorch.zeros((N, 1))
y = minitorch.zeros((N, 1))
for i in range(N):
    x_val = random.random()
    X[i, 0] = x_val
    y[i, 0] = 2.5 * x_val + 1.0 + (random.random() - 0.5) * 0.1

class LinearRegression(minitorch.Module):
    def __init__(self):
        super().__init__()
        self.w = minitorch.Parameter(minitorch.tensor([0.0]))
        self.b = minitorch.Parameter(minitorch.tensor([0.0]))

    def forward(self, x):
        return x * self.w.value + self.b.value

model = LinearRegression()
# ... use training loop above ...
```

### 2. Convolutional Neural Network (CNN)

Using 1D Convolution for sequence processing (e.g., Sentiment Analysis).

```python
class SimpleCNN(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Weights: (out_channels, in_channels, kernel_width)
        self.weights = minitorch.Parameter(
            minitorch.rand((out_channels, in_channels, kernel_size))
        )
        self.bias = minitorch.Parameter(minitorch.zeros((out_channels,)))

    def forward(self, x):
        # x: (batch, in_channels, length)
        # 1D Convolution
        conv = minitorch.conv1d(x, self.weights.value) + self.bias.value.view(1, -1, 1)
        
        # Activation (ReLU)
        relu = conv.relu()
        
        # Max Pooling (conceptually simple max over time)
        # Max over the last dimension (time/length)
        pooled = minitorch.max(relu, 2)
        return pooled
```

### 3. Recurrent Neural Network (RNN)

Implementing a simple RNN loop.

```python
class RNNCell(minitorch.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_ih = minitorch.Parameter(minitorch.rand((input_size, hidden_size)))
        self.W_hh = minitorch.Parameter(minitorch.rand((hidden_size, hidden_size)))
        self.bias = minitorch.Parameter(minitorch.zeros((hidden_size,)))

    def forward(self, input, hidden):
        # h' = tanh(x @ W_ih + h @ W_hh + b)
        # minitorch may not have tanh, using sigmoid/relu as proxy
        pre_activation = (input @ self.W_ih.value) + (hidden @ self.W_hh.value) + self.bias.value
        return pre_activation.sigmoid()

class SimpleRNN(minitorch.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, inputs):
        # inputs: (batch, seq_len, input_size)
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        
        # Initialize hidden state
        h = minitorch.zeros((batch_size, self.hidden_size))
        
        outputs = []
        for t in range(seq_len):
            # Extract time step t: (batch, input_size)
            # Note: requires careful slicing/viewing in minitorch
            # Here assuming simplified input handling
            xt = inputs[:, t, :] 
            h = self.cell(xt, h)
            outputs.append(h)
            
        return outputs[-1] # Return last hidden state
```

---

## Running Tests

Verify your installation and code using the test suite.

```bash
# Run all tests
bash run_tests.sh

# Run specific module tests
pytest -m task1_1
```

## Project Examples

Check the `project/` directory for full implementations:

-   `project/run_mnist.py`: Digit classification (MLP).
-   `project/run_sentiment.py`: Sentiment analysis (CNN).
-   `project/run_scalar.py` & `project/run_tensor.py`: Basic networks.
