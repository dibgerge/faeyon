# X - Delayed Operations

`X` is a special placeholder object that allows you to create delayed operations on tensors and modules.

## Basic Usage

```python
from faeyon import X

# Indexing operations
pooling = X[..., 0, :]  # Take first element along second dimension
slice_op = X[:, :10]    # Take first 10 elements along last dimension

# Method calls
mean_op = X.mean(dim=-1, keepdim=True)
sum_op = X.sum(dim=0)

# Chaining operations
complex_op = X.transpose(0, 1) >> X.reshape(-1, 10) >> X.mean(dim=0)
```

## Using X with Data

```python
import torch
from faeyon import X

x = torch.randn(32, 10, 128)

# Apply delayed operation
pooling = X[..., 0, :]
result = x >> pooling  # Shape: (32, 128)
```

## Using X with Modules

```python
from torch import nn
from faeyon import X

# Create a model with X
model = nn.Linear(10, 5) >> nn.ReLU() >> X.mean(dim=-1)
```

## Advanced Usage

X supports a wide range of operations including:

- Indexing and slicing
- Method calls
- Arithmetic operations
- Comparison operations
- Unary operations

See the [API Reference](../../api/magic.md) for complete details.

