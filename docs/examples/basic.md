# Basic Examples

## Simple Model

```python
import torch
from torch import nn
from faeyon import X

# Create a simple model
model = nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 1)

# Forward pass
x = torch.randn(32, 10)
output = model(x)
```

## Using X

```python
from faeyon import X

# Create delayed operations
pooling = X[..., 0, :]
normalize = X.mean(dim=-1, keepdim=True)

# Apply to data
x = torch.randn(32, 10, 128)
result = x >> pooling >> normalize
```

## Saving and Loading

```python
from faeyon.io import save, load

# Save
model.save("model.yaml", save_state=True)

# Load
model = load("model.yaml", load_state=True)
```

