# Quick Start

This guide will help you get started with Faeyon in just a few minutes.

## Basic Usage

### Creating Models

Faeyon extends PyTorch's `nn.Module` with additional functionality:

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

### Using X for Delayed Operations

The `X` object allows you to create delayed operations:

```python
from faeyon import X

# Create a delayed operation
pooling = X[..., 0, :]  # Take first element along second dimension

# Apply to data
x = torch.randn(32, 10, 128)
result = x >> pooling
```

### Saving and Loading Models

```python
from faeyon.io import save, load

# Save a model
model.save("model.yaml", save_state=True)

# Load a model
model = load("model.yaml", load_state=True)
```

### Training Example

```python
from faeyon.training import Trainer
from faeyon.metrics import Accuracy

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=nn.MSELoss(),
    metrics=[Accuracy()],
)

# Train
trainer.fit(train_loader, val_loader, epochs=10)
```

## Next Steps

- Learn about the [Magic System](user-guide/magic/overview.md)
- Explore [Model I/O](user-guide/models/io.md)
- Check out [Training Recipes](user-guide/training/recipes.md)

