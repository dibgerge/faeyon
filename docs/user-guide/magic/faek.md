# Faek

`Faek` is a singleton that enables magic functionality on PyTorch modules by monkey-patching `nn.Module`.

## Overview

By default, Faek is enabled when you import Faeyon. It adds the following capabilities to `nn.Module`:

- Delayed operations with `X`
- Arithmetic operations between modules
- Enhanced `__call__` method
- `save()` and `load()` methods
- `clone()` method

## Controlling Faek

```python
from faeyon import faek

# Faek is enabled by default, but you can control it:
faek.off()  # Disable
faek.on()   # Enable

# Or use as context manager
with faek:
    # Your code here
    pass
```

## What Faek Adds

### Arithmetic Operations

```python
from torch import nn

model1 = nn.Linear(10, 5)
model2 = nn.Linear(5, 1)

# Addition
combined = model1 + model2

# Chaining
pipeline = model1 >> model2
```

### Save/Load

```python
model = nn.Linear(10, 1)

# Save
model.save("model.yaml", save_state=True)

# Load
model = nn.Linear.from_file("model.yaml", load_state=True)
```

## Learn More

See the [API Reference](../../api/magic.md) for complete API documentation.

