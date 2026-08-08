# Faek

`Faek` is a singleton that enables magic functionality on PyTorch modules by monkey-patching `nn.Module`.

## Overview

Faek is **opt-in**: importing Faeyon does not change PyTorch's behavior. When enabled, it adds the following capabilities to `nn.Module`:

- Delayed operations with `X`
- Arithmetic operations between modules
- Enhanced `__call__` method
- `save()` and `load()` methods
- `clone()` method

## Controlling Faek

```python
from faeyon import faek

# Enable explicitly (process-wide):
faek.on()   # Enable
faek.off()  # Disable

# Or scope it to the code that builds expressions (reentrant,
# restores the previous state on exit):
with faek:
    model = nn.Linear(10, 5) >> nn.ReLU()
```

Only *building* expressions from modules requires Faek to be on. Evaluating or
materializing an already-built tree (`data | expr`, `FaeModule`, `lower`) works with it
off, so you can build models inside `with faek:` and use them anywhere.

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

