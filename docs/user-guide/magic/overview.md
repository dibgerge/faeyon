# Magic System Overview

The Magic System is one of Faeyon's core features, providing powerful abstractions for delayed evaluation and lazy operations.

## Key Concepts

### X - Delayed Operations

`X` is a placeholder that allows you to create delayed operations on tensors and modules:

```python
from faeyon import X

# Indexing
pooling = X[..., 0, :]

# Method calls
normalize = X.mean(dim=-1, keepdim=True)

# Chaining
op = X.transpose(0, 1) >> X.reshape(-1, 10)
```

### Delayable

`Delayable` is an abstract base class for objects that can be evaluated lazily:

```python
from faeyon.magic.spells import Delayable

class MyDelayable(Delayable):
    def _resolve(self, data):
        # Resolve the delayable with data
        return processed_data
```

### Faek

`Faek` enables magic functionality on PyTorch modules by monkey-patching `nn.Module`:

```python
from faeyon import faek

# Faek is enabled by default, but you can control it:
with faek:
    # Your code here
    pass
```

## Learn More

- [X Documentation](x.md)
- [Delayable Documentation](delayable.md)
- [Faek Documentation](faek.md)

