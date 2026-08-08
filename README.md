# faeyon

Consider a simple model like this:

```python
from torch import nn
from faeyon import faek

with faek:  # or faek.on() to enable process-wide
    model = nn.Linear(10, 5) >> nn.Linear(5, 3)
```

Importing faeyon does not change PyTorch's behavior. The `nn.Module` interception
(`faek`) is opt-in: enable it explicitly with `faek.on()`, or scope it to the code that
builds expressions with `with faek:`. Evaluating or materializing an already-built tree
(`data | expr`, `FaeModule`, `lower`) does not require it.
