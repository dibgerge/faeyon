# Faeyon

A PyTorch library for building and training neural networks with a focus on flexibility, expressiveness, and ease of use.

## Features

- **Magic System**: Powerful abstractions for delayed evaluation and lazy operations
- **Model I/O**: Easy saving and loading of models with YAML configuration
- **Training**: Comprehensive training utilities with distributed support
- **Metrics**: Flexible metric system for tracking training progress
- **Neural Networks**: Extended PyTorch modules with additional functionality

## Quick Example

```python
import torch
from faeyon import X
from torch import nn

# Create a model with lazy operations
model = nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 1)

# Use X for delayed operations
x = torch.randn(32, 10)
result = x >> model(X)
```

## Installation

```bash
pip install faeyon
```

## Getting Started

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)

## Documentation

Browse the documentation to learn more about:

- [Magic System](user-guide/magic/overview.md) - Delayed evaluation and lazy operations
- [Models](user-guide/models/overview.md) - Model creation and I/O
- [Training](user-guide/training/overview.md) - Training utilities and recipes
- [Metrics](user-guide/metrics/overview.md) - Metric tracking
- [Neural Networks](user-guide/nn/overview.md) - Extended PyTorch modules

## License

See [LICENSE](../LICENSE) for details.

