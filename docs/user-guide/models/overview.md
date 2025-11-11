# Models Overview

Faeyon provides utilities for working with PyTorch models, including model I/O, implementations, and tasks.

## Model Implementations

Faeyon includes several model implementations:

- **ViT** (Vision Transformer)
- **Qwen** (Large Language Model)
- **SAM** (Segment Anything Model)
- **Recurrent** models

See [Implementations](implementations.md) for details.

## Model I/O

Faeyon supports saving and loading models with YAML configuration:

```python
from faeyon.io import save, load

# Save model
model.save("model.yaml", save_state=True)

# Load model
model = load("model.yaml", load_state=True)
```

See [Model I/O](io.md) for complete documentation.

## Tasks

Tasks are high-level abstractions for common model tasks:

```python
from faeyon.models import ClassifyTask

task = ClassifyTask(num_hidden=768, num_labels=10)
```

## Learn More

- [Model I/O](io.md)
- [Implementations](implementations.md)

