# Training Overview

Faeyon provides comprehensive training utilities for PyTorch models.

## Trainer

The `Trainer` class provides a high-level interface for training:

```python
from faeyon.training import Trainer
from faeyon.metrics import Accuracy

trainer = Trainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=nn.CrossEntropyLoss(),
    metrics=[Accuracy()],
)

trainer.fit(train_loader, val_loader, epochs=10)
```

## Recipes

Recipes provide pre-configured training setups:

```python
from faeyon.training import Recipe

recipe = Recipe.from_file("recipe.yaml")
recipe.train()
```

See [Recipes](recipes.md) for details.

## Distributed Training

Faeyon supports distributed training:

```python
from faeyon.training import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    # ... other args
)
```

See [Distributed Training](distributed.md) for details.

## Learn More

- [Recipes](recipes.md)
- [Distributed Training](distributed.md)
- [API Reference](../../api/training.md)

