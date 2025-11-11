# Distributed Training

Faeyon supports distributed training across multiple GPUs and nodes.

## Basic Usage

```python
from faeyon.training import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=nn.CrossEntropyLoss(),
)

trainer.fit(train_loader, val_loader, epochs=10)
```

## Multi-Node Training

For multi-node training, see the [Multi-Node Training Guide](../../../scripts/MULTI_NODE_TRAINING.md).

## Learn More

See the [API Reference](../../api/training.md) for complete documentation.

