# Training Examples

## Basic Training

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

## With Callbacks

```python
from faeyon.training import Trainer, Callback

class MyCallback(Callback):
    def on_epoch_end(self, trainer, epoch):
        print(f"Epoch {epoch} completed")

trainer = Trainer(
    model=model,
    # ... other args
    callbacks=[MyCallback()],
)
```

## Distributed Training

```python
from faeyon.training import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=nn.CrossEntropyLoss(),
)

trainer.fit(train_loader, val_loader, epochs=10)
```

