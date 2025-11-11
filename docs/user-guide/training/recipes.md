# Training Recipes

Recipes provide pre-configured training setups that can be saved and loaded.

## Creating Recipes

```python
from faeyon.training import Recipe

recipe = Recipe(
    model=model,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=nn.CrossEntropyLoss(),
    # ... other configuration
)
```

## Saving and Loading

```python
# Save recipe
recipe.save("recipe.yaml")

# Load recipe
recipe = Recipe.from_file("recipe.yaml")
```

## Running Recipes

```python
# Train with recipe
recipe.train(train_loader, val_loader, epochs=10)
```

## Learn More

See the [API Reference](../../api/training.md) for complete documentation.

