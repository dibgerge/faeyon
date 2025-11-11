# Model I/O

Faeyon provides flexible model saving and loading with YAML configuration support.

## Saving Models

### Basic Save

```python
from faeyon.io import save

# Save to YAML
model.save("model.yaml", save_state=True)

# Save to PyTorch file
model.save("model.pt", save_state=True)
```

### Save Options

```python
# Save without state
model.save("model.yaml", save_state=False)

# Custom state file name
model.save("model.yaml", save_state="weights.pt")

# Trust code (allows arbitrary code in config)
model.save("model.yaml", trust_code=True)
```

## Loading Models

### Basic Load

```python
from faeyon.io import load

# Load from YAML
model = load("model.yaml", load_state=True)

# Load from PyTorch file
model = load("model.pt", load_state=True)
```

### Load Options

```python
# Load without state
model = load("model.yaml", load_state=False)

# Load with custom state file
model = load("model.yaml", load_state="custom_weights.pt")

# Load from remote
model = load("s3://bucket/model.yaml", cache=True)
```

## File Formats

### YAML Format

YAML files contain the model configuration:

```yaml
_target_: torch.nn.Linear
_args_: []
_kwargs_:
  in_features: 10
  out_features: 5
_meta_:
  state_file: model.pt
```

### PyTorch Format

PyTorch files contain both config and state:

```python
{
    "_config_": "...",  # YAML string
    "_state_": {...}    # State dict
}
```

## Builtin Configs

Faeyon includes builtin model configurations:

```python
from faeyon.io import load

# Load builtin config
model = load("vit/vit-base-patch16-224", load_state=True)
```

## Learn More

See the [API Reference](../../api/models.md) for complete documentation.

