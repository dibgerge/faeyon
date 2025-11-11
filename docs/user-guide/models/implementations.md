# Model Implementations

Faeyon includes several pre-implemented models ready to use.

## Vision Transformer (ViT)

```python
from faeyon.models import ViT

model = ViT(
    image_size=224,
    patch_size=16,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
)
```

## Qwen

```python
from faeyon.models import Qwen

model = Qwen(
    vocab_size=151936,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
)
```

## SAM (Segment Anything Model)

```python
from faeyon.models import SAM

model = SAM(
    image_encoder_vit="vit_h",
    prompt_encoder_embed_dim=256,
    mask_decoder_embed_dim=256,
)
```

## Recurrent Models

```python
from faeyon.models import Recurrent

model = Recurrent(
    input_size=10,
    hidden_size=128,
    num_layers=2,
)
```

## Learn More

See the [API Reference](../../api/models.md) for complete documentation on each model.

