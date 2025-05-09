from flax import nnx
import jax.numpy as jnp


class ViTBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        mlp_size: int,
        rngs: nnx.Rngs,
    ):
        self.lnorm_in = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.lnorm_out = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.linear1 = nnx.Linear(hidden_size, mlp_size, rngs=rngs)
        self.linear2 = nnx.Linear(mlp_size, hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            rngs=rngs,
            decode=False
        )

    def __call__(self, x):
        in_x = x
        x = self.lnorm_in(in_x)
        x = self.attention(x)
        x2 = x + in_x
        x = self.lnorm_out(x2)
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + x2
        return x


class ViT(nnx.Module):
    def __init__(
        self,
        heads: int,
        image_height: int,
        image_width: int,
        patch_size: int,
        layers: int,
        hidden_size: int,
        mlp_size: int,
        rngs: nnx.Rngs
    ):
        self.layers = [
            ViTBlock(
                num_heads=heads,
                hidden_size=hidden_size,
                mlp_size=mlp_size,
                rngs=rngs
            )
            for _ in range(layers)
        ]

        self.patch_embedding = nnx.Conv(
            in_features=3,
            out_features=hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs
        )

        num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.cls_token = nnx.Param(
            nnx.initializers.normal()(rngs.params(), (1, 1, hidden_size))
        )
        self.positional_embedding = nnx.Param(
            nnx.initializers.normal()(rngs.params(), (num_patches + 1, hidden_size))
        )
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
        self.classifier = nnx.Linear(hidden_size, 1000, rngs=rngs)

        self.deterministic = False

    def __call__(self, x):
        x = self.patch_embedding(x)
        *batch, height, width, channels = x.shape

        x = x.reshape((*batch, height * width, channels))

        cls_token = jnp.tile(self.cls_token, (*batch, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.positional_embedding
        x = self.dropout(x, deterministic=self.deterministic)
        
        for layer in self.layers:
            x = layer(x)

        x = x[..., 0, :]
        x = self.classifier(x)
        return x
