import math
import torch
from torch import nn
from typing import Optional
from faeyon.utils import ImageSize
from faeyon.layers import InterpEmbedding, TokenizedMask
from faeyon import FaeArgs, X, Op, faek


class ViTBlock(nn.Module):
    """
    #TODO Describe the Block architecture using mermaid for documentation
    """
    
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        mlp_size: int,
        dropout: float, 
        lnorm_eps: float,
    ) -> None:
        super().__init__()
        self.lnorm_in = nn.LayerNorm(embed_dim, eps=lnorm_eps)
        self.lnorm_out = nn.LayerNorm(embed_dim, eps=lnorm_eps)
        self.linear1 = nn.Linear(embed_dim, mlp_size)
        self.linear2 = nn.Linear(mlp_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.gelu = nn.GELU()

    def forward(
        self, 
        x: torch.Tensor, 
        head_mask: Optional[torch.Tensor] = None, 
        return_weights: bool = False
    ) -> torch.Tensor:
        weights = FaeVar(condition=return_weights)
        block1 = x + (
            x 
            >> self.lnorm_in 
            >> Input(FaeIn, FaeIn, FaeIn, need_weights=return_weights) 
            >> self.attention 
            >> weights(FaeIn[1])
        )

        out = block1 + (
            block1
            >> self.lnorm_out
            >> self.linear1
            >> self.gelu
            >> self.linear2
            >> self.dropout
        )

        if return_weights:
            return out, ~weights
        return out


class ViT(nn.Module):
    """
    TODO:
    - [ ] Weight initialize
    - [ ] parameter saving and adding parameters to constructor
    - [ ] checkpointing
    - [ ] Loading pre-trained model
    - [ ] Figure out how to handle the `rngs` arguments correctly and cleanly.
    """
    # mask_token: Optional[nn.Parameter] = None
    
    def __init__(
        self,
        heads: int,
        image_size: int | tuple[int, int] | ImageSize,
        patch_size: int | tuple[int, int] | ImageSize,
        layers: int,
        embed_size: int,
        mlp_size: int,
        use_patch_mask: bool = False,
        in_channels: int = 3,
        dropout: float = 0.1,
        lnorm_eps: float = 1e-12,
    ):
        self.flatten = nn.Flatten(-2)
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.image_size = ImageSize(*image_size)
        self.patch_size = ImageSize(*patch_size)
        self.embed_size = embed_size

        self.patch_count = ImageSize(
            (self.image_size.height // self.patch_size.height),
            (self.image_size.width // self.patch_size.width)
        )
        self.total_patches = math.prod(self.patch_count)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

        # if use_patch_mask:
        #     self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_size))

        self.mask_token = TokenizedMask(embed_size, enabled=use_patch_mask)

        #self.pos_embeddings = nn.Parameter(torch.randn(1, self.total_patches + 1, embed_size))
        self.pos_embeddings = InterpEmbedding(
            size=self.patch_count,
            embedding_dim=embed_size,
            interpolate="bicubic",
            align_corners=False,
        )   

        self.blocks = layers * ViTBlock(
            num_heads=heads,
            embed_dim=embed_size,
            mlp_size=mlp_size,
            dropout=dropout,
            lnorm_eps=lnorm_eps,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_size, 1000)
        self.lnorm = nn.LayerNorm(embed_size, eps=lnorm_eps)

    def interpolate_pos_encoding(self, input_size: ImageSize) -> torch.Tensor:
        """
        Interpolate the positional embeddings to match the input image size.

        Parameters
        ----------
        input_embeddings: torch.Tensor
            Input embeddings of shape `(B, P, E)`
        input_size: ImageSize
            Input image size

        Returns
        -------
        torch.Tensor
            Interpolated positional embeddings of shape `(B, P, E)`
        """
        # TODO: Check if this need to be typed as torch int when tracing 
        # (See https://github.com/huggingface/transformers/pull/33226)

        # always interpolate when tracing to ensure exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and input_size == self.image_size:
            return self.pos_embeddings

        input_patch_count = ImageSize(
            (input_size.height // self.patch_size.height),
            (input_size.width // self.patch_size.width)
        )

        patch_pos_embed = (
            self.pos_embeddings[:, 1:]
            .reshape(1, *self.patch_count, self.embed_size)
            .permute(0, 3, 1, 2)
        )

        patch_pos_embed = (
            nn.functional.interpolate(
                patch_pos_embed,
                size=input_patch_count,
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .view(1, -1, self.embed_size)
        )

        return torch.cat((self.pos_embeddings[:, :1], patch_pos_embed), dim=1)

    def forward(
        self,
        img: torch.Tensor,
        patch_mask: Optional[torch.BoolTensor] = None,
        interpolate: bool = False,
    ) -> torch.Tensor:
        """
        Let the sizes be:
            B: batch size
            P: number of patches
            E: embedding size
            C: number of input image channels
            H: input image height
            W: input image width

        Parameters
        ----------
        img: torch.Tensor
            Input image of shape `(B, C, H, W)`

        patch_mask: torch.BoolTensor
            Indicates which patches are masked (True) and which aren't (False).
            Should be of shape `(B, P)`.
        """

        faek.on()

        batch_size, channels, height, width = img.shape

        embeddings = (
            img 
            >> self.patch_embedding             # B, E, H, W
            >> (self.pos_embeddings + (
                Op(X.flatten(-2).mT) >> FaeArgs(X, mask=patch_mask)
                >> self.mask_token
            ))

        )
        embeddings = embeddings.mT # (B, E, P)

        # if patch_mask is not None:
        #     if self.mask_token is None:
        #         raise ValueError(
        #             f"Mask token is not initialized. Please set `use_patch_mask=True` "
        #             f"when initializing {self.__class__.__name__}."
        #         )
        #     mask_tokens = self.mask_token.expand(batch_size, self.total_patches, -1)
        #     mask = patch_mask.unsqueeze(-1).type_as(mask_tokens)
        #     embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        input_size = ImageSize(height, width)
        # if interpolate:
        #     pos_embeddings = self.interpolate_pos_encoding(input_size)
        # else:
        #     if input_size != self.image_size:
        #         raise ValueError(
        #             f"Input image size ({height}*{width}) doesn't match model"
        #             f" ({self.image_size.height}*{self.image_size.width}). Set `interpolate=True`"
        #             f" to interpolate the positional embeddings."
        #         )
        #     pos_embeddings = self.pos_embeddings

        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, E)
        embeddings = torch.cat([cls_token, embeddings], dim=1)  # (B, P + 1, E)


        x = (
            embeddings + (embeddings >> self.pos_embeddings)
            >> self.dropout
            >> self.blocks
            >> self.lnorm
        )

        x = x[..., 0, :]
        x = self.classifier(x)
        return x


# Define the pre-trained model configurations given by google
