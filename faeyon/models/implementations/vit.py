import torch
from torch import nn
from typing import Optional
from faeyon.enums import ImageSize
from faeyon.nn import (
    InterpolatedPosEmbedding, 
    TokenizedMask, 
    FaeBlock,
    head_to_attn_mask,
    Concat
)
from faeyon import X, F    


class ViT(nn.Module):
    """
    The vision transformer model.

    Parameters
    ----------

    image_size : int | tuple[int, int]
        This is needed 

    TODO
    ----
    - [ ] Weight initialize
    - [x] parameter saving and adding parameters to constructor
    - [-] checkpointing (I don't think I need this right now, since pausing training features)
    - [x] Loading pre-trained model
    """
    def __init__(
        self,
        embed_size: int,
        heads: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        num_layers: int,
        mlp_size: int,
        pos_embedding: Optional[nn.Module] = None,
        use_patch_mask: bool = False,
        in_channels: int = 3,
        dropout: float = 0.1,
        lnorm_eps: float = 1e-12,
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.image_size = ImageSize(*image_size)
        self.patch_size = ImageSize(*patch_size)
        self.embed_size = embed_size
        self.heads = heads
        self.num_layers = num_layers

        self.feature_count = ImageSize(
            (self.image_size.height // self.patch_size.height),
            (self.image_size.width // self.patch_size.width)
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.mask_token = TokenizedMask(embed_size, enabled=use_patch_mask)
        self.pos_embeddings = InterpolatedPosEmbedding(
            size=self.feature_count,
            embeddings=embed_size,
            interpolate="bicubic",
            align_corners=False,
        )

        self.blocks = FaeBlock({
            "lnorm_in": nn.LayerNorm(embed_size, eps=lnorm_eps),
            "lnorm_out": nn.LayerNorm(embed_size, eps=lnorm_eps),
            "linear1": nn.Linear(embed_size, mlp_size),
            "linear2": nn.Linear(mlp_size, embed_size),
            "dropout": nn.Dropout(dropout),
            "attention": nn.MultiheadAttention(
                embed_dim=embed_size,
                num_heads=heads,
                batch_first=True,
                dropout=dropout,
            ),
            "gelu": nn.GELU(),
        }, repeats=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(embed_size, eps=lnorm_eps)
        self.concat = Concat()

    def forward(
        self,
        img: torch.Tensor,
        patch_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        keep_attn_weights: bool = False,
        keep_hidden: bool = False,
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
        
        head_mask: torch.Tensor
            Indicates which heads are masked. Should be of shape `(num_heads,)` 
            or `(num_hidden_layers, num_heads)`.
        """
        cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        attn_mask = F(
            head_to_attn_mask, 
            head_mask, 
            X.shape[0], 
            X.shape[1], 
            X.shape[1],
            ravel=True,
            num_layers=self.num_layers
        )

        return (
            img 
            >> self.patch_embedding
            >> self.pos_embeddings(X.shape[2:]) + self.mask_token(X, mask=patch_mask)
            >> X.flatten(-2).mT
            >> self.concat(cls_token, X, dim=1)
            >> self.dropout
            >> self.fstate.hidden.if_(keep_hidden)
            >> (
                F(X) + (
                    self.blocks.lnorm_in
                    << self.blocks.attention(
                        X, X, X, 
                        # TODO: Fix this
                        #attn_mask=W.Fanout(attn_mask), 
                        need_weights=keep_attn_weights
                    )
                    << self.fstate.attn_weights.if_(keep_attn_weights) @ X[1]
                    << X[0]
                )
                << F(X) + (
                    self.blocks.lnorm_out
                    << self.blocks.linear1
                    << self.blocks.gelu
                    << self.blocks.linear2  
                    << self.blocks.dropout
                )
                << self.fstate.hidden.if_(keep_hidden)
            )
            >> self.lnorm
        )
