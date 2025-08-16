import torch
from torch import nn
from typing import Optional
from faeyon.utils import ImageSize
from faeyon.nn import (
    PosInterpEmbedding, 
    TokenizedMask, 
    FaeSequential, 
    head_to_attn_mask,
    Concat
)
from faeyon import A, X, Op, FDict, FList, Wiring


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
        inputs: torch.Tensor, 
        head_mask: Optional[torch.Tensor] = None, 
        return_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        inputs: torch.Tensor
            Shape (B, P, E)
        head_mask: Optional[torch.Tensor]
            Shape (B, P, P)
        return_weights: bool
            Whether to return the attention weights
        """
        outputs = FDict()

        result = (
            inputs 
            >> outputs["hidden_states"].if_(return_hidden_states)
            >> self.lnorm_in 
            >> self.attention(X, X, X, need_weights=return_weights)
            >> outputs["attention_weights"].if_(return_weights) @ X[1]
            >> Op(X[0] + inputs)
            >> Op(X) + (
                self.lnorm_out
                >> self.linear1
                >> self.gelu
                >> self.linear2
                >> self.dropout
            )
        )
        
        if outputs.is_empty:
            return result
        
        return result, +outputs


class ViT(nn.Module):
    """
    TODO:
    - [ ] Weight initialize
    - [x] parameter saving and adding parameters to constructor
    - [ ] checkpointing
    - [ ] Loading pre-trained model
    """
    # mask_token: Optional[nn.Parameter] = None
    
    def __init__(
        self,
        embed_size: int,
        heads: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        num_layers: int,
        mlp_size: int,
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
        self.pos_embeddings = PosInterpEmbedding(
            size=self.feature_count,
            embedding_dim=embed_size,
            interpolate="bicubic",
            non_positional=1,
            align_corners=False,
        )   
        self.blocks = FaeSequential(*num_layers * ViTBlock(
            num_heads=heads,
            embed_dim=embed_size,
            mlp_size=mlp_size,
            dropout=dropout,
            lnorm_eps=lnorm_eps,
        ))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_size, 1000)
        self.lnorm = nn.LayerNorm(embed_size, eps=lnorm_eps)
        self.concat = Concat()

    def forward(
        self,
        img: torch.Tensor,
        patch_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        return_hidden_states: bool = False,
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
        attention_weights = FList().if_(return_attention_weights)
        hidden_states = FList().if_(return_hidden_states)
        cls_token = self.cls_token.expand(img.shape[0], -1, -1)
        head_mask = Op(
            head_to_attn_mask, 
            head_mask, 
            X.shape[0], 
            X.shape[1], 
            X.shape[1], 
            num_layers=self.num_layers
        )
        out = (
            img 
            >> self.patch_embedding
            >> (self.pos_embeddings(X.shape[2:]) >> Op(X.mT)) + (
                self.mask_token(X, mask=patch_mask)
                >> Op(X.flatten(-2).mT)
                >> self.concat(cls_token, X, dim=1)
            )
            >> self.dropout
            >> hidden_states
            >> A(
                X,
                head_mask=head_mask,
                return_weights=return_attention_weights,
                return_hidden_states=return_hidden_states
            )
            >> self.blocks.reset().wire(
                X[0] if return_hidden_states or return_attention_weights else X, head_mask=Wiring.Fanout).report(
                hidden_states @ X[1]["hidden_states"], 
                attention_weights @ X[1]["attention_weights"]
            )
            >> Op(X[0]).if_(return_hidden_states or return_attention_weights)
            >> self.lnorm
            >> Op(X[..., 0, :])
            >> self.classifier
        )

        if return_hidden_states:
            return out, +hidden_states
        return out


# Define the pre-trained model configurations given by google
