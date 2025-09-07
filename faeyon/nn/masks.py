import torch
from torch import nn
from typing import Optional


class TokenizedMask(nn.Module):
    """
    Given an input, this applies the following mask:
    x * (1.0 - mask) + tokens * mask
    """
    tokens: Optional[nn.Parameter]

    def __init__(self, embedding_dim: int, enabled: bool = True) -> None:
        super().__init__()
        if enabled:
            self.tokens = nn.Parameter(torch.randn(embedding_dim))
        else:
            self.tokens = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:        
        """
        Parameters
        ----------
        x: 
            Tensor of shape (B, E, *size)

        mask: 
            Indicates which locations are masked (True) and which aren't (False).
            Should be of shape `(B, *size)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, E, *size)
        """
        if self.tokens is None and mask is not None:
            raise ValueError(
                "Mask token is not initialized. Please set `enabled=True` "
                "when initializing {self.__class__.__name__}."

            )
        if mask is None or self.tokens is None:
            return x

        batch_size, _, *size = x.shape
        tokens = self.tokens.view(1, -1, *[1]*len(size)).expand(batch_size, -1, *size)
        float_mask = mask.unsqueeze(1).type_as(x)
        return x * (1.0 - float_mask) + tokens * float_mask


def head_to_attn_mask(
    head_mask: Optional[torch.BoolTensor], 
    batch_size: int,
    src_len: int, 
    tgt_len: int,
    ravel: bool = False,
    num_layers: Optional[int] = None
) -> torch.Tensor | list[None] | None:
    """
    Applies selective masks to the attention weights of a multihead attention layer. This allows 
    using masks for particular heads and reshapes the head to the correct shape based on expected 
    sequence shape.

    H: Number of heads
    L: Number of layers
    B: Batch size
    S: Source sequence length
    T: Target sequence length

    Parameters
    ----------
    head_mask: torch.Tensor
        Indicates which heads are masked and which aren't. Should be of shape `(H,)` or `(L, H)`.

    batch_size: int
        The batch size of the input tensor, will be used to expand the head mask to the 
        correct shape.
    
    src_len: int
        The length of the source sequence, will be used to expand the head mask to the 
        correct shape.
    
    tgt_len: int
        The length of the target sequence, will be used to expand the head mask to the 
        correct shape.
    
    ravel: bool
        If True, the batch size and head dimensions will be combined. This format conforms to 
        what is expected by the `MultiHeadAttention` class.
    
    num_layers: int
        The number of layers in the model, this is required if `head_mask` is of shape (H,) and 
        attention mask needs to be repeated for each layer.
    
    Returns
    -------
    list[None]
        If `head_mask` is None and `num_layers` is given. The list has `num_layers` elements, 
        all of which are `None`.
    
    None
        If `head_mask` is None and `num_layers` is not given.

    torch.Tensor
        Tensor of transformed mask with the following possible shapes:
        (L, B, H, S, T) : if number of layers if given or can be deduced from `head_mask`.
        (B, H, S, T) : If number of layers is not known.
        (L, B * H, S, T) if ravel is True and number of layers is known.
        (B * H, S, T) otherwise
    """
    if head_mask is None:
        if num_layers is None:
            return None
        return [None] * num_layers
    
    shape = head_mask.shape

    if len(shape) not in (1, 2):
        raise ValueError(
            "`head_mask` should be of shape (# heads,) or (# layers, # heads)."
        )

    if len(shape) == 1:
        num_heads = shape[0]
    else:
        num_heads = shape[1]
        if num_layers is not None and shape[0] != num_layers:
            raise ValueError(
                "Given `num_layers` does not match the first dimension of `head_mask`."
            )
        if num_layers is None:
            num_layers = shape[0]
            
    out = head_mask.view(-1, 1, num_heads, 1, 1).expand(-1, batch_size, num_heads, src_len, tgt_len)

    if len(shape) == 1:
        if num_layers is None:
            out = out.squeeze(0)
        else:
            out = out.expand(num_layers, *out.shape[1:])
    
    if ravel:
        if num_layers is not None:
            out = out.reshape(num_layers, -1, src_len, tgt_len)
        else:
            out = out.reshape(-1, src_len, tgt_len)
    
    return out
