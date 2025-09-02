from torch import nn
from faeyon.nn import RotaryEmbedding, FaeBlock, MultiHeadAttention


class Qwen:
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int, 
        padding_idx: int
    ) -> None:
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)

        self.decoder = FaeBlock({
            "attention": MultiHeadAttention(
                embed_size=hidden_size,
                num_heads=8,
                key_size=hidden_size,
                query_size=hidden_size,
                value_size=hidden_size,
                dropout=0.1,
                bias=False,
                fq=RotaryEmbedding(),
                fk=RotaryEmbedding(),
            ),
            "ln_out": nn.LayerNorm(hidden_size),
            "mlp": nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ),
        })

    def forward(self, x: torch.LongTensor, pos: torch.LongTensor) -> torch.Tensor:
         return (
            x 
            >> self.embedding
         )
