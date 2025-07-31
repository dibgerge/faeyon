import enum
import torch

from dataclasses import dataclass, asdict
from typing import Optional

from faeyon.layers import Attention
from torch import nn


class CellType(enum.Enum):

    def __init__(self, value: str, class_name: nn.RNNBase) -> None:
        self.class_name = class_name

    def __new__(cls, value: str, class_name: nn.RNNBase) -> 'CellType':
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    GRU = ("gru", nn.GRU)
    LSTM = ("lstm", nn.LSTM)
    RNN = ("rnn", nn.RNN)

    def __call__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        batch_first: bool,
        bias: bool
    ) -> nn.RNNBase:
        return self.class_name(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias
        )


@dataclass
class DefaultEmbedding:
    num_embeddings: int
    embedding_dim: int

    def __call__(self) -> nn.Embedding:
        return nn.Embedding(**asdict(self))


class Encoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding | DefaultEmbedding | dict,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        batch_first: bool = True,
        bias: bool = True,
        cell: CellType | str = CellType.GRU
    ):
        super().__init__()

        if isinstance(embedding, nn.Embedding):
            self.embedding = embedding
        else:
            if isinstance(embedding, dict):
                embedding = DefaultEmbedding(**embedding)

            self.embedding = embedding()

        self.vocab_size = self.embedding.num_embeddings
        self.input_size = self.embedding.embedding_dim
        self.num_layers = num_layers

        if hidden_size is None:
            self.hidden_size = self.input_size
        else:
            self.hidden_size = hidden_size

        self.cell_type = CellType(cell)  # type: ignore

        self.model = self.cell_type(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias
        )

    def forward(
        self, x: torch.Tensor,
        hidden: Optional[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        if hidden is not None and self.cell_type == CellType.LSTM:
            if isinstance(hidden, torch.Tensor):
                hidden = (hidden, torch.zeros_like(hidden))

        x = self.embedding(x)
        output, _ = self.model(x, hidden)
        return output

    @property
    def output_size(self):
        if self.model.bidirectional:
            return self.hidden_size * 2
        return self.hidden_size


class Decoder(Encoder):
    def __init__(
        self,
        embedding: nn.Embedding | DefaultEmbedding | dict,
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        bias: bool = True,
        cell: CellType | str = CellType.GRU
    ) -> None:
        super().__init__(
            embedding=embedding,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            cell=cell,
            bidirectional=False
        )


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        attention: Attention
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

        # This is not in the original paper, but we use here to keep attention / decoder separated
        self.encoder_proj = nn.Linear(encoder.output_size, decoder.hidden_size)
        self.hidden_proj = nn.Linear(
            encoder.output_size + decoder.output_size,
            attention.embed_size
        )
        self.out_proj = nn.Linear(attention.embed_size, decoder.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence length, input_size)

        hidden : torch.Tensor
            Hidden state tensor of shape (batch_size, hidden_size)   

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch_size, output_size)

        attention : torch.Tensor
            Attention tensor of shape (batch_size, sequence length, embed_size)
        """
        ye = self.encoder(x, hidden)
        ye_proj = self.encoder_proj(ye[..., -1, :])

        hidden = torch.repeat_interleave(ye_proj[None], self.decoder.num_layers, dim=0)
        yd = self.decoder(target, hidden)
        alpha = self.attention(key=ye, query=yd, value=ye)

        ht = torch.tanh(self.hidden_proj(torch.cat([yd, alpha], dim=-1)))
        out = self.out_proj(ht)
        return out, alpha
