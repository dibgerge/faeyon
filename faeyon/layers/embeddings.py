from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)