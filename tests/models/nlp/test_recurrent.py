import pytest
import torch

from torch import nn
from faeyon.models.nlp import Encoder, Decoder, Seq2Seq, CellType, DefaultEmbedding
from faeyon.nn import AdditiveAttention


def test_cell_type():
    """
    Tests the enum type `CellType` and makes sure it generated the correct cell type.
    """
    input_size = 5
    hidden_size = 10
    num_layers = 2
    bidirectional = False
    dropout = 0.0
    batch_first = True
    bias = True

    for cell_type in CellType:
        cell = cell_type(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias
        )
        assert isinstance(cell, cell_type.class_name)


def test_default_embedding():
    """
    Tests the `DefaultEmbedding` class and makes sure it generated the correct embedding.
    """
    vocab_size = 10000
    embedding_size = 100
    embedding = DefaultEmbedding(vocab_size, embedding_size)
    assert isinstance(embedding(), nn.Embedding)


class TestEncoder:

    @pytest.mark.parametrize(
        "embedding",
        [
            DefaultEmbedding(
                num_embeddings=100,
                embedding_dim=10
            ),
            {
                "num_embeddings": 100,
                "embedding_dim": 10
            },
            nn.Embedding(100, 10)
        ]
    )
    def test_init_various_embeddings(self, embedding):
        """
        Checks that we can specify the embedding in three different ways. 
        """
        hidden_size = 5

        encoder = Encoder(
            embedding=embedding,
            hidden_size=hidden_size,
        )

        assert encoder.vocab_size == 100
        assert encoder.input_size == 10
        assert isinstance(encoder.embedding, nn.Embedding)

    def test_init_str_cell(self):
        """
        Checks that we can specify the cell type as a string.
        """
        encoder = Encoder(
            embedding=DefaultEmbedding(100, 10),
            hidden_size=10,
            cell="lstm"
        )

        assert encoder.vocab_size == 100
        assert isinstance(encoder.model, nn.LSTM)

    @pytest.mark.parametrize("batch_size", [0, 2])
    @pytest.mark.parametrize("num_layers", [1, 3])
    @pytest.mark.parametrize("bidirectional", [True, False])
    def test_forward(self, bidirectional, num_layers, batch_size):
        """
        Test the forward pass of the encoder and make sure it returns the correct output shape.
        """
        hidden_size = 10
        tx = 7

        encoder = Encoder(
            embedding=DefaultEmbedding(100, 10),
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            cell="lstm"
        )

        if batch_size > 0:
            x = torch.randint(0, 100, (batch_size, tx))
            expected_output_shape = (batch_size, tx, encoder.output_size)
        else:
            x = torch.randint(0, 100, (tx, ))
            expected_output_shape = (tx, encoder.output_size)
        output = encoder(x)

        assert output.shape == expected_output_shape


class TestDecoder:
    @pytest.mark.parametrize("batch_size", [0, 2])
    @pytest.mark.parametrize("num_layers", [1, 3])
    @pytest.mark.parametrize("cell", ["gru", "lstm", "rnn"])
    def test_forward(self, num_layers, batch_size, cell):
        """
        Test the forward pass of the encoder and make sure it returns the correct output shape.
        """
        hidden_size = 10
        T = 7

        decoder = Decoder(
            embedding=DefaultEmbedding(100, 10),
            hidden_size=hidden_size,
            num_layers=num_layers,
            cell=cell,
            batch_first=True
        )

        if batch_size > 0:
            x = torch.randint(0, 100, (batch_size, T))
            expected_output_shape = (batch_size, T, hidden_size)
            hidden = torch.randn(num_layers, batch_size, hidden_size)
        else:
            x = torch.randint(0, 100, (T, ))
            expected_output_shape = (T, hidden_size)
            hidden = torch.randn(num_layers, hidden_size)

        if cell == "lstm":
            output = decoder(x, (hidden, hidden))
        else:
            output = decoder(x, hidden)

        assert output.shape == expected_output_shape


class TestSeq2Seq:
    def test_forward(self):
        batch_size = 2
        Tx = 7
        Ty = 6
        encoder_hidden_size = 20
        decoder_hidden_size = 15
        decoder_vocab_size = 100
        encoder_vocab_size = 200
        attention_embed_size = 5

        encoder = Encoder(
            embedding=DefaultEmbedding(encoder_vocab_size, 10),
            hidden_size=encoder_hidden_size,
            num_layers=2,
            cell="lstm",
            batch_first=True,
            bidirectional=True
        )

        decoder = Decoder(
            embedding=DefaultEmbedding(decoder_vocab_size, 10),
            hidden_size=decoder_hidden_size,
            num_layers=2,
            cell="lstm",
            batch_first=True,
        )

        attention = AdditiveAttention(
            embed_size=attention_embed_size,
            key_size=encoder.output_size,
            query_size=decoder.output_size,
            value_size=encoder.output_size
        )

        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            attention=attention
        )

        x = torch.randint(0, encoder_vocab_size, (batch_size, Tx))
        target = torch.randint(0, decoder_vocab_size, (batch_size, Ty))
        output, alpha = model(x, target)
        assert output.shape == (batch_size, Ty, decoder_vocab_size)
