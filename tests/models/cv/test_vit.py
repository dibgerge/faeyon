import pytest
import torch
from faeyon import FList
from faeyon.models import ViT
from faeyon.models.tasks import ClassifyTask


@pytest.fixture
def model():
    model = ViT(
        embed_size=4,
        heads=2,
        image_size=8,
        patch_size=4,
        num_layers=2,
        mlp_size=4,
        task=ClassifyTask(num_hidden=4, pooling=0, num_labels=1000),
    )
    model.eval()
    return model


@pytest.fixture
def x():
    return torch.randn(2, 3, 8, 8)


class TestViT:
    def test_vit_forward(self, model, x):
        out = model(x)
        assert out.shape == (2, 1000)        

    def test_vit_forward_hidden_states(self, model, x):
        out = model(x, keep_hidden=True)
        hidden = +model.fstate.hidden
        assert isinstance(model.fstate.hidden, FList)
        assert len(hidden) == 3

        x2 = torch.randn(2, 3, 8, 8)
        out2 = model(x2, keep_hidden=True)
        hidden2 = +model.fstate.hidden
        assert len(hidden2) == 3

        # the state should reset and update with each forward pass
        for t1, t2 in zip(hidden, hidden2):
            with pytest.raises(AssertionError):
                torch.testing.assert_close(t1, t2)

        out3 = model(x, keep_hidden=True)
        hidden3 = +model.fstate.hidden
        assert len(hidden3) == 3

        for t1, t2 in zip(hidden, hidden3):
            torch.testing.assert_close(t1, t2)

    def test_vit_forward_attn_weights(self, model, x):
        out = model(x, keep_attn_weights=True)
        attn_weights = +model.fstate.attn_weights
        assert len(model.fstate.attn_weights) == 2
        assert attn_weights[0].shape == (2, 5, 5)

    def test_vit_forward_keep(self, model, x):
        out = model(x, keep_attn_weights=True, keep_hidden=True)
        assert len(model.fstate.attn_weights) == 2
        assert len(model.fstate.hidden) == 3

    def test_head_mask(self, model, x):
        head_mask = torch.tensor([True, False])
        out = model(x, head_mask=head_mask)
        assert out.shape == (2, 1000)
