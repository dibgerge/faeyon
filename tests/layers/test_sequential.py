import pytest
import torch
from torch import nn
from faeyon.layers import FaeSequential
from faeyon import X, Wiring, FaeArgs, FaeVar, FaeList, FaeDict, FaeMultiMap


@pytest.fixture
def advanced_model():
    class MyLayer(nn.Module):
        def forward(self, x, y, mask=None):
            out = x + y 
            if mask is not None:
                out *= mask
            return out, torch.ones_like(out)
            
    return FaeSequential(MyLayer(), MyLayer())


@pytest.fixture
def basic_model():
    return FaeSequential(nn.Linear(10, 4), nn.Linear(4, 1))


class TestFaeSequential:
    def test_forward(self, basic_model):
        x = torch.randn(1, 10)
        y = basic_model(x)
        assert y.shape == (1, 1)

    def test_init_error(self):
        """ Cannot initialize FaeSequentials using modules from different classes. """
        with pytest.raises(ValueError):
            model = FaeSequential(nn.Linear(10, 4), nn.Conv2d(1, 1, 3))

    def test_forward_wired(self, advanced_model):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([0.1, 0.2])
        mask = torch.tensor([1.0, 0.0])

        advanced_model.wire(x=X[0], y=X[1])
        out = advanced_model(x, y, mask=mask)

        assert isinstance(out, tuple)
        assert len(out) == 2
        torch.testing.assert_close(out[0], torch.tensor([2.1, 0.0]))
        torch.testing.assert_close(out[1], torch.tensor([1.0, 1.0]))

    def test_forward_wired_with_fanout(self, advanced_model):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        mask = torch.tensor([1.0, 0.0])

        out = (
            FaeArgs(x, y, mask=mask) 
            >> advanced_model
                .wire(x=X[0], y=Wiring.Fanout)
        )

        assert isinstance(out, tuple)
        assert len(out) == 2
        torch.testing.assert_close(out[0], torch.tensor([1.4, 0.0]))
        torch.testing.assert_close(out[1], torch.tensor([1.0, 1.0]))

    @pytest.mark.parametrize(
        "var,expected",
        [
            (FaeVar(), torch.tensor([1.4, 0.0])),
            (FaeList(), [torch.tensor([1.1, 0.0]), torch.tensor([1.4, 0.0])]),
            (FaeDict(), {"0": torch.tensor([1.1, 0.0]), "1": torch.tensor([1.4, 0.0])}),
            (FaeDict()["foo"], {"foo": torch.tensor([1.4, 0.0])}),
            (FaeMultiMap()["foo"], {"foo": [torch.tensor([1.1, 0.0]), torch.tensor([1.4, 0.0])]}),
            (FaeMultiMap(), {"0": [torch.tensor([1.1, 0.0])], "1": [torch.tensor([1.4, 0.0])]}),
        ]
    )
    def test_forward_with_reports(sef, advanced_model, var, expected):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        mask = torch.tensor([1.0, 0.0])
        out = (
            FaeArgs(x, y, mask=mask) 
            >> advanced_model
                .wire(x=X[0], y=Wiring.Fanout)
                .report(var @ X[0])
        )
        torch.testing.assert_close(+var, expected)

    def test_forward_with_pipe(self, advanced_model):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        mask = torch.tensor([1.0, 0.0])
        var = FaeVar()
        out = (
            FaeArgs(x, y, mask=mask) 
            >> advanced_model
                % FaeArgs(x=X[0], y=Wiring.Fanout)
                % (var @ X[1])
        )

        assert isinstance(out, tuple)
        assert len(out) == 2
        torch.testing.assert_close(out[0], torch.tensor([1.4, 0.0]))
        torch.testing.assert_close(out[1], torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(+var, torch.tensor([1.0, 1.0]))
