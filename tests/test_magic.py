from torch import nn
from faeyon import faek


def test_faek_as_context_manager():
    with faek:
        print("he")
        model = nn.Linear(10, 10)
        print(model)
    #     assert isinstance(model, ModuleMixin)
    # assert not isinstance(model, ModuleMixin)


def test_faek_as_manual():
    faek.on()
    model = nn.Linear(10, 10)
    assert isinstance(model, ModuleMixin)
    faek.off()
    assert not isinstance(model, ModuleMixin)

