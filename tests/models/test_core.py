from faeyon.models.core import ModelConfigs, configs
from torch import nn


class TestModelConfigs:
    def test_singleton(self):
        assert configs is ModelConfigs()

    def test_load_files(self):
        assert len(configs.files) > 0

    def test_getitem(self):
        assert isinstance(configs["vit-base-patch16-224"], dict)

    def test_load_model(self):
        model = configs.load_model("vit-base-patch16-224")
        assert isinstance(model, nn.Module)
