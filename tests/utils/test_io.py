import pytest
from faeyon.models.core import BuiltinConfigs


class TestBuiltinConfigs:
    @pytest.fixture
    def config_file(self) -> str:
        return "vit-base-patch16-224"

    def test_ls(self):
        configs = BuiltinConfigs()
        print(list(configs.ls()))
        # assert configs.ls() == ["vit/vit-base-patch16-224.yaml"]

    def test_read(self):
        configs = BuiltinConfigs()
        assert configs.read("vit/vit-base-patch16-224.yaml") == "vit/vit-base-patch16-224.yaml"

    def test_open(self):
        configs = BuiltinConfigs()
        assert configs.open("vit/vit-base-patch16-224.yaml") == "vit/vit-base-patch16-224.yaml"
