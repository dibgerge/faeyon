import string
import pytest
import yaml

from unittest.mock import MagicMock
from faeyon.models.core import BuiltinConfigs


class TestBuiltinConfigs:
    """
    To test BuiltinConfigs, a mock config directory is built with 3 model config files.
    """
    def setup_class(self):
        self.num_files = 3
        template = {
            "model": {
                "name": "test_model_$n",
                "layers": "$layers",
                "hiddensize": 100
            }
        }
        self.template = string.Template(yaml.safe_dump(template))

    @pytest.fixture(autouse=True)
    def config_base_path(self, tmp_path, monkeypatch):
        tmp_path.mkdir(parents=True, exist_ok=True)

        config_files = []
        for i in range(self.num_files):
            config_path = tmp_path / f"model{i}.yaml"
            config_path.write_text(self.template.substitute(n=i+1, layers=i+3))
            config_files.append(config_path)

        mock_traversable = MagicMock()
        mock_traversable.joinpath.return_value = tmp_path
        mock_traversable.walk.return_value = [(tmp_path, [], config_files)]
        monkeypatch.setattr("importlib.resources.files", lambda x: mock_traversable)

    @pytest.fixture(autouse=True)
    def configs(self):
        return BuiltinConfigs()

    def test_ls(self, configs):
        # Should make sure the returned paths are without the base path (i.e. just config keys)
        assert len(list(configs.ls())) == self.num_files
        assert 0

    def test_read(self, configs):
        assert configs.read("vit/vit-base-patch16-224.yaml") == "vit/vit-base-patch16-224.yaml"

    def test_open(self):
        configs = BuiltinConfigs()
        assert configs.open("vit/vit-base-patch16-224.yaml") == "vit/vit-base-patch16-224.yaml"