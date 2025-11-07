import string
import pytest
import yaml

from unittest.mock import MagicMock
from faeyon.models.core import BuiltinConfigs
from faeyon.models import save, load

import torch
from torch import nn
from tests.common import BasicModel


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
            config_path.write_text(self.template.substitute(n=i, layers=i+3))
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

    def test_read(self, configs):
        model = configs.read("model1.yaml")
        assert model["model"]["name"] == "test_model_1"


def test_save(tmp_path):
    """
    Save configuration and state dictionary to a file.
    """
    model = BasicModel()
    file_path = tmp_path / "test_model.pt"
    save(model, file_path)
    assert file_path.exists()

    with open(file_path, "rb") as f:
        data = torch.load(f)

    assert "_config_" in data
    assert "_state_" in data


@pytest.mark.parametrize("save_state, embedding", [
    (True, None), 
    (False, None), 
    ("test_state.pt", None),
    (True, nn.Embedding(10, 5)),
    (False, nn.Embedding(10, 5)),
    ("test_state.pt", nn.Embedding(10, 5))
])
def test_save_yaml(tmp_path, save_state, embedding):
    """
    Save configuration and state dictionary to a yaml file, with or without a state file.
    """
    model = BasicModel(embedding=embedding)
    file_path = tmp_path / "test_model.yaml"

    if isinstance(save_state, bool):
        state_file = file_path.with_suffix(".pt")
    else:
        save_state = tmp_path / save_state
        state_file = save_state

    save(model, file_path, save_state=save_state)
    assert file_path.exists()

    if isinstance(save_state, bool):
        assert state_file.exists() == save_state
    else:
        assert state_file.exists()


def test_load_yaml(tmp_path):
    """
    Load a model from a yaml file, without a state file.
    """
    model = BasicModel(embedding=nn.Embedding(10, 5))
    file_path = tmp_path / "test_model.yaml"
    save(model, file_path, save_state=False)
    assert file_path.exists()

    loaded_model = load(file_path, load_state="test_state.pt")
    assert loaded_model.state_dict() == model.state_dict()