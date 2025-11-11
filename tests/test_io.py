import string
import pytest
import yaml

from unittest.mock import MagicMock
from faeyon.io import BuiltinConfigs, save, load

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
            model = BasicModel(num_hidden=2 * (i + 1))
            config_path = tmp_path / f"model{i}.yaml"
            model.save(config_path, save_state=True)
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
        model = configs.read("model1")
        assert model["_kwargs_"]["num_hidden"] == 4

    def test_valid(self, configs):
        assert configs.valid("model1")
        assert not configs.valid("model1.yaml")
        assert not configs.valid("/a/b/model1.yaml")
        assert not configs.valid("../model1")
        assert not configs.valid("s3://model1")
        assert not configs.valid("/a/b/c")

    def test_load(self):
        """ Loading a model from a builtin config file."""
        expected = BasicModel(num_hidden=4)
        model = load("model1", True)
        assert isinstance(model, BasicModel)
        assert model.is_similar(expected)


@pytest.mark.parametrize("save_state", [True, False])
def test_save_pt(tmp_path, save_state):
    """
    Save configuration and state dictionary to a file.
    """
    model = BasicModel()
    file_name = tmp_path / "test_model.pt"
    save(model, file_name=file_name, save_state=save_state)
    assert file_name.exists()

    # Make sure the saved data format is as expected
    with open(file_name, "rb") as f:
        data = torch.load(f)

    assert "_config_" in data

    if save_state:
        assert "_state_" in data
    else:
        assert "_state_" not in data


def test_save_pt_error(tmp_path):
    """
    When saving using pytorch, the state and config will both be saved to the same file. We
    are not allowed to give a string for the `save_state` argument.
    """
    model = BasicModel()
    file_name = tmp_path / "test_model.pt"
    with pytest.raises(ValueError):
        save(model, file_name=file_name, save_state=tmp_path / "test_state.pt")


@pytest.mark.parametrize("save_state", [True, False, "test_state.pt"])
@pytest.mark.parametrize("embedding", [None, nn.Embedding(10, 5)])
def test_save_yaml(tmp_path, save_state, embedding):
    """
    Save configuration and state dictionary to a yaml file, with or without a state file.
    """

    model = BasicModel(embedding=embedding)
    file_path = tmp_path / "test_model.yaml"

    if not isinstance(save_state, bool):
        save_state = tmp_path / save_state

    save(model, file_path, save_state=save_state)
    assert file_path.exists()

    if isinstance(save_state, bool):
        assert file_path.with_suffix(".pt").exists() == save_state
    else:
        assert save_state.exists()


@pytest.mark.parametrize("load_state", [True, False])
@pytest.mark.parametrize("file_name", ["test_model.yaml", "test_model.pt"])
def test_load(tmp_path, load_state, file_name):
    """
    Load a model from a yaml file, without a state file.
    """
    model = BasicModel(embedding=nn.Embedding(10, 5))
    file_name = tmp_path / file_name
    save(model, file_name, save_state=load_state)
    assert file_name.exists()

    loaded_model = load(file_name, load_state)
    assert isinstance(loaded_model, BasicModel)

    if load_state:
        assert loaded_model == model
    else:
        assert loaded_model.is_similar(model)


