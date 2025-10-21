import os
import re
import torch
import yaml
import importlib
from torch import nn
from importlib import resources
from pathlib import Path
from typing import Optional, IO, Generator
from faeyon.utils import Singleton


class BuiltinConfigs:
    """
    Class to lazily load the builtin model configurations. Provides a simple read-only interface 
    to load the builtin model configurations.
    """
    # _files: Optional[dict[str, Path]] = None

    def __init__(self) -> None:
        self.traversable = resources.files(__package__).joinpath("configs")

    def open(self, name: str) -> IO[str]:
        return self.traversable.joinpath(name).open("r")

    def read(self, name: str) -> str:
        with self.open(name) as f:
            return f.read()

    def ls(self) -> Generator[str, None, None]:
        for base, _, files in self.traversable.walk():  # type: ignore[attr-defined]
            for file in files:
                key, ext = os.path.splitext(file)
                if ext not in [".yml", ".yaml"]:
                    continue

                yield base / file

    # @property
    # def files(self) -> dict[str, Path]:
    #     if self._files is not None:
    #         return self._files

    #     self._files = {}

        

            
    #     return self._files
    

class ModelConfigs(metaclass=Singleton):
    """
    Base class for all model configurations.
    """
   

    def __getitem__(self, name: str) -> dict:
        # Check for remote URLs (e.g., s3://, etc.)
        if re.match(r"^\w+://", name):
            # TODO: Make sure fsspec is installed
            import fsspec            
            file_obj = fsspec.open(name, "r")
        # Check if it's a path (has '/' or is a file in current directory)
        elif "/" in name or os.path.isfile(name):
            path = Path(name)
            if not path.is_file():
                raise FileNotFoundError(f"Could not find the config file at {name}.")
            file_object = path.open("r")
        # Else, load from the package configs
        else:
            file_object = self.files[name].open()
        
        with file_object as f:
            return yaml.safe_load(f)

    def _instantiate_config(self, cfg, force_class: bool = False):
        if force_class:
            if not isinstance(cfg, dict) or "_target_" not in cfg:
                raise ValueError("Config must be a dict with a _target_ key.")
                
        if isinstance(cfg, dict):
            try:
                target = cfg.pop("_target_")
            except KeyError:
                return {k: self._instantiate_config(v) for k, v in cfg.items()}

            args, kwargs = [], {}

            if "_args_" in cfg:
                if not isinstance(cfg["_args_"], list):
                    raise ValueError("'_args_' must be a list of arguments.")
                args = [self._instantiate_config(arg) for arg in cfg.pop("_args_")]

            if "_kwargs_" in cfg:
                if not isinstance(cfg["_kwargs_"], dict):
                    raise ValueError("'_kwargs_' must be a dict of keyword arguments.")

                for k, v in cfg.pop("_kwargs_").items():
                    kwargs[k] = self._instantiate_config(v)

            # All remaining keys are treated as keyword arguments
            for k, v in cfg.items():
                kwargs[k] = self._instantiate_config(v)

            mod_name, cls_name = target.rsplit(".", 1)
            mod = importlib.import_module(mod_name)

            cls = getattr(mod, cls_name)
            return cls(*args, **kwargs)
        elif isinstance(cfg, list):
            return [self._instantiate_config(item) for item in cfg]
        else:
            return cfg

    def load_model(
        self,
        name: str,
        pretrained: bool = False
    ) -> nn.Module:

        model = self._instantiate_config(self[name], force_class=True)

        if pretrained:
            # TODO
            # Find model in the hub, download it and load the weights
            from huggingface_hub import hf_hub_download, snapshot_download
            model.load_pretrained(name)

        return model

    def save_model(self, model: nn.Module, name: str, path: str) -> None:
        """
        Save the model to a file.
        """
        model.save_pretrained(path)
        model.push_to_hub(name)


class FaeModel(nn.Module):
    """
    Base class for all Faeyon model, which allows to save and load the model.
    """
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        """
        name = self.__class__.name
        state = self.state_dict()

        # Need to save configuration of the model, 
    
        output = {
            "model":{
                "name": name,
                "state": state,
            }
        }

        torch.save(output, path)
        from huggingface_hub import snapshot_download, hf_hub_download

    def init_weights(self, pretrained: bool = False) -> None:
        pass


configs = ModelConfigs()
load_model = configs.load_model
