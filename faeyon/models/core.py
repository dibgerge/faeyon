import os
import torch
import yaml
import importlib
from torch import nn
from importlib import resources
from pathlib import Path
from typing import Optional
from faeyon.utils import Singleton


class ModelConfigs(metaclass=Singleton):
    """
    Base class for all model configurations.
    """
    _files: Optional[dict[str, Path]]

    def __init__(self) -> None:
        self._files = None

    @property
    def files(self) -> dict[str, Path]:
        if self._files is not None:
            return self._files

        self._files = {}

        traversable = resources.files(__package__).joinpath("configs")
        for base, _, files in traversable.walk():  # type: ignore[attr-defined]
            for file in files:
                key, ext = os.path.splitext(file)
                if ext not in [".yml", ".yaml"]:
                    continue

                if key in self._files:
                    raise KeyError(
                        f"File {file} already exists. You cannot have multiple config "
                        f"files with the same name."
                    )
                self._files[key] = base / file
            
        return self._files

    def __getitem__(self, name: str) -> dict:
        with self.files[name].open() as f:
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
                # _kwargs_ should be a dict of keyword arguments
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
        config = self[name]
        return self._instantiate_config(config, force_class=True)


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
