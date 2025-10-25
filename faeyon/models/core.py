from __future__ import annotations
import os
import re
import torch
import yaml
import fsspec
import importlib
from torch import nn
from importlib import resources
from pathlib import Path
from typing import Optional, IO, Generator, Any
from faeyon.utils import Singleton
from faeyon.utils.io import cache_dir


class ModelConfig:
    def __init__(self, name: str):
        pass

    def is_builtin(self) -> bool:
        pass
    

class BuiltinConfigs:
    """
    Class to lazily load the builtin model configurations. Provides a simple read-only interface 
    to load the builtin model configurations.
    """
    # _files: Optional[dict[str, Path]] = None

    def __init__(self) -> None:
        self.traversable = resources.files(__package__).joinpath("configs")

    def open(self, name: str) -> IO[str]:
        if not self.valid(name):
            raise ValueError(
                f"Invalid builtin config path: {name}. The path should not be an absolute path, "
                f"contain . or .. sections, or start with a protocol like protocol://."
            )
        return self.traversable.joinpath(name).open("r")

    def read(self, name: str) -> dict[str, Any]:
        with self.open(name) as f:
            return yaml.safe_load(f)

    def ls(self) -> Generator[str, None, None]:
        for base, _, files in self.traversable.walk():  # type: ignore[attr-defined]
            for file in files:
                key, ext = os.path.splitext(file)
                if ext not in [".yml", ".yaml"]:
                    continue

                yield base / key

    def valid(self, name: str) -> bool:
        has_protocol = "://" in name.split("/")[0]
        is_absolute = os.path.isabs(name)
        has_dots = any(part in ('.', '..') for part in Path(name).parts)
        return not (is_absolute or has_dots or has_protocol)


def _parse_config(cfg: Any, force_class: bool = False) -> Any:
    """
    Parse a configuration section recursively and return loaded objects. 
    """
    if force_class:
        if not isinstance(cfg, dict) or "_target_" not in cfg:
            raise ValueError("Config must be a dict with a _target_ key.")

    match cfg:
        case dict():
            try:
                target = cfg.pop("_target_")
            except KeyError:
                return {k: _parse_config(v) for k, v in cfg.items()}

            args, kwargs = [], {}

            if "_args_" in cfg:
                if not isinstance(cfg["_args_"], list):
                    raise ValueError("'_args_' must be a list of arguments.")
                args = [_parse_config(arg) for arg in cfg["_args_"]]

            if "_kwargs_" in cfg:
                if not isinstance(cfg["_kwargs_"], dict):
                    raise ValueError("'_kwargs_' must be a dict of keyword arguments.")

                for k, v in cfg["_kwargs_"].items():
                    kwargs[k] = _parse_config(v)

            # All remaining keys are treated as keyword arguments
            for k, v in cfg.items():
                kwargs[k] = _parse_config(v)

            mod_name, cls_name = target.rsplit(".", 1)
            mod = importlib.import_module(mod_name)

            cls = getattr(mod, cls_name)
            return cls(*args, **kwargs) 

        case list():
            return [_parse_config(item) for item in cfg]

        case _:
            return cfg


def _read_config(name: str) -> tuple[dict[str, Any], bool]:
    """
    A given name can either be a local path, a remote URL/filesystem path, or a file in the package configs. The lookup logic is as follows:
    * If path is normalized (no . or .. segments, or trailing slash/protocol), look in current working directory. If not found, look in package configs.
    * Otherwise lookup with fsspec.


    Returns a configuration file and a boolean indicating if the file is a builtin config.
    """

    try:
        with fsspec.open(name, "r") as f:
            return yaml.safe_load(f), False
    except FileNotFoundError as e:
        config = BuiltinConfigs()
        
        if not config.valid(name):
            raise e

        return config.read(name), True


def load(
    name: str,
    pretrained: bool | str = False,
    cache: bool = True,
    **kwargs: Any
) -> nn.Module:
    cfg, is_builtin = _read_config(name)
    keys = set(cfg.keys())

    if "_model_" in cfg:
        model_cfg = cfg["_model_"]
    else:
        model_cfg = cfg

    model = _parse_config(model_cfg, force_class=True)
    if "_model_" not in keys or keys == {"_model_"}:
        return model

    if keys != {"_model_", "_optimizer_", "_dataloader_"}:
        # TODO: Implement recipe loading
        raise ValueError(f"Trying to load a recipe here...")

    if pretrained:
        # TODO
        # Find model in the hub, download it and load the weights
        #from huggingface_hub import hf_hub_download, snapshot_download

        if is_builtin:
            name = f"hf://dibgerges/faeyon/{name}"
        
        fs, remote_path = fsspec.core.url_to_fs(name, **kwargs)

        # Determine whether the protocol is remote (i.e., not 'file')
        protocol = getattr(fs, "protocol", None)
        if isinstance(protocol, (tuple, list)):
            protocol = protocol[0]
        is_remote = protocol not in (None, "file", "local")

        # Only wrap with filecache if protocol is remote and caching is desired
        if is_remote and cache:
            fs = fsspec.filesystem(
                "filecache",
                target_protocol=fs,
                cache_storage=cache_dir(),
                check_files=True,
                expiry_time=None,
            )

        with fs.open(remote_path) as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

    return model


class FaeModel(nn.Module):
    """
    Base class for all Faeyon model, which allows to save and load the model.
    """
    def save(self, file_name: Optional[str] = None, config_only: bool = False) -> None:
        """
        Save the model to a file.
        """
        name = self.__class__.name
        state = self.state_dict()

        self._arguments
    
        output = {
            "model":{
                "name": name,
                "state": state,
            }
        }

    @classmethod
    def from_config(cls, name: str) -> FaeModel:
        """
        Loads a model from a YAML configuration file.
        """
        cfg, is_builtin = _read_config(name)
        keys = set(cfg.keys())

        if "_model_" in cfg:
            model_cfg = cfg["_model_"]
        else:
            model_cfg = cfg

        parsed_cfg = _parse_config(model_cfg, force_class=False)

        if isinstance(parsed_cfg, nn.Module):
            if not isinstance(parsed_cfg, cls):
                raise ValueError(
                    f"Expected a {cls.__name__} model, but got a {parsed_cfg.__class__.__name__} model."
                )
            return parsed_cfg
        
        # TODO: use `A` to represent an arguments object.
        if isinstance(parsed_cfg, dict):
            return cls(**parsed_cfg)

    def load(self, path: str, strict: bool = True, assign: bool = False) -> None:
        with fsspec.open(path, "rb") as f:
            state_dict = torch.load(f)
            self.load_state_dict(state_dict, strict=strict, assign=assign)

    def init_weights(self, initializer: str) -> None:
        pass

