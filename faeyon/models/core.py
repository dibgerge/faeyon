from __future__ import annotations
import os
from pickle import UnpicklingError
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
from faeyon import A


def is_yaml(file_name: str) -> bool:
    """ 
    Checks if the file extension is for a YAML file.
    """
    return os.path.splitext(file_name)[1].lower() in [".yml", ".yaml"]


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

    def read(self, name: str, trust_code: bool = False) -> dict[str, Any]:
        loader = yaml.Loader if trust_code else yaml.SafeLoader
        with self.open(name) as f:
            return yaml.load(f, Loader=loader)

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


def _parse_dict(cfg: dict[str, Any]) -> Any:
    try:
        target = cfg.pop("_target_")
    except KeyError:
        return {k: _parse_config(v) for k, v in cfg.items()}

    args, kwargs, other = [], {}, {}
    target = None
    for k, v in cfg.items():
        match k:
            case "_args_":
                if not isinstance(v, list):
                    raise ValueError("'_args_' must be a list of arguments.")
                args = [_parse_config(arg) for arg in v]

            case "_kwargs_":
                if not isinstance(v, dict):
                    raise ValueError("'_kwargs_' must be a dict of keyword arguments.")
                for kwarg, kwval in v.items():
                    kwargs[kwarg] = _parse_config(kwval)

            case "_target_":
                target = v

            # _meta_ is not used by the parser
            case "_meta_":
                pass

            case _:
                other[k] = _parse_config(v)

    if target is not None:
        mod_name, cls_name = target.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        
        # All remaining keys are treated as keyword arguments
        if other:
            kwargs.update(other)

        return cls(*args, **kwargs) 
    else:
        if args or kwargs:
            return A(*args, **kwargs, **other)
        
        return other


def _parse_config(cfg: Any, force_class: bool = False) -> Any:
    """
    Parse a configuration section recursively and return loaded objects. 
    """
    if force_class:
        if not isinstance(cfg, dict) or "_target_" not in cfg:
            raise ValueError("Config must be a dict with a _target_ key.")

    match cfg:
        case dict():
            return _parse_dict(cfg)
        case list():
            return [_parse_config(item) for item in cfg]
        case _:
            return cfg


def _read_config(name: str, trust_code: bool = False, **kwargs: Any) -> tuple[dict[str, Any], bool]:
    """
    A given name can either be a local path, a remote URL/filesystem path, or a file in the 
    package configs. The lookup logic is as follows:
    * If path is normalized (no . or .. segments, or trailing slash/protocol), look in current 
      working directory. If not found, look in package configs.
    * Otherwise lookup with fsspec.

    Returns a configuration file and a boolean indicating if the file is a builtin config.
    """
    # loader = yaml.Loader if trust_code else yaml.SafeLoader

    # try:
    #     with fsspec.open(name, "r", **kwargs) as f:
    #         try:
    #             data = torch.load(f)
    #         except UnpicklingError as e:
    #             data = yaml.load(f, Loader=loader)
    # except FileNotFoundError as e:
    #     config = BuiltinConfigs()
        
    #     if not config.valid(name):
    #         raise e

    #     return config.read(name)


def _read_yaml(name: str, trust_code: bool = False, **kwargs: Any) -> tuple[Any, Optional[str]]:
    loader = yaml.Loader if trust_code else yaml.SafeLoader

    try:
        with fsspec.open(name, "r", **kwargs) as f:
            data = yaml.load(f, Loader=loader)
    except FileNotFoundError as e:
        config = BuiltinConfigs()
        if not config.valid(name):
            raise e
        data = config.read(name)

    output = _parse_config(data)

    if "_meta_" in data:
        state_file = data["_meta_"].get("state_file")
    else:
        state_file = None

    return output, state_file



def load(
    name: str | nn.Module | type[nn.Module],
    load_state: bool | str = True,
    state_file: str | None = None,
    /,
    cache: bool = True,
    trust_code: bool = False,
    **kwargs: Any
) -> nn.Module:
    """
    Loads a model from a YAML configuration file or a pytorch serialized file. Can also be a 
    builtin model configuration, which is a YAML file in the package configs.

    Usages
    load("model.yaml", "weights_path.pt") -- should contain the state dict
    load("model.yaml", True/False) -- path to the weights file in yaml if True
    load("model.pt", True/False) -- should contain both {state, config} keys
    load(model, "weights_path.pt") -- should only contain the state dict
    load(ModelClass, "model.pt"/model.yaml) -- should only contain the state dict
    """
    loader = yaml.Loader if trust_code else yaml.SafeLoader

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

    try:
        with fs.open(name, "r", **kwargs) as f:
            try:
                data = torch.load(f)

                config = data.get("config")
                state = data.get("state")

                model = _parse_config(config, force_class=True)

                if not isinstance(load_state, str) and load_state:
                    model.load_state_dict(state)
                return model

            except UnpicklingError as e:
                data = yaml.load(f, Loader=loader)
    except FileNotFoundError as e:
        config = BuiltinConfigs()
        
        if not config.valid(name):
            raise e

        return config.read(name)






    dumper = yaml.Dumper if trust_code else yaml.SafeDumper
    if is_yaml(name):
        with fsspec.open(name, "r", **kwargs) as f:
            cfg = yaml.load(f, Loader=dumper)


    else:
    cfg, is_builtin = _read_config(name)
    keys = set(cfg.keys())

    model = _parse_config(model_cfg, force_class=True)

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
    def save(
        self, 
        file_name: Optional[str] = None, 
        save_state: bool | str = True, 
        trust_code: bool = False
    ) -> Optional[dict[str, Any]]:
        """
        Save the model to a file, including information to load the model later.

        Parameters
        ----------
        file_name : str, optional
            If the file name is not given, then the model saved state and configuration will be 
            returned as a string.

            If file_name extension is .yml or .yaml, then the model configuration only will be saved 
            to the file. Otherwise, use `torch.save` to save the model state (if requested) and 
            configuration to the file.
        
        save_state : bool or str, optional
            If `True`, then the model state will be saved to the file if it is a pytorch 
            serialized file. If file_name is yaml, then the model state will be saved as a file 
            with same name but with .pt extension in the same directory.
            
            A string is allowed only if file_name is yaml, in which case the string should be the 
            name of the file to save the model state to.

        trust_code : bool, optional
            If `True`, then the code will be trusted and the model configuration can saved 
            unknown objects.If `False`, then the model configuration will be saved as a YAML string
            using safe_dump.
        """
        if file_name is not None:
            base_file, ext = os.path.splitext(file_name)
            is_yaml =  ext.lower() in [".yml", ".yaml"]
        else:
            is_yaml = False
            base_file = None

        if not is_yaml and isinstance(save_state, str):
            raise ValueError(
                "If `file_name` does have have a yaml extension, then save_state must be a boolean."
            )
                
        target = f"{self.__class__.__module__}.{self.__class__.__name__}"

        args = []
        for arg in self._arguments.args:
            if isinstance(arg, nn.Module):
                args.append(arg.save())
            else:
                args.append(arg)

        kwargs = {}
        for k, v in self._arguments.kwargs.items():
            if isinstance(v, nn.Module):
                kwargs[k] = v.save()
            else:
                kwargs[k] = v
        
        config = {
            "_target_": target,
            "_args_": args,
            "_kwargs_": kwargs,
        }

        dumper = yaml.Dumper if trust_code else yaml.SafeDumper

        if is_yaml:
            if isinstance(save_state, str):
                config["_state_"] = save_state
            elif save_state:
                config["_state_"] = f"{base_file}.pt"
            
            with fsspec.open(file_name, "w") as f:
                yaml.dump(config, f, Dumper=dumper)

            with fsspec.open(config["_state_"], "wb") as f:
                torch.save(self.state_dict(), f)

            return None

        else:
            output = {
                "config": yaml.dump(config, Dumper=dumper),
                "state": self.state_dict(),   
            }

            if file_name is not None:
                with fsspec.open(file_name, "wb") as f:
                    torch.save(output, f)

            return output


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

