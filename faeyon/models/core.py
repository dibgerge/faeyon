from __future__ import annotations
import io
import os
import torch
import yaml
import fsspec
import importlib
from torch import nn
from importlib import resources
from pathlib import Path
from typing import Optional, Generator, Any
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


def _read_yaml(
    name: str | io.IOBase, 
    trust_code: bool = False, 
    **kwargs: Any
) -> tuple[Any, Optional[str]]:
    loader = yaml.Loader if trust_code else yaml.SafeLoader

    if isinstance(name, str):
        try:
            with fsspec.open(name, "r", **kwargs) as f:
                data = yaml.load(f, Loader=loader)
        except FileNotFoundError as e:
            config = BuiltinConfigs()
            if not config.valid(name):
                raise e
            data = config.read(name)
    elif isinstance(name, io.IOBase):
        data = yaml.load(name, Loader=loader)

    output = _parse_config(data)

    if "_meta_" in data:
        state_file = data["_meta_"].get("state_file")
    else:
        state_file = None

    return output, state_file


def _read_pt(
    name: str, 
    cache: bool = True, 
    **kwargs: Any
) -> dict[str, Any]:
    fs, remote_path = fsspec.core.url_to_fs(name, **kwargs)
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

    with fs.open(name, "rb", **kwargs) as f:
        data = torch.load(f)

    return data


def _load_from_files(
    config_file: str, 
    load_state: bool | str = True, 
    trust_code: bool = False,
    cache: bool = True,
    module: Optional[type[nn.Module]] = None,
    **kwargs: Any
) -> nn.Module:
   
    try:
        parsed, state_file = _read_yaml(config_file, trust_code, **kwargs)
        state = None

    except yaml.YAMLError as e:
        data = _read_pt(config_file, cache, **kwargs)

        # Read the state from the pt file
        parsed, state_file = _read_yaml(io.StringIO(data["_config_"]))
        state = data["_state_"]
    
    if isinstance(parsed, A):
        if module is None:
            raise ValueError(f"Expected a model, but got a {model.__class__.__name__}.")
        
        model = parsed.call(module)
    elif isinstance(parsed, nn.Module):
        model = parsed
    else:
        raise ValueError(f"Expected a model, but got a {parsed.__class__.__name__}.")

    # Precendence over the state file in the yaml file, if any
    if isinstance(load_state, str):
        state_file = load_state

    if load_state:
        if state_file is None:
            raise ValueError(
                "No state file found in the yaml file. Please provide a path for the state "
                "file, or update the YAML file to include the state file path."
            )
        state = _read_pt(state_file, cache, **kwargs)

    if state is not None:
        model.load_state_dict(state)

    return model


def load(
    name: str | nn.Module,
    load_state: bool | str = True,
    module: Optional[type[nn.Module]] = None,
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
    if isinstance(name, str):
        return _load_from_files(
            config_file=name, 
            load_state=load_state,
            module=module,
            cache=cache, 
            trust_code=trust_code, 
            **kwargs
        )

    elif isinstance(name, nn.Module):
        if not isinstance(load_state, str):
            raise ValueError("`load_state` must be a string if `name` is a model.")
    
        if module is not None:
            raise ValueError("`module` is not supported when `name` is a model instance.")

        state = _read_pt(load_state, cache, **kwargs)
        name.load_state_dict(state)
        return name
    else:
        raise ValueError(
            f"`name` expected a string or `nn.Module` instance, but got a "
            f" {name.__class__.__name__}."
        )


class FaeModel(nn.Module):
    """
    Base class for all Faeyon models, which allows to save and load the model.
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
            "_meta_": {}
        }

        dumper = yaml.Dumper if trust_code else yaml.SafeDumper

        if is_yaml:
            if isinstance(save_state, str):
                config["_meta_"]["state_file"] = save_state
            elif save_state:
                config["_meta_"]["state_file"] = f"{base_file}.pt"

            with fsspec.open(file_name, "w") as f:
                yaml.dump(config, f, Dumper=dumper)

            state_file = config["_meta_"].get("state_file")

            if state_file is not None:
                with fsspec.open(state_file, "wb") as f:
                    torch.save(self.state_dict(), f)

            return None

        else:
            output = {
                "_config_": yaml.dump(config, Dumper=dumper),
                "_state_": self.state_dict(),   
            }

            if file_name is not None:
                with fsspec.open(file_name, "wb") as f:
                    torch.save(output, f)

            return output
