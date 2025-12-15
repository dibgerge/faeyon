from __future__ import annotations
import io
import os
import torch
import yaml
import fsspec
import importlib
import platform

from os import PathLike
from torch import nn
from importlib import resources
from pathlib import Path
from typing import Optional, Generator, Any, IO
from faeyon import A


def cache_dir(mkdir: bool = True) -> Path:
    """ 
    Get the cache directory for Faeyon, which is used to store downloaded models
    and other cached data.
    """
    home = Path.home()

    try:
        cache_dir = Path(os.environ["FAEYON_CACHE_DIR"])
    except KeyError:
        match platform.system():
            case "Windows":
                base = Path(os.getenv("LOCALAPPDATA", home / "AppData" / "Local"))
            case "Darwin":
                base = home / "Library" / "Caches"
            case _:
                base = Path(os.getenv("XDG_CACHE_HOME", home / ".cache"))

        cache_dir = base / "faeyon"

    if mkdir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


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
        self.traversable = resources.files(__package__).joinpath("models/configs")

    def open(self, name: str) -> IO[str]:
        if not self.valid(name):
            raise ValueError(
                f"Invalid builtin config path: {name}. The path should not be an absolute path, "
                f"contain . or .. sections, or start with a protocol like protocol://."
            )
        return self.traversable.joinpath(f"{name}.yaml").open("r")

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
        if not isinstance(name, str):
            return False
        file_name = Path(name)
        has_protocol = "://" in name
        has_dots = any(part == ".." for part in file_name.parts)
        return not (file_name.is_absolute() or has_dots or has_protocol or file_name.suffix)


def _parse_dict(cfg: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> Any:
    try:
        target = cfg.pop("_target_")
    except KeyError:
        return {k: _parse_config(v) for k, v in cfg.items()}

    args, kwargs, other = [], {}, {}
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

    has_args = args or kwargs

    # All remaining keys are treated as keyword arguments
    if other:
        kwargs.update(other)

    if overrides is not None:
        for k, v in overrides.items():
            kwargs[k] = v

    if target is not None:
        mod_name, cls_name = target.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)        
        return cls(*args, **kwargs)
    else:
        if has_args:
            return A(*args, **kwargs)
        return kwargs


def _parse_config(cfg: Any, overrides: Optional[dict[str, Any]] = None, force_class: bool = False) -> Any:
    """
    Parse a configuration section recursively and return loaded objects. 
    """
    if force_class:
        if not isinstance(cfg, dict) or "_target_" not in cfg:
            raise ValueError("Config must be a dict with a _target_ key.")

    match cfg:
        case dict():
            return _parse_dict(cfg, overrides)
        case list():
            return [_parse_config(item) for item in cfg]
        case _:
            return cfg


def _read_yaml(
    name: str | PathLike | io.IOBase,
    trust_code: bool = False, 
    **kwargs: Any
) -> tuple[Any, Optional[dict[str, Any]]]:
    loader = yaml.Loader if trust_code else yaml.SafeLoader

    if isinstance(name, str | PathLike):
        try:
            with fsspec.open(name, "r", **kwargs) as f:
                data = yaml.load(f, Loader=loader)
        except FileNotFoundError as e:
            config = BuiltinConfigs()
            if isinstance(name, str) and config.valid(name):
                data = config.read(name)
            else:
                raise e
            
    elif isinstance(name, io.IOBase):
        data = yaml.load(name, Loader=loader)
    else:
        raise ValueError(
            f"Expected a string | PathLike | IOBase, but got a {name.__class__.__name__}."
        )

    output = _parse_config(data)
    return output, data.get("_meta_")


def _read_pt(
    name: str | PathLike[str], 
    cache: bool = True, 
    **kwargs: Any
) -> dict[str, Any]:
    fs, _ = fsspec.core.url_to_fs(name, **kwargs)

    protocol = getattr(fs, "protocol", None)
    if isinstance(protocol, (tuple, list)):
        protocol = protocol[0]
    is_remote = protocol not in (None, "file", "local")

    # Only wrap with filecache if protocol is remote and caching is desired
    if is_remote and cache:
        fs = fsspec.filesystem(
            "filecache",
            fs=fs,
            cache_storage=str(cache_dir()),
            check_files=True,
            expiry_time=None,
        )

    with fs.open(name, "rb", **kwargs) as f:
        data = torch.load(f)

    return data


def _load_from_files(
    config_file: str | PathLike[str],
    load_state: bool | str | PathLike[str] = True, 
    trust_code: bool = False,
    cache: bool = True,
    module: Optional[type[nn.Module]] = None,
    **kwargs: Any
) -> nn.Module:
    """
    The precedence order for loading the state is:
    1. load_state argument given as a string or pathlike object
    2. state saved in the .pt file
    3. state_file name in the YAML config (can be its own file, or inside .pt file)
    """
   
    try:
        parsed, meta = _read_yaml(config_file, trust_code, **kwargs)
        state = None

    except (yaml.YAMLError, UnicodeError):
        data = _read_pt(config_file, cache, **kwargs)

        # Read the state from the pt file
        parsed, meta = _read_yaml(io.StringIO(data["_config_"]))
        state = data.get("_state_")
    
    if isinstance(parsed, A):
        if module is None:
            raise ValueError(f"Expected a model, but got a arguments object {parsed}.")
        
        model = parsed.call(module)
    elif isinstance(parsed, nn.Module):
        model = parsed
    else:
        raise ValueError(f"Expected a model, but got a {parsed.__class__.__name__}.")

    if load_state:

        if meta is not None:
            state_file = meta.get("state")
        else:
            state_file = None

        # Based on file loading precedence, load the state
        if isinstance(load_state, str | PathLike):
            state = _read_pt(load_state, cache, **kwargs)
        elif state is None and state_file is not None: 
            state = _read_pt(state_file, cache, **kwargs)
        elif state is None:
            raise ValueError(
                "No state found in the yaml file or the .pt file. Please provide a path for the "
                "state file, or update the YAML file to include the state file path."
            )

    if state is not None:
        model.load_state_dict(state)

    return model


def load(
    name: str | PathLike | nn.Module,
    load_state: bool | str | PathLike = True,
    module: Optional[type[nn.Module]] = None,
    cache: bool = True,
    trust_code: bool = False,
    overrides: Optional[dict[str, Any]] = None,
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
    if isinstance(name, str | PathLike):
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

        if "_state_" in state:
            state = state["_state_"]

        name.load_state_dict(state)
        return name
    else:
        raise ValueError(
            f"`name` expected a string or `nn.Module` instance, but got a "
            f" {name.__class__.__name__}."
        )


def save(
    model: nn.Module,
    file_name: Optional[str | PathLike] = None, 
    save_state: bool | str | PathLike = True, 
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

    if not is_yaml and not isinstance(save_state, bool):
        raise ValueError("If `file_name` is not a YAML file, then `save_state` must be a boolean.")
            
    target = f"{model.__class__.__module__}.{model.__class__.__name__}"

    args = []
    for arg in model._arguments.args:
        if isinstance(arg, nn.Module):
            args.append(arg.save(save_state=False, trust_code=trust_code)["_config_"])
        else:
            args.append(arg)

    kwargs = {}
    for k, v in model._arguments.kwargs.items():
        if isinstance(v, nn.Module):
            kwargs[k] = v.save(save_state=False, trust_code=trust_code)["_config_"]
        else:
            kwargs[k] = v

    config = {
        "_target_": target,
        "_meta_": {}
    }
    if args:
        config["_args_"] = args
        config["_kwargs_"] = kwargs
    else:
        config.update(kwargs)

    dumper = yaml.Dumper if trust_code else yaml.SafeDumper
    if is_yaml:
        if not isinstance(save_state, bool):
            config["_meta_"]["state"] = str(save_state)
        elif save_state:
            config["_meta_"]["state"] = f"{base_file}.pt"

        with fsspec.open(file_name, "w") as f:
            yaml.dump(config, f, Dumper=dumper)

        if state_file := config["_meta_"].get("state"):
            with fsspec.open(state_file, "wb") as f:
                torch.save(model.state_dict(), f)

        return None

    else:
        output = {"_config_": config}

        if save_state:
            output["_state_"] = model.state_dict()

        if file_name is not None:
            output["_config_"] = yaml.dump(output["_config_"], Dumper=dumper)
            with fsspec.open(file_name, "wb") as f:
                torch.save(output, f)

        return output
