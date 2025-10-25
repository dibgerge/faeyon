import platform
import os.path
from pathlib import Path
from typing import Optional
import fsspec


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


def get_protocol(name: str) -> Optional[str]:
    """
    Get the protocol for a given name.
    """
    fs, _ = fsspec.core.url_to_fs(name)
    return fs.protocol if hasattr(fs, "protocol") else None


def get_file(name: str, cache: bool = True) -> Path:
    """
    Get the cache directory for a given name.
    """
    protocol = get_protocol(name)

    if protocol is None:
        protocol = "unknown"

    if protocol == "file" or not cache:
        return

    cache_path = cache_dir() 

    return fs.get(name, cache_dir() / name)
