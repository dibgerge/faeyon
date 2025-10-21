import platform
import os.path
from pathlib import Path


def cache_dir(mkdir: bool = True):
    """ 
    Get the cache directory for Faeyon, which is used to store downloaded models
    and other cached data.
    """
    home = Path.home()

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


