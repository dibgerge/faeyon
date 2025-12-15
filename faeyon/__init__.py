from .magic import (
    faek, 
    X, 
    A,
    FDict, 
    FList, 
    FMMap,
    FVar,
    Op,
    Serials,
    Parallels,
    Wire,
    W,
    fae_mode
)

from . import models
from . import metrics

# Enable Faek by default to ensure all features are working as expected
faek.on()
