from .magic import (
    faek, 
    X,
    A,
    Input,
    I,
    FDict, 
    FList, 
    FMMap,
    FVar,
    F,
    Serials,
    Parallels,
    Wire,
    W
)

from . import models
from . import metrics

# Enable Faek by default to ensure all features are working as expected
faek.on()
