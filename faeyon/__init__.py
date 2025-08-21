from .magic import (
    faek, 
    X, 
    A,
    FDict, 
    FList, 
    FMMap,
    FVar,
    Op,
    Wire,
    W,
)

from . import models

# Enable Faek by default to ensure all features are working as expected
faek.on()
