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
    Chain,
    Wire,
    W,
    Modifiers
)

from . import models
from . import metrics

# Enable Faek by default to ensure all features are working as expected
faek.on()
