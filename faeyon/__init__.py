from .magic import (
    faek, 
    X,
    A,
    Input,
    FDict, 
    FList, 
    # FMMap,
    FVar,
    F,
    Chain,
    I,
    DelayedModule,
)

from . import models
from . import metrics
from . import modifiers

# Enable Faek by default to ensure all features are working as expected
faek.on()
