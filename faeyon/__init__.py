from .magic import (
    faek, 
    X, 
    FaeArgs,
    FaeDict, 
    FaeList, 
    FaeMultiMap,
    FaeVar,
    Op,
    Wire,
    Wiring,
)

from . import models

# Enable Faek by default to ensure all features are working as expected
faek.on()
