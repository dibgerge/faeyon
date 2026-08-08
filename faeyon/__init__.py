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
    P,
    R,
    DelayedModule,
    Substitute,
    FaeModule,
    lower,
)

from . import models
from . import metrics
from . import modifiers

# Faek (the nn.Module interception) is NOT enabled automatically: importing faeyon
# must not change the behavior of PyTorch for the rest of the process. Opt in with:
#
#     faek.on()          # explicit, process-wide (call faek.off() to restore)
#
#     with faek:         # scoped: patch nn.Module only while building expressions
#         expr = nn.Linear(10, 5) >> nn.ReLU()
#
# Evaluating / materializing an already-built tree (data | expr, FaeModule, lower)
# does not require faek to be on -- only *building* expressions from nn.Module does.
