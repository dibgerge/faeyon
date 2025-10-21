from enum import Flag, auto, StrEnum
from collections import namedtuple


ImageSize = namedtuple("ImageSize", ["height", "width"])


class ClfTask(StrEnum):
    """
    Specifies the type of predictions/targets for classification tasks.

    * BINARY: Predictions are probabilities in [0, 1] and targets are 0 or 1.
    * SPARSE: Predictions are integer values and targets are integer values.
    * CATEGORICAL: Predictions are probabilities in [0, 1] and targets are one-hot encoded.
    * MULTILABEL: Predictions are probabilities in [0, 1] and targets are one-hot encoded.
    """
    BINARY = "binary"
    SPARSE = "sparse"
    CATEGORICAL = "categorical"
    MULTILABEL = "multilabel"


class PeriodUnit(StrEnum):
    EPOCHS = "epochs"
    STEPS = "steps"
    SECONDS = "seconds"


class TrainStage(StrEnum):
    TRAIN = "train"
    VAL = "val"
    PREDICT = "predict"
    TEST = "test"


class TrainStateMode(Flag):
    NONE = auto()
    TRAIN_BEGIN = auto()
    TRAIN_END = auto()
    EPOCH_BEGIN = auto()
    EPOCH_END = auto()
    TRAIN_STEP = auto()
    VAL_BEGIN = auto()
    VAL_END = auto()
    VAL_STEP = auto()
