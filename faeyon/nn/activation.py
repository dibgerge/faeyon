from enum import StrEnum


class Activation(StrEnum):
    ReLU = "relu"
    LeakyReLU = "leaky_relu"
    GELU = "gelu"
    Swish = "swish"
    Mish = "mish"
    HardSwish = "hard_swish"
    HardSigmoid = "hard_sigmoid"
    SELU = "selu"
    ELU = "elu"
    Silu = "silu"
    Sigmoid = "sigmoid"
    Softmax = "softmax"
    Softplus = "softplus"
    Softsign = "softsign"