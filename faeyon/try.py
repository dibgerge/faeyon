from faeyon import faek
from torch import nn

faek.on()


block = 4 * nn.Linear(10, 10)
print(block)
