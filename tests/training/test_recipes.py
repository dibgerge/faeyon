import pytest

import torch
from torch import nn
from faeyon.training import Recipe, FaeOptimizer
from torch.utils.data import DataLoader
from faeyon.datasets import FaeDataset


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(5, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        return x >> self.hidden >> self.output


@pytest.fixture
def recipe():
    model = Model()
    optimizer = FaeOptimizer(name="SGD", lr=0.01)
    return Recipe(
        model=model, 
        loss=nn.BCEWithLogitsLoss(), 
        optimizer=optimizer
    )


@pytest.fixture(scope="session")
def train_data():
    data = torch.randn(10, 5)
    targets = torch.randint(0, 2, (10, 1), dtype=torch.float)
    dataset = FaeDataset(data, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return loader


@pytest.fixture(scope="session")
def val_data():
    data = torch.randn(10, 5)
    targets = torch.randint(0, 2, (10, 1), dtype=torch.float)
    dataset = FaeDataset(data, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return loader


class TestRecipe:
    def test_fit(self, recipe, train_data, val_data):
        recipe.fit(train_data, max_period="2e")
