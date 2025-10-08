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
        optimizer=optimizer,
        val_period="1e",
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
    @pytest.mark.parametrize("use_val_data", [True, False])
    def test_fit(self, recipe, train_data, val_data, mocker, use_val_data):
        events = [
            ("on_epoch_begin", 2),
            ("on_train_step_begin", 6),
            ("on_train_step_end", 6),
            ("on_epoch_end", 2),
            ("on_val_begin", 0 if not use_val_data else 2),
            ("on_val_step_begin", 0 if not use_val_data else 6),
            ("on_val_step_end", 0 if not use_val_data else 6),
            ("on_val_end", 0 if not use_val_data else 2),
        ]
        _, expected = zip(*events)
        spies = [mocker.spy(recipe.state, event) for event, _ in events]
        recipe.fit(
            train_data, 
            max_period="2e", 
            val_data=val_data if use_val_data else None
        )
        for spy, count in zip(spies, expected):
            assert spy.call_count == count        
