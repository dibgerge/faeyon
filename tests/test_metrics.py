import torch
import pytest
from faeyon import metrics

from torch.nn.functional import one_hot


def clf_test_data(ptask, ttask):
    preds = {
        "binary": torch.tensor([0.1, 0.6, 0.7, 0.9]),
        "sparse": torch.tensor([0, 0, 1, 2]),
        "categorical": torch.tensor([
            [0.2, 0.6, 0.3], 
            [0.4, 0.3, 0.3], 
            [0.8, 0.1, 0.1], 
            [0.1, 0.2, 0.7]
        ]),
        "multilabel": torch.tensor([
            [0.9, 0.6, 0.7], 
            [0.4, 0.2, 0.8], 
            [0.3, 0.2, 0.1], 
            [0.9, 0.2, 0.7]
        ]),
    }

    targets = {
        "binary": torch.tensor([0, 1, 1, 0]),
        "sparse": torch.tensor([1, 0, 0, 2]),
        "categorical": torch.tensor([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1]
        ]),
        "multilabel": torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0]
        ]),
    }
    return preds[ptask], targets[ttask]


class TestClfMetricBase:

    @pytest.mark.parametrize("kwargs", [
        # Cannot use both `thresholds` and `topk`
        {"thresholds": 0.5, "topk": 1},
        # Need `thresholds` for multilabel predictions
        {"multilabel": True},
        # `num_classes` must be an integer
        {"num_classes": 1.0},
        # `num_classes` must be greater than 0
        {"num_classes": -2},
        # `thresholds` must be a float, list/tuple of floats, or tensor of floats
        {"thresholds": "a"},
        # `thresholds` must be between 0 and 1
        {"thresholds": [0.5, 2.4]},
    ])
    def test_init_errors(self, kwargs):
        # Cannot use both `thresholds` and `topk`
        print(kwargs)
        with pytest.raises(ValueError):
            metrics.ClfMetricBase(**kwargs)

    def test_init_binary(self):
        """ num_classes 1 or 2 sets task to BINARY"""
        metric1 = metrics.ClfMetricBase(num_classes=2)
        assert metric1.task == metrics.ClfTask.BINARY
        assert metric1.num_classes == 2

        metric2 = metrics.ClfMetricBase(num_classes=1)
        assert metric2.task == metrics.ClfTask.BINARY
        assert metric2.num_classes == 2

    @pytest.mark.parametrize("thresh, expected", [
        (3, [0.0, 0.5, 1.0]),
        (0.5, [0.5]),
        ([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]),
        (torch.tensor([0.1, 0.5, 0.9]), [0.1, 0.5, 0.9])
    ])
    def test_init_with_thresholds(self, thresh, expected):
        """ Different types of `thresholds` are supported """
        metric = metrics.ClfMetricBase(thresholds=thresh)
        torch.testing.assert_close(metric.thresholds,  torch.tensor(expected))

    @pytest.mark.parametrize("kwargs, preds, targets", [
        # Different batch size between preds & targets
        ({}, torch.rand(4, 3), torch.randint(0, 3, (5, 3))), 
        # Preds have wrong number of dimensions
        ({}, torch.rand(4, 3, 2), torch.randint(0, 3, (4, 3))),
        # Targets have wrong number of dimensions
        ({}, torch.rand(4, 3), torch.randint(0, 3, (4, 3, 2))),
        # Targets are not integer values
        ({}, torch.rand(4, 3), torch.rand(4, 3)),
        #T argets are negative for sparse inputs
        ({}, torch.randint(0, 3, (4,)), torch.randint(-3, 0, (4,))),
        # Preds of shape (B, C) are not probabilities
        ({}, torch.randint(0, 3, (4, 3)), torch.randint(0, 3, (4, 3))),
        # Number of classes mismatch between preds & targets
        ({}, torch.rand(4, 3), one_hot(torch.randint(0, 5, (4,)), num_classes=5)),
        # mismatch between categorical preds & num_classes
        ({"num_classes": 5}, torch.rand(4, 3), one_hot(torch.randint(0, 5, (4,)), num_classes=5)),
        # mismatch between sparse preds & num_classes
        (
            {"num_classes": 3}, 
            torch.randint(3, 5, (4,)), 
            one_hot(torch.randint(0, 3, (4,)), num_classes=5)
        ),
        # mismatch between targets & num_classes
        ({"num_classes": 3}, torch.rand(4, 3), one_hot(torch.randint(0, 5, (4,)), num_classes=5)),
        # sparse targets has more classes than num_classes
        ({"num_classes": 3}, torch.randint(0, 3, (4,)), torch.randint(3, 5, (4,))),
        # Multilabel targets with unspecified multilabel argument
        ({}, torch.rand(4, 3), torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0]])),
        # Average input not supported for binary predictions
        (
            {"average": "weighted", "thresholds": 0.5}, 
            torch.tensor([0.1, 0.6, 0.7, 0.9]), 
            torch.tensor([0, 1, 1, 0])
        ),
        # Binary task: Targets must be 0 or 1
        ({"thresholds": 0.5}, torch.tensor([0.1, 0.6, 0.7, 0.9]), torch.tensor([0, 1, 2, 0])),
        # Binary task: 2-D targets must be of shape (B, 2) or (B, 1)
        (
            {"thresholds": 0.5}, 
            torch.tensor([0.1, 0.6, 0.7, 0.9]), 
            torch.tensor([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        ),
        # Binary pred inputs require thresholds
        ({}, torch.tensor([0.1, 0.6, 0.7, 0.9]), torch.tensor([0, 1, 1, 0])),
        # Binary pred inputs must be probabilities
        ({"thresholds": 0.5}, torch.tensor([0.1, 0.6, 0.7, 2.9]), torch.tensor([0, 1, 1, 0])),
        # Sparse predictions should not have thresholds
        (
            {"thresholds": 0.5, "num_classes": 3}, 
            torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))
        ),
        # Sparse predictions should not have topk
        ({"topk": 1, "num_classes": 3}, torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))),
        # If preds and targets are sparse, we should have num_classes
        ({}, torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))),
    ])
    def test_update_errors(self, kwargs, preds, targets):
        metric = metrics.ClfMetricBase(**kwargs)
        with pytest.raises(ValueError):
            metric.update(preds, targets)


    @pytest.mark.parametrize("preds, targets", [
        clf_test_data("binary", "binary"),
    ])
    def test_binary_update(self, preds, targets):
        metric = metrics.ClfMetricBase(thresholds=0.5, average=None)
        expected = torch.tensor([[
            [1, 0], 
            [1, 2]
        ]])            
        metric.update(preds, targets)
        torch.testing.assert_close(metric._state.to_dense(), expected)

