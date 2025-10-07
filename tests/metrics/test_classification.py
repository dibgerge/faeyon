import torch
import pytest
from faeyon.metrics.classification import ClfMetricBase
from faeyon.metrics import Accuracy

from torch.nn.functional import one_hot


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
        with pytest.raises(ValueError):
            ClfMetricBase(**kwargs)

    def test_init_binary(self):
        """ num_classes 1 or 2 sets task to BINARY"""
        metric1 = ClfMetricBase(num_classes=2)
        assert metric1.num_classes == 1

        metric2 = ClfMetricBase(num_classes=1)
        assert metric2.num_classes == 1

    @pytest.mark.parametrize("thresh, expected", [
        (3, [0.0, 0.5, 1.0]),
        (0.5, [0.5]),
        ([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]),
        (torch.tensor([0.1, 0.5, 0.9]), [0.1, 0.5, 0.9])
    ])
    def test_init_with_thresholds(self, thresh, expected):
        """ Different types of `thresholds` are supported """
        metric = ClfMetricBase(thresholds=thresh)
        torch.testing.assert_close(metric.thresholds,  torch.tensor(expected))

    @pytest.mark.parametrize("kwargs, preds, targets", [
        # 0 Different batch size between preds & targets
        ({}, torch.rand(4, 3), torch.randint(0, 3, (5, 3))), 
        # 1 Preds have wrong number of dimensions
        ({}, torch.rand(4, 3, 2), torch.randint(0, 3, (4, 3))),
        # 2 Targets have wrong number of dimensions
        ({}, torch.rand(4, 3), torch.randint(0, 3, (4, 3, 2))),
        # 3 Targets are not integer values
        ({}, torch.rand(4, 3), torch.rand(4, 3)),
        # 4 Targets are negative for sparse inputs
        ({}, torch.randint(0, 3, (4,)), torch.randint(-3, 0, (4,))),
        
        # 5 Preds of shape (B, C) are not floating point
        ({}, torch.randint(0, 3, (4, 3)), torch.randint(0, 3, (4, 3))),
        # 6 Number of classes mismatch between preds & targets
        ({}, torch.rand(4, 3), one_hot(torch.randint(0, 5, (4,)), num_classes=5)),
        # 7 mismatch between categorical preds & num_classes
        ({"num_classes": 5}, torch.rand(4, 3), one_hot(torch.randint(0, 5, (4,)), num_classes=5)),
        # 8 mismatch between sparse preds & num_classes
        (
            {"num_classes": 3}, 
            torch.randint(3, 5, (4,)), 
            one_hot(torch.randint(0, 3, (4,)), num_classes=5)
        ),
        # 9 mismatch between targets & num_classes
        ({"num_classes": 3}, torch.rand(4, 3), one_hot(torch.randint(0, 5, (4,)), num_classes=5)),
        # 10 sparse targets has more classes than num_classes
        ({"num_classes": 3}, torch.randint(0, 3, (4,)), torch.randint(3, 5, (4,))),

        # 11 Binary task: Targets must be 0 or 1
        ({"thresholds": 0.5}, torch.tensor([0.1, 0.6, 0.7, 0.9]), torch.tensor([0, 1, 2, 0])),
        # 12 Binary task: 2-D targets should not have more than 2 class
        (
            {"thresholds": 0.5}, 
            torch.tensor([0.1, 0.6, 0.7, 0.9]), 
            torch.tensor([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        ),
        # 13 Binary pred inputs require thresholds
        ({}, torch.tensor([0.1, 0.6, 0.7, 0.9]), torch.tensor([0, 1, 1, 0])),
        # 14 Binary pred inputs must be probabilities        
        ({"thresholds": 0.5}, torch.tensor([0.1, 0.6, 0.7, 2.9]), torch.tensor([0, 1, 1, 0])),
        # 15 Binary: Targets must be binary for preds of shape (B, 2)
        (
            {"thresholds": 0.5}, 
            torch.randn((4, 2)), 
            [[0, 1], [0, 1], [0, 0], [0, 1]]
        ),
        # 16 Binary: topk is invalid for preds of shape (B, 2)
        (
            {"topk": 2}, 
            torch.randn((4, 2)), 
            [[0, 1], [0, 1], [0, 1], [0, 1]]
        ),
        # 17 Binary: preds and targets are sparse - bad targets
        ({"num_classes": 1}, torch.randint(0, 2, (4,)), torch.randint(2, 5, (4,))),
        # 18 Binary: preds and targets are sparse - bad preds
        ({"num_classes": 1}, torch.randint(2, 5, (4,)), torch.randint(0, 2, (4,))),
        # 19 Binary: preds sparse, targets is (B, 1) - bad targets
        ({}, torch.randint(0, 2, (4,)), torch.randint(3, 5, (4, 1))),
        # 20 Binary: preds sparse, targets is (B, 2) - bad targets
        ({}, torch.randint(0, 2, (4,)), torch.tensor([[0, 1], [0, 1], [0, 1], [1, 1]])),
        # 21 Binary: preds sparse, targets is (B, 1) - topk cannot be used
        ({"topk": 1}, torch.randint(0, 2, (4,)), torch.randint(0, 2, (4, 1))),
        # 22 Binary: preds sparse, targets is (B, 1) - thresholds cannot be used
        ({"thresholds": 0.5}, torch.randint(0, 2, (4,)), torch.randint(0, 2, (4, 1))),
        # 23 Binary: preds sparse, targets is (B, 1) - preds must be {0, 1}
        ({}, torch.randint(2, 5, (4,)), torch.randint(0, 2, (4, 1))),

        # 24 Sparse predictions should not have thresholds
        (
            {"thresholds": 0.5, "num_classes": 3}, 
            torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))
        ),

        # 25 Sparse predictions should not have topk
        ({"topk": 1, "num_classes": 3}, torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))),

        # 26 If preds and targets are sparse, we should have num_classes
        ({}, torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))),

        # 27 Categorical predictions and sparse targets - targets with wrong # classes
        ({}, torch.rand(4, 3), torch.randint(3, 5, (4,))),

        # 28 Multilabel targets with unspecified multilabel argument
        ({}, torch.rand(4, 3), [[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0]]),

        # 29 Multilabel not supported for preds of shape (B, )
        (
            {"multilabel": True, "thresholds": 0.5}, 
            torch.rand(4), 
            [[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0]]
        ),
        # 30 Multilabel preds must be probabilities
        (
            {"multilabel": True, "thresholds": 0.5}, 
            torch.rand(4, 3) + 2, 
            [[1, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0]]
        ),
        # 31 Multilabel targets must be one-hot encoded
        (
            {"multilabel": True, "thresholds": 0.5}, 
            torch.rand(4, 3), 
            torch.randint(0, 3, (4,))
        ),
        # 32 Multilabel targets cannot be of shape (B, 1)
        (
            {"multilabel": True, "thresholds": 0.5}, 
            torch.rand(4, 3), 
            torch.randint(0, 2, (4, 1))
        ),

        # 33 Preds: (B,) Targets: (B, C) - with topk
        (
            {"topk": 1}, 
            torch.randint(0, 3, (4,)), 
            [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        ),
        # 34 Preds: (B,) Targets: (B, C) - with max_pred > C
        (
            {}, 
            torch.randint(3, 4, (4,)), 
            [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        ),
    ])
    def test_update_errors(self, kwargs, preds, targets):
        metric = ClfMetricBase(**kwargs)

        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        with pytest.raises(ValueError):
            metric.update(preds, targets)

    def test_update_task_mismatch(self):
        """ Make two updates where second update has a different task than first update."""
        metric = ClfMetricBase(thresholds=0.5, average=None)
        metric.update(torch.rand(4,), torch.randint(0, 2, (4,)))
        assert metric.num_classes == 1

        with pytest.raises(ValueError):
            metric.update(torch.rand(4, 3), torch.randint(0, 3, (4,)))

    @pytest.mark.parametrize("kwargs, preds, targets, expected", [
        # 0 Preds: (B,) INT, targets: (B,)
        (
            {"num_classes": 1}, 
            [0, 1, 1, 1], 
            [0, 1, 1, 0],
            [[1, 0], [1, 2]]
        ),
        # 1 Preds: (B,) INT, targets: (B, 1)
        (
            {"num_classes": 1}, 
            [0, 1, 1, 1], 
            [[0], [1], [1], [0]],
            [[1, 0], [1, 2]]
        ),
        # 2 Preds: (B,) INT, targets: (B, 2)
        (
            {}, 
            [0, 1, 1, 1], 
            [[1, 0], [0, 1], [0, 1], [1, 0]],
            [[1, 0], [1, 2]]
        ),
        # 3 Preds: (B,) FLOAT, targets: (B,)
        (
            {"thresholds": 0.5}, 
            [0.2, 0.7, 0.6, 0.8], 
            [0, 1, 1, 0],
            [[[1, 0], [1, 2]]]
        ),
        # 4 Preds: (B,) FLOAT, targets: (B, 1)
        (
            {"thresholds": 0.5}, 
            [0.2, 0.7, 0.6, 0.8], 
            [[0], [1], [1], [0]],
            [[[1, 0], [1, 2]]]
        ),
        # 5 Preds: (B,) FLOAT, targets: (B, 2)
        (
            {"thresholds": 0.5}, 
            [0.2, 0.7, 0.6, 0.8], 
            [[1, 0], [0, 1], [0, 1], [1, 0]],
            [[[1, 0], [1, 2]]]
        ),

        # 6 Preds: (B, 2) FLOAT, targets: (B,)
        (
            {}, 
            [[0.1, -2.0], [0.6, 1.1], [0.7, 1.5], [0.9, 2.0]], 
            [0, 1, 1, 0],
            [[1, 0], [1, 2]]
        ),
        # 7 Preds: (B, 2) FLOAT, targets: (B, 1)
        (
            {}, 
            [[0.1, -2.0], [0.6, 1.1], [0.7, 1.5], [0.9, 2.0]], 
            [[0], [1], [1], [0]],
            [[1, 0], [1, 2]]
        ),
        # 8 Preds: (B, 2) FLOAT, targets: (B, 2)
        (
            {}, 
            [[0.1, -2.0], [0.6, 1.1], [0.7, 1.5], [0.9, 2.0]], 
            [[1, 0], [0, 1], [0, 1], [1, 0]],
            [[1, 0], [1, 2]]
        ),

        # 9 Preds: (B,) SPARSE, targets: (B,) SPARSE
        (
            {"num_classes": 3}, 
            [0, 2, 1, 2, 2],
            [0, 2, 2, 2, 1],
            [[1, 0, 0], [0, 0, 1], [0, 1, 2]]
        ),
        # 10 Preds: (B,) SPARSE, targets: (B, 1) SPARSE
        (
            {"num_classes": 3}, 
            [0, 2, 1, 2, 2],
            [[0], [2], [2], [2], [1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 2]]
        ),
        # 11 Preds: (B,) SPARSE, targets: (B, C) SPARSE
        (
            {}, 
            [0, 2, 1, 2, 2],
            [[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 2]]
        ),
        # 12 Preds: (B, C) FLOAT, targets: (B,) SPARSE
        (
            {}, 
            [
                [0.1, -2.0, 0.0], 
                [0.6, 1.1, 2.0], 
                [0.7, 1.5, 0.0], 
                [0.9, 1.0, 2.0], 
                [-0.9, 0.1, 0.6]
            ],
            [0, 2, 2, 2, 1],
            [[1, 0, 0], [0, 0, 1], [0, 1, 2]]
        ),
        # 13 Preds: (B, C) FLOAT, targets: (B, 1) SPARSE
        (
            {}, 
            [
                [0.1, -2.0, 0.0], 
                [0.6, 1.1, 2.0], 
                [0.7, 1.5, 0.0], 
                [0.9, 1.0, 2.0], 
                [-0.9, 0.1, 0.6]
            ],
            [[0], [2], [2], [2], [1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 2]]
        ),
        # 14 Preds: (B, C) FLOAT, targets: (B,) SPARSE
        (
            {"topk": 2}, 
            [
                [0.1, -2.0, 0.0], 
                [0.6, 1.1, 2.0], 
                [0.7, 1.5, 0.0], 
                [0.9, 1.0, 2.0], 
                [-0.9, 0.1, 0.6]
            ],
            [0, 2, 2, 2, 1],
            [[1, 0, 0], [0, 1, 1], [0, 0, 2]]
        ),
        # 15 Preds: (B, C) FLOAT, targets: (B,) SPARSE
        (
            {"thresholds": 0.35}, 
            [
                [0.6, 0.2, 0.2], 
                [0.1, 0.1, 0.8],
                [0.2, 0.7, 0.1], 
                [0.1, 0.3, 0.6], 
                [0.4, 0.0, 0.6]
            ],
            [0, 2, 2, 2, 1],
            [
                [[[3, 0], [1, 1]]],
                [[[3, 1], [1, 0]]],
                [[[1, 1], [1, 2]]]
            ]
        ),
    ])
    def test_update(self, kwargs, preds, targets, expected):
        preds = torch.tensor(preds)
        targets = torch.tensor(targets)

        if preds.ndim == 1:
            preds_list = [preds, preds.unsqueeze(-1)]
        else:
            preds_list = [preds]

        for preds in preds_list:   
            metric = ClfMetricBase(**kwargs)
            metric.update(preds, targets)
            torch.testing.assert_close(metric._state.to_dense(), torch.tensor(expected))

    @pytest.mark.parametrize("args, kwargs", [
        ([], {"num_classes": 3}),
        (["foo"], {"thresholds": 0.5, "average": "micro"}),
        (["bar", 3], {"average": "micro"}),
        (["baz", 2, 0.5], {"multilabel": True}),
    ])
    def test_clone(self, args, kwargs):
        metric = ClfMetricBase(*args, **kwargs)
        cloned_metric = metric.clone()
        assert cloned_metric.name == metric.name
        assert cloned_metric.multilabel == metric.multilabel
        assert cloned_metric.thresholds == metric.thresholds
        assert cloned_metric.average == metric.average
        assert cloned_metric.num_classes == metric.num_classes
        assert cloned_metric.task == metric.task
        assert cloned_metric is not metric


class TestAccuracy:

    @pytest.mark.parametrize("kwargs, preds, targets, expected", [
        # Sparse task - (C, C) state
        ({"num_classes": 3, "average": "micro"}, [0, 2, 1, 2, 2], [0, 2, 2, 2, 1], 0.6),
        ({"num_classes": 3, "average": "macro"}, [0, 2, 1, 2, 2], [0, 2, 2, 2, 1], 0.55555556),
        ({"num_classes": 3, "average": "weighted"}, [0, 2, 1, 2, 2], [0, 2, 2, 2, 1], 0.6),
        (
            {"num_classes": 3, "average": "none"}, 
            [0, 2, 1, 2, 2], 
            [0, 2, 2, 2, 1], 
            [1.0, 0.0, 0.666667]
        ),

        # Categorical (Binary) task - (T, 2, 2) state
        (
            {"average": "micro", "thresholds": [0.5, 0.8]}, 
            [0.1, 0.6, 0.7, 0.9, 0.3], 
            [0, 1, 1, 0, 1],
            [0.6, 0.2]
        ),
    ])
    def test_compute(self, kwargs, preds, targets, expected):
        preds = torch.tensor(preds)
        targets = torch.tensor(targets)
        metric = Accuracy(**kwargs)
        metric.update(preds, targets)
        res = metric.compute()
        torch.testing.assert_close(res, torch.tensor(expected, dtype=res.dtype))
