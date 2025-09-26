import torch
import pytest
from faeyon import metrics


class TestClfMetricBase:
    def test_init(self):
        metric = metrics.ClfMetricBase(thresholds=0.5, average=None)

        preds = torch.rand(4, 3)
        targets = torch.randint(0, 2, (4, 3))

        print(preds)
        print(targets)
        print(metric.update(preds, targets))
