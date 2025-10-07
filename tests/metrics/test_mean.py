import torch
import pytest
from faeyon.metrics import MeanMetric


class TestMeanMetric:
    def test_init(self):
        metric = MeanMetric()
        assert metric.name == "meanmetric"
        assert metric.compute() == torch.tensor(0)

    def test_update(self):
        metric = MeanMetric()
        metric.update(torch.tensor([1, 2, 3]))
        metric.update(torch.tensor([4, 5]))
        torch.testing.assert_close(metric.compute(), torch.tensor(3.0))

    def test_update_with_count(self):
        metric = MeanMetric()
        metric.update(torch.tensor(6.0), 3)
        metric.update(torch.tensor([9.0]), 2)
        torch.testing.assert_close(metric.compute(), torch.tensor(3.0))

    def test_update_with_count_extra(self):
        metric = MeanMetric()
        metric.update(torch.tensor([1., 2., 3.]), 3)
        metric.update(torch.tensor([4., 5.]), 2)
        torch.testing.assert_close(metric.compute(), torch.tensor(3.0))

    def test_update_with_count_mismatch(self):
        metric = MeanMetric()
        with pytest.raises(ValueError):
            metric.update(torch.tensor([1., 2., 3.]), 4)

    def test_update_with_count_nonscalar(self):
        metric = MeanMetric()
        with pytest.raises(ValueError):
            metric.update(torch.tensor([1., 2., 3.]), torch.tensor([4, 5]))

    def test_update_with_nd_values(self):
        metric = MeanMetric()
        metric.update(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        torch.testing.assert_close(metric.compute(), torch.tensor(3.5))
