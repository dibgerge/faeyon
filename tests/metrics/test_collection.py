import torch
import pytest
from faeyon.metrics import MetricCollection, Accuracy
from faeyon.metrics.classification import BinaryProbabilityTask


class TestMetricCollection:
    @pytest.fixture
    def metric1(self):
        return Accuracy("accuracy1", thresholds=0.5)
    
    @pytest.fixture
    def metric2(self):
        return Accuracy("accuracy2", thresholds=0.8)

    @pytest.fixture
    def metrics(self, metric1, metric2):
        return MetricCollection([metric1, metric2])

    def test_init(self, metrics):
        assert metrics.name == "metrics"
        assert len(metrics) == 2

    def test_init_with_name(self):
        metric = MetricCollection(name="foo")
        assert metric.name == "foo"
    
    def test_update(self, metrics, metric1, metric2):
        preds = torch.tensor([0.1, 0.6, 0.7, 0.9, 0.3])
        targets = torch.tensor([0, 1, 1, 0, 1])
        metrics.update(preds, targets)
        assert isinstance(metric1.task, BinaryProbabilityTask)
        assert isinstance(metric2.task, BinaryProbabilityTask)

    def test_compute(self, metrics):
        preds = torch.tensor([0.1, 0.6, 0.7, 0.9, 0.3])
        targets = torch.tensor([0, 1, 1, 0, 1])
        metrics.update(preds, targets)
        res = metrics.compute()
        torch.testing.assert_close(res["accuracy1"], torch.tensor([0.6]))
        torch.testing.assert_close(res["accuracy2"], torch.tensor([0.2]))

    def test_duplicate_names(self):
        metric1 = Accuracy("accuracy1", thresholds=0.5)
        metric2 = Accuracy("accuracy1", thresholds=0.8)
        with pytest.raises(ValueError):
            MetricCollection([metric1, metric2])
    
    def test_clone(self, metrics):
        cloned_metrics = metrics.clone()
        assert cloned_metrics.name == metrics.name
        assert len(cloned_metrics) == len(metrics)
        assert cloned_metrics.metrics.keys() == metrics.metrics.keys()
        for name, metric in cloned_metrics.metrics.items():
            assert metric.name == name

    def test_reset(self, metrics):
        preds = torch.tensor([0.1, 0.6, 0.7, 0.9, 0.3])
        targets = torch.tensor([0, 1, 1, 0, 1])
        metrics.update(preds, targets)

        res = metrics.compute()
        torch.testing.assert_close(res["accuracy1"], torch.tensor([0.6]))
        torch.testing.assert_close(res["accuracy2"], torch.tensor([0.2]))

        metrics.reset()

        with pytest.raises(ValueError):
            metrics.compute()

    def test_getitem(self, metrics, metric1, metric2):
        assert metrics["accuracy1"] is metric1
        assert metrics["accuracy2"] is metric2

        with pytest.raises(KeyError):
            metrics["accuracy3"]
